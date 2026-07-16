//! Directory scanning and auto-compaction helpers.

use crate::{
    compactor::{compact_session, parse_session_text},
    store::Store,
    CompactionResult,
};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Scan a directory for JSON/JSONL session files, import any that are new or
/// changed, compact them, and persist the results.
///
/// # Errors
///
/// Returns an error if a database operation fails catastrophically. Individual
/// file failures are logged and skipped.
pub async fn scan_and_compact(
    store: &Store,
    session_dir: &Path,
    max_turns: usize,
    archive_age_hours: u64,
) -> Result<Vec<CompactionResult>> {
    if !session_dir.exists() {
        return Ok(Vec::new());
    }

    let entries = collect_session_files(session_dir)?;
    let mut results = Vec::with_capacity(entries.len());

    for path in entries {
        let session_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let text = match tokio::fs::read_to_string(&path).await {
            Ok(t) => t,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "failed to read session file");
                continue;
            }
        };

        let mut session = match parse_session_text(&session_id, &text) {
            Ok(s) => s,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "failed to parse session file");
                continue;
            }
        };

        if let Some(existing) = store.load_session(&session_id).await? {
            if existing.last_compacted_at.is_some() && existing.len() >= session.len() {
                debug!(session_id = %session_id, "session unchanged since last compaction");
                continue;
            }
        }

        let result = compact_session(&mut session, max_turns, archive_age_hours)
            .map_err(|e| anyhow::anyhow!("compaction failed for {session_id}: {e}"))?;
        store.save_session(&session).await?;
        store.save_compaction_result(&result).await?;
        info!(
            session_id = %session_id,
            path = %path.display(),
            duplicates_removed = result.metrics.duplicates_removed,
            continuations_collapsed = result.metrics.continuation_turns_collapsed,
            turns_archived = result.metrics.turns_archived,
            "compacted session"
        );
        results.push(result);
    }

    Ok(results)
}

fn collect_session_files(session_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries: Vec<PathBuf> = walkdir::WalkDir::new(session_dir)
        .max_depth(3)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            let p = e.path();
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
            p.is_file() && (ext.eq_ignore_ascii_case("json") || ext.eq_ignore_ascii_case("jsonl"))
        })
        .map(|e| e.path().to_path_buf())
        .collect();
    entries.sort();
    Ok(entries)
}

/// Write a sample session file for testing or demonstration.
///
/// # Errors
///
/// Returns an error if the directory or file cannot be created.
pub async fn write_sample_session(session_dir: &Path, session_id: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(session_dir)
        .with_context(|| format!("failed to create session dir {}", session_dir.display()))?;

    let path = session_dir.join(format!("{session_id}.json"));
    let sample = serde_json::json!([
        {
            "id": "t1",
            "role": "user",
            "content": "continue",
            "created_at": "2024-01-01T00:00:00Z"
        },
        {
            "id": "t2",
            "role": "user",
            "content": "continue",
            "created_at": "2024-01-01T00:00:01Z"
        },
        {
            "id": "t3",
            "role": "assistant",
            "content": "ok",
            "created_at": "2024-01-01T00:00:02Z",
            "tool_calls": [
                { "name": "Read", "arguments": "{}", "output": "dup" }
            ]
        },
        {
            "id": "t4",
            "role": "assistant",
            "content": "ok",
            "created_at": "2024-01-01T00:00:03Z",
            "tool_calls": [
                { "name": "Read", "arguments": "{}", "output": "dup" }
            ]
        }
    ]);

    tokio::fs::write(&path, serde_json::to_string_pretty(&sample)?)
        .await
        .with_context(|| format!("failed to write sample session {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    #[tokio::test]
    async fn scan_and_compact_finds_json_file() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("test.db");
        let session_dir = dir.path().join("sessions");
        let store = Store::open(&db).await.unwrap();

        write_sample_session(&session_dir, "sample").await.unwrap();

        let results = scan_and_compact(&store, &session_dir, 10, 24)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].metrics.continuation_turns_collapsed > 0);
    }
}
