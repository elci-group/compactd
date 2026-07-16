//! SQLite persistence layer for compactd.

use crate::{CompactionMetrics, CompactionResult, Session, SessionId, Turn, TurnId};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Persistent store backed by SQLite.
#[derive(Debug, Clone)]
pub struct Store {
    path: PathBuf,
    conn: Arc<Mutex<Connection>>,
}

impl Store {
    /// Open (or create) a SQLite store at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or initialized.
    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create database directory {}", parent.display())
            })?;
        }
        let conn = Connection::open(&path)
            .with_context(|| format!("failed to open database {}", path.display()))?;
        let store = Self {
            path,
            conn: Arc::new(Mutex::new(conn)),
        };
        store.init_schema().await?;
        info!(path = %store.path.display(), "store opened");
        Ok(store)
    }

    /// Open an in-memory store for tests.
    ///
    /// # Errors
    ///
    /// Returns an error if the in-memory database cannot be initialized.
    pub async fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("failed to open in-memory database")?;
        let store = Self {
            path: PathBuf::from(":memory:"),
            conn: Arc::new(Mutex::new(conn)),
        };
        store.init_schema().await?;
        Ok(store)
    }

    async fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_compacted_at TEXT,
                turn_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls TEXT,
                created_at TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_turns_archived ON turns(archived);

            CREATE TABLE IF NOT EXISTS compaction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                duplicates_removed INTEGER NOT NULL DEFAULT 0,
                continuation_turns_collapsed INTEGER NOT NULL DEFAULT 0,
                turns_archived INTEGER NOT NULL DEFAULT 0,
                token_bytes_saved INTEGER NOT NULL DEFAULT 0,
                turns_remaining INTEGER NOT NULL DEFAULT 0,
                compacted_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_compaction_session
                ON compaction_results(session_id);
            "#,
        )
        .context("failed to initialize schema")?;
        Ok(())
    }

    /// Persist a session, replacing any existing turns for that session.
    ///
    /// # Errors
    ///
    /// Returns an error if the database write fails.
    pub async fn save_session(&self, session: &Session) -> Result<()> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction().context("failed to begin transaction")?;

        tx.execute(
            r#"
            INSERT INTO sessions (id, created_at, last_compacted_at, turn_count)
            VALUES (?1, ?2, ?3, ?4)
            ON CONFLICT(id) DO UPDATE SET
                last_compacted_at = excluded.last_compacted_at,
                turn_count = excluded.turn_count
            "#,
            params![
                session.id,
                session.created_at.to_rfc3339(),
                session.last_compacted_at.map(|t| t.to_rfc3339()),
                session.len() as i64
            ],
        )
        .context("failed to save session")?;

        tx.execute(
            "DELETE FROM turns WHERE session_id = ?1 AND archived = 0",
            params![session.id],
        )
        .context("failed to clear live turns")?;

        for turn in &session.turns {
            Self::insert_turn_tx(&tx, &session.id, turn, false)?;
        }

        tx.commit().context("failed to commit session save")?;
        debug!(session_id = %session.id, "saved session");
        Ok(())
    }

    fn insert_turn_tx(
        tx: &rusqlite::Transaction<'_>,
        session_id: &SessionId,
        turn: &Turn,
        archived: bool,
    ) -> Result<()> {
        let tool_calls =
            serde_json::to_string(&turn.tool_calls).context("failed to serialize tool calls")?;
        tx.execute(
            r#"
            INSERT INTO turns
                (id, session_id, role, content, tool_calls, created_at,
                 input_tokens, output_tokens, archived)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            ON CONFLICT(id) DO UPDATE SET
                role = excluded.role,
                content = excluded.content,
                tool_calls = excluded.tool_calls,
                input_tokens = excluded.input_tokens,
                output_tokens = excluded.output_tokens,
                archived = excluded.archived
            "#,
            params![
                turn.id,
                session_id,
                turn.role,
                turn.content,
                tool_calls,
                turn.created_at.to_rfc3339(),
                turn.input_tokens as i64,
                turn.output_tokens as i64,
                archived as i64
            ],
        )
        .context("failed to insert turn")?;
        Ok(())
    }

    /// Load a session including all live (non-archived) turns.
    ///
    /// # Errors
    ///
    /// Returns an error if the database read fails or the stored JSON is invalid.
    pub async fn load_session(&self, session_id: &SessionId) -> Result<Option<Session>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, created_at, last_compacted_at, turn_count
                FROM sessions
                WHERE id = ?1
                "#,
            )
            .context("failed to prepare session select")?;

        let session_row = stmt
            .query_row(params![session_id], |row| {
                let created_at: String = row.get(1)?;
                let last_compacted_at: Option<String> = row.get(2)?;
                Ok((
                    row.get::<_, String>(0)?,
                    DateTime::parse_from_rfc3339(&created_at)
                        .map(|d| d.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    last_compacted_at.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .map(|d| d.with_timezone(&Utc))
                            .ok()
                    }),
                ))
            })
            .optional()
            .context("failed to load session")?;

        let Some((id, created_at, last_compacted_at)) = session_row else {
            return Ok(None);
        };

        let turns = Self::load_turns_for_session(&conn, session_id, false)?;
        Ok(Some(Session {
            id,
            turns,
            created_at,
            last_compacted_at,
        }))
    }

    fn load_turns_for_session(
        conn: &Connection,
        session_id: &SessionId,
        archived: bool,
    ) -> Result<Vec<Turn>> {
        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, role, content, tool_calls, created_at,
                       input_tokens, output_tokens
                FROM turns
                WHERE session_id = ?1 AND archived = ?2
                ORDER BY created_at ASC
                "#,
            )
            .context("failed to prepare turns select")?;

        let rows = stmt
            .query_map(params![session_id, archived as i64], |row| {
                let tool_calls_json: String = row.get(3)?;
                let created_at: String = row.get(4)?;
                let tool_calls: Vec<crate::ToolCall> =
                    serde_json::from_str(&tool_calls_json).unwrap_or_default();
                Ok(Turn {
                    id: row.get(0)?,
                    role: row.get(1)?,
                    content: row.get(2)?,
                    tool_calls,
                    created_at: DateTime::parse_from_rfc3339(&created_at)
                        .map(|d| d.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    input_tokens: row.get::<_, i64>(5)? as usize,
                    output_tokens: row.get::<_, i64>(6)? as usize,
                })
            })
            .context("failed to query turns")?;

        let mut turns = Vec::new();
        for row in rows {
            turns.push(row.context("failed to read turn row")?);
        }
        Ok(turns)
    }

    /// List stored session IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if the database read fails.
    pub async fn list_sessions(&self) -> Result<Vec<SessionId>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT id FROM sessions ORDER BY created_at DESC")
            .context("failed to prepare session list")?;
        let ids = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .context("failed to query sessions")?;
        ids.collect::<Result<Vec<_>, _>>()
            .context("failed to collect session ids")
    }

    /// Save the result of a compaction run.
    ///
    /// # Errors
    ///
    /// Returns an error if the database write fails.
    pub async fn save_compaction_result(&self, result: &CompactionResult) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            r#"
            INSERT INTO compaction_results
                (session_id, duplicates_removed, continuation_turns_collapsed,
                 turns_archived, token_bytes_saved, turns_remaining, compacted_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                result.session_id,
                result.metrics.duplicates_removed as i64,
                result.metrics.continuation_turns_collapsed as i64,
                result.metrics.turns_archived as i64,
                result.metrics.token_bytes_saved as i64,
                result.metrics.turns_remaining as i64,
                result.compacted_at.to_rfc3339()
            ],
        )
        .context("failed to save compaction result")?;
        debug!(session_id = %result.session_id, "saved compaction result");
        Ok(())
    }

    /// Aggregate metrics across all compaction runs.
    ///
    /// # Errors
    ///
    /// Returns an error if the database read fails.
    pub async fn aggregate_metrics(&self) -> Result<CompactionMetrics> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT
                    COALESCE(SUM(duplicates_removed), 0),
                    COALESCE(SUM(continuation_turns_collapsed), 0),
                    COALESCE(SUM(turns_archived), 0),
                    COALESCE(SUM(token_bytes_saved), 0),
                    COALESCE(SUM(turns_remaining), 0)
                FROM compaction_results
                "#,
            )
            .context("failed to prepare aggregate metrics")?;

        let row = stmt
            .query_row([], |row| {
                Ok(CompactionMetrics {
                    duplicates_removed: row.get::<_, i64>(0)? as usize,
                    continuation_turns_collapsed: row.get::<_, i64>(1)? as usize,
                    turns_archived: row.get::<_, i64>(2)? as usize,
                    token_bytes_saved: row.get::<_, i64>(3)? as usize,
                    turns_remaining: row.get::<_, i64>(4)? as usize,
                })
            })
            .context("failed to aggregate metrics")?;
        Ok(row)
    }

    /// Archive turns by moving them to the `archived = 1` column.
    ///
    /// # Errors
    ///
    /// Returns an error if the database write fails.
    pub async fn archive_turns(
        &self,
        session_id: &SessionId,
        turn_ids: &[TurnId],
    ) -> Result<usize> {
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction()
            .context("failed to begin archive transaction")?;
        let mut updated = 0;
        for id in turn_ids {
            updated += tx
                .execute(
                    "UPDATE turns SET archived = 1 WHERE id = ?1 AND session_id = ?2",
                    params![id, session_id],
                )
                .context("failed to archive turn")?;
        }
        tx.commit()
            .context("failed to commit archive transaction")?;
        Ok(updated)
    }

    /// Delete a session and all associated turns/results.
    ///
    /// # Errors
    ///
    /// Returns an error if the database write fails.
    pub async fn delete_session(&self, session_id: &SessionId) -> Result<bool> {
        let conn = self.conn.lock().await;
        let deleted = conn
            .execute("DELETE FROM sessions WHERE id = ?1", params![session_id])
            .context("failed to delete session")?;
        Ok(deleted > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Turn;

    #[tokio::test]
    async fn save_and_load_session_roundtrip() {
        let store = Store::open_in_memory().await.unwrap();
        let mut session = Session::new("s1");
        session.add_turn(Turn::new("user", "hello"));
        store.save_session(&session).await.unwrap();

        let loaded = store
            .load_session(&"s1".to_string())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.turns[0].content, "hello");
    }

    #[tokio::test]
    async fn aggregate_metrics_after_save() {
        let store = Store::open_in_memory().await.unwrap();
        let session = Session::new("s1");
        store.save_session(&session).await.unwrap();
        let result = CompactionResult {
            session_id: "s1".to_string(),
            metrics: CompactionMetrics {
                duplicates_removed: 3,
                token_bytes_saved: 100,
                ..Default::default()
            },
            compacted_at: Utc::now(),
        };
        store.save_compaction_result(&result).await.unwrap();
        let agg = store.aggregate_metrics().await.unwrap();
        assert_eq!(agg.duplicates_removed, 3);
        assert_eq!(agg.token_bytes_saved, 100);
    }
}
