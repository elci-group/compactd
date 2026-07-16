//! Axum REST API for compactd.

use crate::{config::Config, scanner::scan_and_compact, store::Store, CompactionMetrics};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info};

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub store: Store,
}

/// Simple health response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
}

/// Daemon status response.
#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub daemon: String,
    pub port: u16,
    pub session_dir: String,
    pub sessions_tracked: usize,
    pub aggregate: CompactionMetrics,
}

/// Request body for `/compact` and `/watch`.
#[derive(Debug, Deserialize)]
pub struct SessionDirRequest {
    pub session_dir: PathBuf,
}

/// Response body for `/compact`.
#[derive(Debug, Serialize)]
pub struct CompactResponse {
    pub compacted: usize,
    pub aggregate: CompactionMetrics,
}

/// Response body for `/watch`.
#[derive(Debug, Serialize)]
pub struct WatchResponse {
    pub watching: String,
    pub compacted: usize,
    pub aggregate: CompactionMetrics,
}

/// API error type convertible to an HTTP response.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("storage error: {0}")]
    Storage(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(serde_json::json!({ "error": self.to_string() }));
        (status, body).into_response()
    }
}

/// Build the Axum router.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/status", get(status_handler))
        .route("/metrics", get(metrics_handler))
        .route("/compact", post(compact_handler))
        .route("/watch", post(watch_handler))
        .with_state(state)
}

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn status_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<StatusResponse>, ApiError> {
    let sessions = state.store.list_sessions().await?;
    let aggregate = state.store.aggregate_metrics().await?;
    Ok(Json(StatusResponse {
        daemon: "compactd".to_string(),
        port: state.config.port,
        session_dir: state.config.session_dir.display().to_string(),
        sessions_tracked: sessions.len(),
        aggregate,
    }))
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> Result<Response, ApiError> {
    let aggregate = state.store.aggregate_metrics().await?;
    let body = format!(
        "# TYPE compactd_duplicates_removed counter\n\
         compactd_duplicates_removed {}\n\
         # TYPE compactd_continuation_turns_collapsed counter\n\
         compactd_continuation_turns_collapsed {}\n\
         # TYPE compactd_turns_archived counter\n\
         compactd_turns_archived {}\n\
         # TYPE compactd_token_bytes_saved counter\n\
         compactd_token_bytes_saved {}\n\
         # TYPE compactd_turns_remaining gauge\n\
         compactd_turns_remaining {}\n",
        aggregate.duplicates_removed,
        aggregate.continuation_turns_collapsed,
        aggregate.turns_archived,
        aggregate.token_bytes_saved,
        aggregate.turns_remaining
    );
    Ok((
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    )
        .into_response())
}

async fn compact_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SessionDirRequest>,
) -> Result<Json<CompactResponse>, ApiError> {
    let results = scan_and_compact(
        &state.store,
        &req.session_dir,
        state.config.max_turns,
        state.config.archive_age_hours,
    )
    .await?;

    let aggregate = sum_results(&results);
    info!(
        session_dir = %req.session_dir.display(),
        compacted = results.len(),
        "compaction run completed"
    );
    Ok(Json(CompactResponse {
        compacted: results.len(),
        aggregate,
    }))
}

async fn watch_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SessionDirRequest>,
) -> Result<Json<WatchResponse>, ApiError> {
    if !req.session_dir.exists() {
        return Err(ApiError::BadRequest(format!(
            "session directory does not exist: {}",
            req.session_dir.display()
        )));
    }

    let results = scan_and_compact(
        &state.store,
        &req.session_dir,
        state.config.max_turns,
        state.config.archive_age_hours,
    )
    .await?;

    let aggregate = sum_results(&results);
    info!(
        session_dir = %req.session_dir.display(),
        compacted = results.len(),
        "watch compaction run completed"
    );

    // Spawn a lightweight background watcher for this directory.
    tokio::spawn({
        let store = state.store.clone();
        let session_dir = req.session_dir.clone();
        let max_turns = state.config.max_turns;
        let archive_age_hours = state.config.archive_age_hours;
        async move {
            if let Err(e) = run_dir_watcher(store, session_dir, max_turns, archive_age_hours).await
            {
                error!(error = %e, "directory watcher exited");
            }
        }
    });

    Ok(Json(WatchResponse {
        watching: req.session_dir.display().to_string(),
        compacted: results.len(),
        aggregate,
    }))
}

fn sum_results(results: &[crate::CompactionResult]) -> CompactionMetrics {
    let mut aggregate = CompactionMetrics::default();
    for r in results {
        aggregate.duplicates_removed += r.metrics.duplicates_removed;
        aggregate.continuation_turns_collapsed += r.metrics.continuation_turns_collapsed;
        aggregate.turns_archived += r.metrics.turns_archived;
        aggregate.token_bytes_saved += r.metrics.token_bytes_saved;
    }
    aggregate.turns_remaining = aggregate.turns_remaining.saturating_add(0);
    aggregate
}

async fn run_dir_watcher(
    store: Store,
    session_dir: PathBuf,
    max_turns: usize,
    archive_age_hours: u64,
) -> anyhow::Result<()> {
    use notify::{Config as NotifyConfig, Event, RecommendedWatcher, RecursiveMode, Watcher};
    use std::sync::mpsc::channel;

    let (tx, rx) = channel::<Result<Event, notify::Error>>();
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            let _ = tx.send(res);
        },
        NotifyConfig::default(),
    )
    .map_err(|e| anyhow::anyhow!("failed to create file watcher: {e}"))?;

    watcher
        .watch(&session_dir, RecursiveMode::Recursive)
        .map_err(|e| anyhow::anyhow!("failed to watch session directory: {e}"))?;

    info!(path = %session_dir.display(), "watching session directory");

    let mut debounce = tokio::time::interval(std::time::Duration::from_secs(2));
    debounce.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        debounce.tick().await;
        let mut changed = false;
        while rx.try_recv().is_ok() {
            changed = true;
        }
        if changed {
            if let Err(e) =
                scan_and_compact(&store, &session_dir, max_turns, archive_age_hours).await
            {
                error!(error = %e, "watcher-triggered scan failed");
            }
        }
    }
}

/// Start the REST server on the configured address.
///
/// In tests the listener should be bound to `127.0.0.1:0`; in production the
/// configured host/port is used.
///
/// # Errors
///
/// Returns an error if the listener cannot be bound or the server fails.
pub async fn serve(
    state: Arc<AppState>,
    listener: Option<tokio::net::TcpListener>,
) -> anyhow::Result<()> {
    let listener = match listener {
        Some(l) => l,
        None => {
            let addr = format!("{}:{}", state.config.host, state.config.port);
            tokio::net::TcpListener::bind(&addr).await?
        }
    };

    let local_addr = listener.local_addr()?;
    info!(addr = %local_addr, "compactd listening");

    axum::serve(listener, router(state))
        .await
        .map_err(|e| anyhow::anyhow!("server error: {e}"))
}
