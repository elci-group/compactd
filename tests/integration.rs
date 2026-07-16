use compactd::{
    api::{router, AppState},
    config::Config,
    scanner::write_sample_session,
    store::Store,
};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;

async fn test_app() -> (axum::Router, Arc<AppState>) {
    let config = Config {
        host: "127.0.0.1".to_string(),
        port: 0,
        session_dir: std::env::temp_dir().join("compactd-test-sessions"),
        database_path: std::env::temp_dir()
            .join(format!("compactd-test-{}.db", uuid::Uuid::new_v4())),
        max_turns: 5,
        archive_age_hours: 24,
        dedup_similarity_threshold: 1.0,
        watch_sessions: false,
    };
    let store = Store::open(&config.database_path).await.unwrap();
    let state = Arc::new(AppState { config, store });
    (router(Arc::clone(&state)), state)
}

async fn spawn_server() -> (SocketAddr, Arc<AppState>) {
    let (app, state) = test_app().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, state)
}

#[tokio::test]
async fn health_returns_ok() {
    let (addr, _state) = spawn_server().await;
    let client = reqwest::Client::new();
    let res = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn status_reports_daemon_state() {
    let (addr, _state) = spawn_server().await;
    let client = reqwest::Client::new();
    let res = client
        .get(format!("http://{addr}/status"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["daemon"], "compactd");
    assert!(body["sessions_tracked"].is_number());
}

#[tokio::test]
async fn compact_endpoint_scans_and_compacts() {
    let (addr, state) = spawn_server().await;
    let client = reqwest::Client::new();
    let session_dir =
        std::env::temp_dir().join(format!("compactd-compact-test-{}", uuid::Uuid::new_v4()));
    write_sample_session(&session_dir, "sample").await.unwrap();

    let payload = json!({ "session_dir": session_dir });
    let res = client
        .post(format!("http://{addr}/compact"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);

    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["compacted"], 1);
    assert!(
        body["aggregate"]["continuation_turns_collapsed"]
            .as_u64()
            .unwrap()
            > 0
            || body["aggregate"]["duplicates_removed"].as_u64().unwrap() > 0
    );

    let sessions = state.store.list_sessions().await.unwrap();
    assert_eq!(sessions.len(), 1);
}

#[tokio::test]
async fn watch_endpoint_starts_watching() {
    let (addr, _state) = spawn_server().await;
    let client = reqwest::Client::new();
    let session_dir =
        std::env::temp_dir().join(format!("compactd-watch-test-{}", uuid::Uuid::new_v4()));
    write_sample_session(&session_dir, "sample").await.unwrap();

    let payload = json!({ "session_dir": session_dir });
    let res = client
        .post(format!("http://{addr}/watch"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);

    let body: serde_json::Value = res.json().await.unwrap();
    assert!(body["watching"]
        .as_str()
        .unwrap()
        .contains("compactd-watch-test"));
    assert_eq!(body["compacted"], 1);
}

#[tokio::test]
async fn metrics_returns_prometheus_text() {
    let (addr, _state) = spawn_server().await;
    let client = reqwest::Client::new();
    let res = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let body = res.text().await.unwrap();
    assert!(body.contains("compactd_duplicates_removed"));
    assert!(body.contains("compactd_token_bytes_saved"));
}

#[test]
fn cli_config_prints_default_path() {
    let output = std::process::Command::new("cargo")
        .args(["run", "--quiet", "--", "config"])
        .current_dir(std::path::Path::new(env!("CARGO_MANIFEST_DIR")))
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("compactd") && stdout.contains("config.toml"));
}
