//! CLI entry point for compactd.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use compactd::{
    api::{serve, AppState},
    config::Config,
    scanner::scan_and_compact,
    store::Store,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// compactd - Context compaction daemon for LLM agent sessions.
#[derive(Parser, Debug)]
#[command(name = "compactd", version, about, long_about = None)]
struct Cli {
    /// Path to a configuration file.
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the REST daemon and optional session watcher.
    Daemon,
    /// Run a one-shot compaction on the configured (or provided) session dir.
    Compact {
        /// Optional session directory to compact.
        #[arg(short, long)]
        session_dir: Option<PathBuf>,
    },
    /// Print the default configuration path.
    Config,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "compactd=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    let config = match &cli.config {
        Some(path) => Config::load(path)?,
        None => Config::load_default()?,
    };

    match cli.command {
        Commands::Daemon => run_daemon(config).await,
        Commands::Compact { session_dir } => run_compact(config, session_dir).await,
        Commands::Config => {
            println!("{}", Config::default_path().display());
            Ok(())
        }
    }
}

async fn run_daemon(config: Config) -> Result<()> {
    let store = Store::open(&config.database_path)
        .await
        .with_context(|| "failed to open store")?;
    let state = Arc::new(AppState {
        config: config.clone(),
        store,
    });

    let scanner_handle: JoinHandle<Result<()>> = tokio::spawn({
        let state = Arc::clone(&state);
        async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                if let Err(e) = scan_and_compact(
                    &state.store,
                    &state.config.session_dir,
                    state.config.max_turns,
                    state.config.archive_age_hours,
                )
                .await
                {
                    error!(error = %e, "session scan failed");
                }
            }
        }
    });

    let server_result = serve(Arc::clone(&state), None).await;
    scanner_handle.abort();
    server_result
}

async fn run_compact(config: Config, session_dir: Option<PathBuf>) -> Result<()> {
    let store = Store::open(&config.database_path).await?;
    let dir = session_dir.unwrap_or(config.session_dir);

    let results =
        scan_and_compact(&store, &dir, config.max_turns, config.archive_age_hours).await?;

    let aggregate: compactd::CompactionMetrics =
        results
            .iter()
            .fold(compactd::CompactionMetrics::default(), |mut acc, r| {
                acc.duplicates_removed += r.metrics.duplicates_removed;
                acc.continuation_turns_collapsed += r.metrics.continuation_turns_collapsed;
                acc.turns_archived += r.metrics.turns_archived;
                acc.token_bytes_saved += r.metrics.token_bytes_saved;
                acc
            });

    println!("compacted {} session(s)", results.len());
    println!(
        "  duplicates removed:          {}",
        aggregate.duplicates_removed
    );
    println!(
        "  continuation turns collapsed: {}",
        aggregate.continuation_turns_collapsed
    );
    println!(
        "  turns archived:              {}",
        aggregate.turns_archived
    );
    println!(
        "  token bytes saved:           {}",
        aggregate.token_bytes_saved
    );

    if results.is_empty() {
        info!("no sessions required compaction");
    }

    Ok(())
}
