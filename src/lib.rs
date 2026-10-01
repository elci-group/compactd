//! compactd - Context compaction daemon for LLM agent sessions.
//!
//! This crate provides algorithms and utilities to reduce context bloat in
//! agent session logs by removing duplicate tool outputs, collapsing repeated
//! "continue" turns, and archiving old turns before the context limit is hit.
//!
//! The compaction algorithms in [`compactor`] build without default features;
//! the daemon, store, scanner and config modules need the `daemon` feature.

#[cfg(feature = "daemon")]
pub mod api;
pub mod compactor;
#[cfg(feature = "daemon")]
pub mod config;
#[cfg(feature = "daemon")]
pub mod scanner;
#[cfg(feature = "daemon")]
pub mod store;

pub use compactor::{
    compact_session, compact_to_budget, parse_session_text, BudgetCompaction, BudgetPolicy,
    CompactionError, CompactionMetrics, CompactionResult, Session, SessionId, ToolCall, Turn,
    TurnId,
};
#[cfg(feature = "daemon")]
pub use config::Config;
#[cfg(feature = "daemon")]
pub use store::Store;
