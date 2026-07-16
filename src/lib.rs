//! compactd - Context compaction daemon for LLM agent sessions.
//!
//! This crate provides algorithms and utilities to reduce context bloat in
//! agent session logs by removing duplicate tool outputs, collapsing repeated
//! "continue" turns, and archiving old turns before the context limit is hit.

pub mod api;
pub mod compactor;
pub mod config;
pub mod scanner;
pub mod store;

pub use compactor::{
    compact_session, parse_session_text, CompactionError, CompactionMetrics, CompactionResult,
    Session, SessionId, ToolCall, Turn, TurnId,
};
pub use config::Config;
pub use store::Store;
