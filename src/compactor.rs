//! Context compaction logic for agent sessions.
//!
//! Provides strategies to reduce context bloat by removing duplicate tool
//! outputs, collapsing repeated continuation turns, and archiving old turns.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use tracing::{debug, trace};

/// Identifier for a single conversational turn.
pub type TurnId = String;

/// Identifier for an agent session.
pub type SessionId = String;

/// A tool invocation recorded inside a turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ToolCall {
    /// Tool name, e.g. "Shell", "Read", "Grep".
    pub name: String,

    /// Serialized tool arguments.
    pub arguments: String,

    /// Serialized tool output.
    pub output: String,
}

impl ToolCall {
    /// Compute a stable hash of the call signature (name + arguments + output).
    #[must_use]
    pub fn digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        hasher.update(self.arguments.as_bytes());
        hasher.update(self.output.as_bytes());
        hex::encode(hasher.finalize())
    }
}

/// A single turn within an agent session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Turn {
    /// Stable turn identifier.
    pub id: TurnId,

    /// Role of the turn author ("user", "assistant", "tool", etc.).
    pub role: String,

    /// Text content of the turn.
    pub content: String,

    /// Tool calls emitted or consumed by this turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,

    /// Estimated input tokens consumed by this turn.
    #[serde(default)]
    pub input_tokens: usize,

    /// Estimated output tokens produced by this turn.
    #[serde(default)]
    pub output_tokens: usize,
}

impl Turn {
    /// Create a new turn with the current timestamp.
    #[must_use]
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: Vec::new(),
            created_at: Utc::now(),
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    /// Return `true` if this turn is a continuation marker.
    #[must_use]
    pub fn is_continuation_marker(&self) -> bool {
        let trimmed = self.content.trim();
        matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "continue"
                | "go on"
                | "proceed"
                | "keep going"
                | "next"
                | "please continue"
                | "carry on"
        )
    }

    /// Estimate the total token weight of this turn.
    #[must_use]
    pub fn token_weight(&self) -> usize {
        let tool_weight: usize = self
            .tool_calls
            .iter()
            .map(|t| t.arguments.len() + t.output.len())
            .sum();
        self.content.len() + tool_weight
    }
}

/// An agent session composed of chronological turns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Session {
    /// Session identifier.
    pub id: SessionId,

    /// Chronological turns.
    pub turns: Vec<Turn>,

    /// Time the session was first seen.
    pub created_at: DateTime<Utc>,

    /// Time the session was last compacted.
    #[serde(default)]
    pub last_compacted_at: Option<DateTime<Utc>>,
}

impl Session {
    /// Create an empty session.
    #[must_use]
    pub fn new(id: impl Into<SessionId>) -> Self {
        Self {
            id: id.into(),
            turns: Vec::new(),
            created_at: Utc::now(),
            last_compacted_at: None,
        }
    }

    /// Append a turn to the session.
    pub fn add_turn(&mut self, turn: Turn) {
        self.turns.push(turn);
    }

    /// Total number of turns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Returns `true` if the session has no turns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Sum of input tokens across all turns.
    #[must_use]
    pub fn total_input_tokens(&self) -> usize {
        self.turns.iter().map(|t| t.input_tokens).sum()
    }

    /// Sum of output tokens across all turns.
    #[must_use]
    pub fn total_output_tokens(&self) -> usize {
        self.turns.iter().map(|t| t.output_tokens).sum()
    }
}

/// Metrics describing the savings achieved by a compaction run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct CompactionMetrics {
    /// Number of duplicate tool outputs removed.
    pub duplicates_removed: usize,

    /// Number of continuation turns collapsed.
    pub continuation_turns_collapsed: usize,

    /// Number of old turns archived.
    pub turns_archived: usize,

    /// Estimated token bytes removed.
    pub token_bytes_saved: usize,

    /// Turns remaining after compaction.
    pub turns_remaining: usize,
}

/// Result returned by a compaction pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompactionResult {
    /// Identifier of the session that was compacted.
    pub session_id: SessionId,

    /// Metrics for the compaction run.
    pub metrics: CompactionMetrics,

    /// Timestamp of the compaction run.
    pub compacted_at: DateTime<Utc>,
}

/// Errors that may occur during compaction.
#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
    /// A requested session was not found in the store.
    #[error("session not found: {0}")]
    SessionNotFound(SessionId),

    /// An I/O or storage operation failed.
    #[error("storage error: {0}")]
    Storage(#[from] anyhow::Error),
}

/// A strategy that compacts a session in place and returns metrics.
pub trait CompactionStrategy {
    /// Run the strategy against the provided session.
    ///
    /// # Errors
    ///
    /// Returns an error if the strategy requires storage access and fails.
    fn compact(&self, session: &mut Session) -> Result<CompactionMetrics, CompactionError>;
}

/// Remove duplicate tool outputs within a session.
#[derive(Debug, Default, Clone, Copy)]
pub struct DeduplicateToolOutputs;

impl CompactionStrategy for DeduplicateToolOutputs {
    fn compact(&self, session: &mut Session) -> Result<CompactionMetrics, CompactionError> {
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut removed = 0;
        let mut saved: usize = 0;

        for turn in &mut session.turns {
            let mut retained = Vec::with_capacity(turn.tool_calls.len());
            for call in &turn.tool_calls {
                let digest = call.digest();
                if let Some(count) = seen.get(&digest) {
                    trace!(turn_id = %turn.id, tool = %call.name, "removing duplicate tool output");
                    removed += 1;
                    saved += call.arguments.len() + call.output.len();
                    if *count == 1 {
                        retained.push(ToolCall {
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                            output: "<duplicate output elided>".to_string(),
                        });
                    }
                    seen.insert(digest, count + 1);
                } else {
                    seen.insert(digest, 1);
                    retained.push(call.clone());
                }
            }
            turn.tool_calls = retained;
        }

        debug!(session_id = %session.id, removed, saved, "deduplicated tool outputs");
        Ok(CompactionMetrics {
            duplicates_removed: removed,
            token_bytes_saved: saved,
            turns_remaining: session.len(),
            ..Default::default()
        })
    }
}

/// Collapse consecutive continuation-marker turns into a single marker.
#[derive(Debug, Default, Clone, Copy)]
pub struct CollapseContinuations;

impl CompactionStrategy for CollapseContinuations {
    fn compact(&self, session: &mut Session) -> Result<CompactionMetrics, CompactionError> {
        let mut compacted = Vec::with_capacity(session.len());
        let mut collapsed = 0;

        for turn in session.turns.drain(..) {
            if turn.is_continuation_marker()
                && compacted
                    .last()
                    .is_some_and(|t: &Turn| t.is_continuation_marker() && t.role == turn.role)
            {
                collapsed += 1;
                continue;
            }
            compacted.push(turn);
        }

        session.turns = compacted;
        debug!(
            session_id = %session.id,
            collapsed,
            "collapsed continuation turns"
        );
        Ok(CompactionMetrics {
            continuation_turns_collapsed: collapsed,
            token_bytes_saved: collapsed * "continue".len(),
            turns_remaining: session.len(),
            ..Default::default()
        })
    }
}

/// Archive turns that are older than an age threshold, keeping the most recent
/// `max_live` turns regardless of age.
#[derive(Debug, Clone, Copy)]
pub struct ArchiveOldTurns {
    /// Maximum live turns to retain.
    pub max_live: usize,

    /// Age threshold. Turns older than this are candidates for archiving.
    pub age_hours: u64,
}

impl ArchiveOldTurns {
    /// Create a new archiver.
    #[must_use]
    pub fn new(max_live: usize, age_hours: u64) -> Self {
        Self {
            max_live,
            age_hours,
        }
    }
}

impl CompactionStrategy for ArchiveOldTurns {
    fn compact(&self, session: &mut Session) -> Result<CompactionMetrics, CompactionError> {
        let cutoff = Utc::now() - chrono::Duration::hours(self.age_hours as i64);
        let mut archived = 0;
        let mut saved: usize = 0;
        let mut live = Vec::with_capacity(session.len());

        let protected_count = session.len().saturating_sub(self.max_live);
        for (idx, turn) in session.turns.drain(..).enumerate() {
            if idx < protected_count && turn.created_at < cutoff {
                archived += 1;
                saved += turn.token_weight();
            } else {
                live.push(turn);
            }
        }

        session.turns = live;
        debug!(session_id = %session.id, archived, "archived old turns");
        Ok(CompactionMetrics {
            turns_archived: archived,
            token_bytes_saved: saved,
            turns_remaining: session.len(),
            ..Default::default()
        })
    }
}

/// Run the full set of compaction strategies in order.
///
/// Order matters: deduplication and collapsing are performed before archiving
/// so that archive decisions are made on the already-reduced context.
///
/// # Errors
///
/// Returns an error if any strategy fails.
pub fn compact_session(
    session: &mut Session,
    max_turns: usize,
    archive_age_hours: u64,
) -> Result<CompactionResult, CompactionError> {
    let mut total = CompactionMetrics::default();

    let m = DeduplicateToolOutputs.compact(session)?;
    total.duplicates_removed += m.duplicates_removed;
    total.token_bytes_saved += m.token_bytes_saved;

    let m = CollapseContinuations.compact(session)?;
    total.continuation_turns_collapsed += m.continuation_turns_collapsed;
    total.token_bytes_saved += m.token_bytes_saved;

    let m = ArchiveOldTurns::new(max_turns, archive_age_hours).compact(session)?;
    total.turns_archived += m.turns_archived;
    total.token_bytes_saved += m.token_bytes_saved;

    total.turns_remaining = session.len();
    session.last_compacted_at = Some(Utc::now());

    Ok(CompactionResult {
        session_id: session.id.clone(),
        metrics: total,
        compacted_at: Utc::now(),
    })
}

/// Limits for [`compact_to_budget`], all measured in [`Turn::token_weight`]
/// units (bytes of content plus tool arguments and outputs).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetPolicy {
    /// Total weight the live turns may occupy after compaction.
    pub budget: usize,
    /// Any single turn's content is cut to this many bytes before fitting.
    pub max_turn_bytes: usize,
    /// Upper bound on the digest that stands in for evicted turns.
    pub digest_bytes: usize,
}

/// Result of [`compact_to_budget`]: the usual metrics, plus a deterministic
/// digest of the turns that no longer fit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BudgetCompaction {
    /// Metrics; `turns_archived` counts turns evicted to meet the budget.
    pub result: CompactionResult,
    /// Plain-text outline of the evicted turns, oldest first, or `None` when
    /// nothing was evicted.
    pub digest: Option<String>,
    /// Number of live turns whose content was cut to `max_turn_bytes`.
    pub turns_truncated: usize,
}

const TRUNCATION_MARK: &str = " … [truncated]";

/// Cut `text` to at most `max` bytes on a char boundary.
fn cut_to(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// First non-empty line of `text`, whitespace-collapsed and cut to `max` bytes.
fn gist(text: &str, max: usize) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.len() <= max {
        line
    } else {
        format!("{}…", cut_to(&line, max.saturating_sub("…".len())))
    }
}

fn digest_line(turn: &Turn) -> String {
    let verb = match turn.role.as_str() {
        "user" => "user asked",
        "assistant" => "assistant answered",
        other => other,
    };
    let mut line = format!("- {verb}: {}", gist(&turn.content, 160));
    if !turn.tool_calls.is_empty() {
        let mut names: Vec<&str> = turn.tool_calls.iter().map(|c| c.name.as_str()).collect();
        names.dedup();
        line.push_str(&format!(" (tools: {})", names.join(", ")));
    }
    line
}

/// Build a digest of `evicted` within `max` bytes. The opening turn is kept
/// for the conversation's topic; the rest of the room favours the most recent
/// evicted turns, since they are the ones the live context continues from.
fn build_digest(evicted: &[Turn], max: usize) -> Option<String> {
    if evicted.is_empty() {
        return None;
    }
    let header = format!(
        "Earlier in this conversation ({} turn{} compacted):",
        evicted.len(),
        if evicted.len() == 1 { "" } else { "s" }
    );
    let mut used = header.len();
    if used > max {
        return None;
    }
    let lines: Vec<String> = evicted.iter().map(digest_line).collect();
    let mut first: Option<&String> = None;
    if used + 1 + lines[0].len() <= max {
        used += 1 + lines[0].len();
        first = Some(&lines[0]);
    }
    // Room for the "… N more turn(s) omitted" marker, so it is never the
    // line squeezed out when the tail fills the bound.
    const GAP_RESERVE: usize = 40;
    let mut tail = Vec::new();
    for line in lines.iter().skip(1).rev() {
        if used + 1 + line.len() + GAP_RESERVE > max {
            break;
        }
        used += 1 + line.len();
        tail.push(line);
    }
    tail.reverse();
    let shown = usize::from(first.is_some()) + tail.len();
    let mut out = header;
    if let Some(line) = first {
        out.push('\n');
        out.push_str(line);
    }
    let omitted = evicted.len() - shown;
    if omitted > 0 && used + GAP_RESERVE <= max {
        out.push_str(&format!("\n- … {omitted} more turn(s) omitted"));
    }
    for line in tail {
        out.push('\n');
        out.push_str(line);
    }
    Some(out)
}

/// Compact a session to fit a weight budget instead of a turn count.
///
/// Runs [`DeduplicateToolOutputs`] and [`CollapseContinuations`], cuts any
/// oversized turn to `policy.max_turn_bytes`, then keeps the longest run of
/// most-recent turns that fits `policy.budget`. Older turns are evicted and
/// summarised in a deterministic digest (no model call), so a caller can keep
/// the conversation's earlier thread in view at a fraction of its size.
///
/// The live turns always form a contiguous suffix of the session: a turn is
/// never dropped from between two kept ones.
///
/// # Errors
///
/// Propagates errors from the underlying strategies.
pub fn compact_to_budget(
    session: &mut Session,
    policy: BudgetPolicy,
) -> Result<BudgetCompaction, CompactionError> {
    let mut total = CompactionMetrics::default();

    let m = DeduplicateToolOutputs.compact(session)?;
    total.duplicates_removed += m.duplicates_removed;
    total.token_bytes_saved += m.token_bytes_saved;

    let m = CollapseContinuations.compact(session)?;
    total.continuation_turns_collapsed += m.continuation_turns_collapsed;
    total.token_bytes_saved += m.token_bytes_saved;

    let mut truncated = 0;
    for turn in &mut session.turns {
        if turn.content.len() > policy.max_turn_bytes {
            let keep = policy.max_turn_bytes.saturating_sub(TRUNCATION_MARK.len());
            let cut = cut_to(&turn.content, keep).to_string();
            total.token_bytes_saved += turn.content.len() - cut.len();
            turn.content = cut + TRUNCATION_MARK;
            truncated += 1;
        }
    }

    let mut used = 0;
    let mut keep_from = session.len();
    for (idx, turn) in session.turns.iter().enumerate().rev() {
        let weight = turn.token_weight();
        if used + weight > policy.budget {
            break;
        }
        used += weight;
        keep_from = idx;
    }
    let evicted: Vec<Turn> = session.turns.drain(..keep_from).collect();
    total.turns_archived = evicted.len();
    total.token_bytes_saved += evicted.iter().map(Turn::token_weight).sum::<usize>();
    let digest = build_digest(&evicted, policy.digest_bytes);

    total.turns_remaining = session.len();
    session.last_compacted_at = Some(Utc::now());
    debug!(
        session_id = %session.id,
        evicted = evicted.len(),
        truncated,
        "compacted session to budget"
    );

    Ok(BudgetCompaction {
        result: CompactionResult {
            session_id: session.id.clone(),
            metrics: total,
            compacted_at: Utc::now(),
        },
        digest,
        turns_truncated: truncated,
    })
}

/// Parse a session from a JSON object or JSONL text.
///
/// If `text` begins with `[` it is parsed as a JSON array of turns; otherwise
/// each non-empty line is parsed as a [`Turn`].
///
/// # Errors
///
/// Returns an error if the text is not valid JSON/JSONL.
pub fn parse_session_text(id: impl Into<SessionId>, text: &str) -> anyhow::Result<Session> {
    let trimmed = text.trim();
    let turns: Vec<Turn> = if trimmed.starts_with('[') {
        serde_json::from_str(trimmed)?
    } else {
        trimmed
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(serde_json::from_str)
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok(Session {
        id: id.into(),
        turns,
        created_at: Utc::now(),
        last_compacted_at: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn_with_content(role: &str, content: &str) -> Turn {
        Turn::new(role, content)
    }

    #[test]
    fn deduplicate_removes_identical_tool_outputs() {
        let mut session = Session::new("s1");
        let mut t1 = Turn::new("assistant", "first");
        t1.tool_calls.push(ToolCall {
            name: "Read".to_string(),
            arguments: "{\"path\": \"a.txt\"}".to_string(),
            output: "hello".to_string(),
        });
        session.add_turn(t1);

        let mut t2 = Turn::new("assistant", "second");
        t2.tool_calls.push(ToolCall {
            name: "Read".to_string(),
            arguments: "{\"path\": \"a.txt\"}".to_string(),
            output: "hello".to_string(),
        });
        session.add_turn(t2);

        let result = DeduplicateToolOutputs.compact(&mut session).unwrap();
        assert_eq!(result.duplicates_removed, 1);
        assert_eq!(
            session.turns[1].tool_calls[0].output,
            "<duplicate output elided>"
        );
    }

    #[test]
    fn collapse_continuations() {
        let mut session = Session::new("s1");
        session.add_turn(turn_with_content("user", "continue"));
        session.add_turn(turn_with_content("user", "Continue"));
        session.add_turn(turn_with_content("user", "go on"));
        session.add_turn(turn_with_content("assistant", "ok"));

        let result = CollapseContinuations.compact(&mut session).unwrap();
        assert_eq!(result.continuation_turns_collapsed, 2);
        assert_eq!(session.len(), 2);
    }

    #[test]
    fn archive_old_turns_keeps_recent() {
        let mut session = Session::new("s1");
        for i in (0..5).rev() {
            let mut t = Turn::new("user", &format!("turn {i}"));
            t.created_at = Utc::now() - chrono::Duration::hours(i as i64 * 10);
            session.add_turn(t);
        }

        let result = ArchiveOldTurns::new(2, 15).compact(&mut session).unwrap();
        assert_eq!(result.turns_archived, 3);
        assert_eq!(session.len(), 2);
        assert!(session
            .turns
            .iter()
            .all(|t| t.content == "turn 0" || t.content == "turn 1"));
    }

    #[test]
    fn full_compaction_composes_strategies() {
        let mut session = Session::new("s1");
        for _ in 0..3 {
            session.add_turn(turn_with_content("user", "continue"));
        }
        let mut t = Turn::new("assistant", "ok");
        t.tool_calls.push(ToolCall {
            name: "Read".to_string(),
            arguments: "{}".to_string(),
            output: "dup".to_string(),
        });
        session.add_turn(t);
        let mut t2 = Turn::new("assistant", "ok");
        t2.tool_calls.push(ToolCall {
            name: "Read".to_string(),
            arguments: "{}".to_string(),
            output: "dup".to_string(),
        });
        session.add_turn(t2);

        let result = compact_session(&mut session, 10, 1).unwrap();
        assert!(
            result.metrics.duplicates_removed > 0
                || result.metrics.continuation_turns_collapsed > 0
        );
        assert!(session.last_compacted_at.is_some());
    }

    fn policy(budget: usize) -> BudgetPolicy {
        BudgetPolicy {
            budget,
            max_turn_bytes: 1_000,
            digest_bytes: 400,
        }
    }

    #[test]
    fn budget_keeps_everything_that_fits() {
        let mut session = Session::new("s1");
        session.add_turn(turn_with_content("user", "hello"));
        session.add_turn(turn_with_content("assistant", "hi there"));

        let out = compact_to_budget(&mut session, policy(100)).unwrap();
        assert_eq!(session.len(), 2);
        assert_eq!(out.result.metrics.turns_archived, 0);
        assert!(out.digest.is_none());
    }

    #[test]
    fn budget_evicts_oldest_contiguously_and_digests_them() {
        let mut session = Session::new("s1");
        session.add_turn(turn_with_content("user", "What does hellhound do?\nmore"));
        session.add_turn(turn_with_content("assistant", &"a".repeat(50)));
        session.add_turn(turn_with_content("user", "And winch?"));
        session.add_turn(turn_with_content("assistant", "Winch resolves deps."));

        let out = compact_to_budget(&mut session, policy(40)).unwrap();
        assert_eq!(session.len(), 2);
        assert_eq!(session.turns[0].content, "And winch?");
        assert_eq!(out.result.metrics.turns_archived, 2);
        let digest = out.digest.unwrap();
        assert!(digest.starts_with("Earlier in this conversation (2 turns compacted):"));
        assert!(digest.contains("- user asked: What does hellhound do?"));
        assert!(!digest.contains("more"));
        assert!(digest.contains("- assistant answered: aaa"));
    }

    #[test]
    fn budget_never_skips_a_turn_to_fit_an_older_one() {
        let mut session = Session::new("s1");
        session.add_turn(turn_with_content("user", "tiny"));
        session.add_turn(turn_with_content("assistant", &"b".repeat(80)));
        session.add_turn(turn_with_content("user", "latest"));

        compact_to_budget(&mut session, policy(20)).unwrap();
        assert_eq!(session.len(), 1);
        assert_eq!(session.turns[0].content, "latest");
    }

    #[test]
    fn budget_truncates_oversized_turns_on_char_boundaries() {
        let mut session = Session::new("s1");
        session.add_turn(turn_with_content("user", &"é".repeat(2_000)));

        let out = compact_to_budget(&mut session, policy(5_000)).unwrap();
        assert_eq!(out.turns_truncated, 1);
        assert!(session.turns[0].content.len() <= 1_000);
        assert!(session.turns[0].content.ends_with("[truncated]"));
    }

    #[test]
    fn digest_stays_within_its_bound_and_keeps_topic_and_latest() {
        let mut session = Session::new("s1");
        session.add_turn(turn_with_content("user", "Opening topic"));
        for i in 0..40 {
            session.add_turn(turn_with_content("assistant", &format!("reply number {i}")));
        }
        session.add_turn(turn_with_content("user", &"z".repeat(30)));

        let out = compact_to_budget(&mut session, policy(30)).unwrap();
        let digest = out.digest.unwrap();
        assert!(digest.len() <= 400, "digest was {} bytes", digest.len());
        assert!(digest.contains("Opening topic"));
        assert!(digest.contains("reply number 39"));
        assert!(digest.contains("more turn(s) omitted"));
    }

    #[test]
    fn budget_collapses_continuations_before_fitting() {
        let mut session = Session::new("s1");
        session.add_turn(turn_with_content("user", "continue"));
        session.add_turn(turn_with_content("user", "continue"));
        session.add_turn(turn_with_content("assistant", "ok"));

        let out = compact_to_budget(&mut session, policy(100)).unwrap();
        assert_eq!(out.result.metrics.continuation_turns_collapsed, 1);
        assert_eq!(session.len(), 2);
    }

    #[test]
    fn parse_session_json_array() {
        let text =
            r#"[{"id":"t1","role":"user","content":"hi","created_at":"2024-01-01T00:00:00Z"}]"#;
        let session = parse_session_text("s1", text).unwrap();
        assert_eq!(session.len(), 1);
        assert_eq!(session.turns[0].content, "hi");
    }

    #[test]
    fn parse_session_jsonl() {
        let text = "{\"id\":\"t1\",\"role\":\"user\",\"content\":\"hi\",\"created_at\":\"2024-01-01T00:00:00Z\"}\n";
        let session = parse_session_text("s1", text).unwrap();
        assert_eq!(session.len(), 1);
    }
}
