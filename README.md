# compactd

`compactd` is a standalone Rust daemon that addresses **Context Compaction** in
LLM agent sessions. It proactively reduces context bloat by:

- **Removing duplicate tool outputs** – identical tool invocations with the same
  output are collapsed after the first occurrence.
- **Collapsing repeated "continue" turns** – consecutive continuation markers
  are merged into a single turn.
- **Archiving old turns** – turns beyond a configurable age/count threshold are
  moved out of the live context before the context limit is hit.

The daemon exposes a small REST API (Axum) for inspecting and triggering
compaction, persists state in SQLite, and can watch a session directory for new
agent logs.

## Install

From the repository root:

```bash
cargo install --path .
```

Or copy the release binary:

```bash
cargo build --release
cp target/release/compactd ~/.local/bin/
```

## Usage

### Start the daemon

```bash
compactd daemon
```

The daemon binds to `127.0.0.1:3103` by default. Configuration is loaded from
`~/.config/compactd/config.toml` (create it if needed) and can be overridden
with environment variables such as `COMPACTD_PORT`.

### One-shot compaction

```bash
compactd compact
# or target a specific directory
compactd compact --session-dir /path/to/sessions
```

### Print the default config path

```bash
compactd config
```

## Configuration

Create `~/.config/compactd/config.toml`:

```toml
host = "127.0.0.1"
port = 3103
session_dir = "~/.config/kimi/sessions"
database_path = "~/.local/share/compactd/state.db"
max_turns = 100
archive_age_hours = 24
dedup_similarity_threshold = 1.0
watch_sessions = true
```

Environment variable overrides:

- `COMPACTD_HOST`
- `COMPACTD_PORT`
- `COMPACTD_SESSION_DIR`
- `COMPACTD_DATABASE_PATH`
- `COMPACTD_MAX_TURNS`
- `COMPACTD_ARCHIVE_AGE_HOURS`
- `COMPACTD_DEDUP_SIMILARITY_THRESHOLD`
- `COMPACTD_WATCH_SESSIONS`

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{ "status": "ok" }` |
| GET | `/status` | Daemon state and aggregate metrics |
| GET | `/metrics` | Prometheus exposition format metrics |
| POST | `/compact` | Scan a session dir and compact all sessions |
| POST | `/watch` | Scan + compact a session dir and start watching it |

### Examples

```bash
# Trigger compaction on a directory
curl -X POST http://127.0.0.1:3103/compact \
  -H 'Content-Type: application/json' \
  -d '{"session_dir": "/home/user/.config/kimi/sessions"}'

# Start watching a directory
curl -X POST http://127.0.0.1:3103/watch \
  -H 'Content-Type: application/json' \
  -d '{"session_dir": "/home/user/.config/kimi/sessions"}'

# Prometheus metrics
curl http://127.0.0.1:3103/metrics
```

## Development

```bash
# Run tests
cargo test

# Run with Clippy warnings as errors
cargo clippy --all-targets -- -D warnings

# Check formatting
cargo fmt --check

# Build release binary
cargo build --release

# Validate deliverables
deliver --spec deliver.toml --strict
```

## License

MIT

Monitored by [kaptaind](https://github.com/elci-group/kaptaind).
