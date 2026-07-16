//! Configuration loading for compactd.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// Default TCP port for the compactd REST daemon.
pub const DEFAULT_PORT: u16 = 3103;

/// Default host to bind the REST daemon.
pub const DEFAULT_HOST: &str = "127.0.0.1";

/// Default directory to scan for agent sessions.
pub const DEFAULT_SESSION_DIR: &str = "~/.config/kimi/sessions";

/// Default maximum number of turns before archiving is triggered.
pub const DEFAULT_MAX_TURNS: usize = 100;

/// Default age in hours after which a turn may be archived.
pub const DEFAULT_ARCHIVE_AGE_HOURS: u64 = 24;

/// Default similarity threshold for considering two tool outputs duplicates.
pub const DEFAULT_DEDUP_SIMILARITY_THRESHOLD: f64 = 1.0;

/// compactd configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    /// Host address to bind.
    pub host: String,

    /// TCP port to listen on.
    pub port: u16,

    /// Directory containing agent session files.
    pub session_dir: PathBuf,

    /// Path to the SQLite state database.
    pub database_path: PathBuf,

    /// Maximum turns retained in the live context.
    pub max_turns: usize,

    /// Age threshold for archiving turns, in hours.
    pub archive_age_hours: u64,

    /// Similarity threshold for duplicate tool output detection (0.0-1.0).
    pub dedup_similarity_threshold: f64,

    /// Whether to enable filesystem watching of session_dir.
    pub watch_sessions: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
            session_dir: expand_home(DEFAULT_SESSION_DIR),
            database_path: default_database_path(),
            max_turns: DEFAULT_MAX_TURNS,
            archive_age_hours: DEFAULT_ARCHIVE_AGE_HOURS,
            dedup_similarity_threshold: DEFAULT_DEDUP_SIMILARITY_THRESHOLD,
            watch_sessions: true,
        }
    }
}

impl Config {
    /// Return the default configuration file path (`~/.config/compactd/config.toml`).
    pub fn default_path() -> PathBuf {
        dirs::config_dir()
            .map(|p| p.join("compactd").join("config.toml"))
            .unwrap_or_else(|| PathBuf::from("compactd.toml"))
    }

    /// Load configuration from a TOML file, falling back to defaults for missing
    /// keys. Environment variables prefixed with `COMPACTD_` override file values.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read or parsed.
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let mut config = if path.as_ref().exists() {
            let text = std::fs::read_to_string(path.as_ref())?;
            toml::from_str(&text)?
        } else {
            warn!(
                path = %path.as_ref().display(),
                "config file not found; using defaults"
            );
            Self::default()
        };

        config.apply_env_overrides()?;
        config.session_dir = expand_home_path(&config.session_dir);
        config.database_path = expand_home_path(&config.database_path);
        debug!(?config, "loaded configuration");
        Ok(config)
    }

    /// Load from the default configuration path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read or parsed.
    pub fn load_default() -> anyhow::Result<Self> {
        Self::load(Self::default_path())
    }

    fn apply_env_overrides(&mut self) -> anyhow::Result<()> {
        if let Ok(host) = std::env::var("COMPACTD_HOST") {
            self.host = host;
        }
        if let Ok(port) = std::env::var("COMPACTD_PORT") {
            self.port = port.parse()?;
        }
        if let Ok(dir) = std::env::var("COMPACTD_SESSION_DIR") {
            self.session_dir = PathBuf::from(dir);
        }
        if let Ok(db) = std::env::var("COMPACTD_DATABASE_PATH") {
            self.database_path = PathBuf::from(db);
        }
        if let Ok(max) = std::env::var("COMPACTD_MAX_TURNS") {
            self.max_turns = max.parse()?;
        }
        if let Ok(age) = std::env::var("COMPACTD_ARCHIVE_AGE_HOURS") {
            self.archive_age_hours = age.parse()?;
        }
        if let Ok(th) = std::env::var("COMPACTD_DEDUP_SIMILARITY_THRESHOLD") {
            self.dedup_similarity_threshold = th.parse()?;
        }
        if let Ok(watch) = std::env::var("COMPACTD_WATCH_SESSIONS") {
            self.watch_sessions = watch.parse::<u8>()? != 0;
        }
        Ok(())
    }
}

fn expand_home(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        dirs::home_dir()
            .map(|h| h.join(rest))
            .unwrap_or_else(|| PathBuf::from(path))
    } else {
        PathBuf::from(path)
    }
}

fn expand_home_path(path: &Path) -> PathBuf {
    let s = path.as_os_str().to_string_lossy();
    expand_home(&s)
}

fn default_database_path() -> PathBuf {
    dirs::data_dir()
        .map(|p| p.join("compactd").join("state.db"))
        .unwrap_or_else(|| PathBuf::from("compactd.db"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn default_config_is_sane() {
        let cfg = Config::default();
        assert_eq!(cfg.port, DEFAULT_PORT);
        assert_eq!(cfg.host, DEFAULT_HOST);
        assert!(cfg.max_turns > 0);
    }

    #[test]
    fn load_from_toml_overrides_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        write!(
            f,
            r#"
host = "0.0.0.0"
port = 9999
max_turns = 42
"#
        )
        .unwrap();

        let cfg = Config::load(path).unwrap();
        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.port, 9999);
        assert_eq!(cfg.max_turns, 42);
    }
}
