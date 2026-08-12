# PADAGONIA Integration Roadmap

Follow `/home/sal/padagonia/docs/enterprise-integration-directives.md`.

## Modules

- `compaction_event_adapter`: record input/output snapshot hashes, retained
  concepts, discarded ranges, and token budgets.
- `decision_writer`: persist compaction policy and tool/model provenance.
- `quality_reader`: correlate compaction with later retrieval and task outcomes.
- `retention`: remove raw context and retain only approved measurements.

## Acceptance gates

Compaction is deterministic for fixtures, events are replay-safe, and sensitive
context is not copied into the graph accidentally.
