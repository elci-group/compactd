use compactd::{compact_session, Session, Turn};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn build_bloated_session(n_turns: usize) -> Session {
    let mut session = Session::new("bench-session");
    for i in 0..n_turns {
        let mut turn = Turn::new("assistant", &format!("turn content {i}"));
        turn.tool_calls.push(compactd::ToolCall {
            name: "Read".to_string(),
            arguments: format!("{{\"path\": \"file{i}.txt\"}}"),
            output: "duplicate output content".to_string(),
        });
        session.add_turn(turn);
    }
    session
}

fn compaction_hot_path(c: &mut Criterion) {
    c.bench_function("compact_session_1000_turns", |b| {
        b.iter_batched(
            || build_bloated_session(1000),
            |mut session| {
                let result = compact_session(&mut session, 100, 24).unwrap();
                black_box(result);
            },
            criterion::BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, compaction_hot_path);
criterion_main!(benches);
