//! `&str` -> `Endpoint`, four ways. serde_json and facet-json know the type
//! at compile time; we recover it from our own DWARF at runtime (cached).

use std::mem::MaybeUninit;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use facet::Facet;
use serde::Deserialize;

#[derive(Debug, Deserialize, Facet)]
struct Endpoint {
    host: String,
    port: u16,
    tags: Vec<String>,
}

const JSON: &str =
    r#"{"host":"rustweek.org","port":443,"tags":["conf","rust","crab"]}"#;

fn bench(c: &mut Criterion) {
    dwarf_json::debug_init();
    let mut g = c.benchmark_group("endpoint");

    g.bench_function("serde_json", |b| {
        b.iter(|| {
            let e: Endpoint = serde_json::from_str(black_box(JSON)).unwrap();
            black_box(e);
        })
    });

    g.bench_function("facet_json", |b| {
        b.iter(|| {
            let e: Endpoint = facet_json::from_str(black_box(JSON)).unwrap();
            black_box(e);
        })
    });

    g.bench_function("dwarf_json_interp", |b| {
        b.iter(|| {
            let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
            dwarf_json::from_json(
                black_box(JSON),
                &mut e as *mut _ as *mut u8,
            );
            black_box(unsafe { e.assume_init() });
        })
    });

    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
