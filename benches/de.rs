//! `&str` -> `Endpoint`, four ways. serde_json and facet-json know the type
//! at compile time; we recover it from our own DWARF at runtime (cached).

use std::mem::MaybeUninit;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
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
            unsafe {
                dwarf_json::from_json(
                    black_box(JSON),
                    &mut e as *mut _ as *mut u8,
                );
                black_box(e.assume_init());
            }
        })
    });

    g.bench_function("dwarf_json_jit", |b| {
        b.iter(|| {
            let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
            unsafe {
                dwarf_json::from_json_jit(
                    black_box(JSON),
                    &mut e as *mut _ as *mut u8,
                );
                black_box(e.assume_init());
            }
        })
    });

    // --- isolate where our time goes ---------------------------------

    g.bench_function("parse_only", |b| {
        b.iter(|| black_box(dwarf_json::json::parse(black_box(JSON))))
    });

    {
        // Resolve + parse once; measure only the bind step.
        let mut warm: MaybeUninit<Endpoint> = MaybeUninit::uninit();
        let plan =
            unsafe { dwarf_json::resolve(&mut warm as *mut _ as *mut u8) };
        let parsed = dwarf_json::json::parse(JSON);

        g.bench_function("bind_only_interp", |b| {
            b.iter(|| {
                let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
                unsafe {
                    dwarf_json::interp::run(
                        &mut e as *mut _ as *mut u8,
                        &plan,
                        &parsed,
                    );
                    black_box(e.assume_init());
                }
            })
        });

        let f = dwarf_json::jit::compiled(0xB17, &plan);
        g.bench_function("bind_only_jit", |b| {
            b.iter(|| {
                let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
                unsafe {
                    f(&mut e as *mut _ as *mut u8, &parsed as *const _);
                    black_box(e.assume_init());
                }
            })
        });
    }

    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
