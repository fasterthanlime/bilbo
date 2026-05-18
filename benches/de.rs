//! `&str` -> `Endpoint`, several ways. serde_json / facet-json know the
//! type at compile time; we recover it from our own DWARF at runtime and
//! JIT a specialized parser. Each bench warms once (JIT compile / cache
//! fill / facet shape init) before timing, so the mean isn't cold-skewed.

use std::mem::MaybeUninit;

use divan::{Bencher, black_box};
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

fn main() {
    divan::main();
}

/// Warm `f` once (discarded) so divan's Tune phase calibrates its
/// sample size on steady-state timing, not a cold first sample (which
/// makes Tune pick sample_size=1 -> 100 noisy single-iter samples).
fn warmed<O>(bencher: Bencher, mut f: impl FnMut() -> O) {
    black_box(f());
    bencher.bench_local(f);
}

#[divan::bench]
fn serde_json(bencher: Bencher) {
    warmed(bencher, || -> Endpoint {
        serde_json::from_str(black_box(JSON)).unwrap()
    });
}

#[divan::bench]
fn facet_json(bencher: Bencher) {
    warmed(bencher, || -> Endpoint {
        facet_json::from_str(black_box(JSON)).unwrap()
    });
}

/// Full ergonomic pipeline: frame-walk + resolve cache + parse + bind.
/// `#[inline(never)]` so `from_json`'s one-frame unwind lands here.
#[divan::bench]
#[inline(never)]
fn dwarf_json(bencher: Bencher) {
    warmed(bencher, || -> Endpoint {
        let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
        unsafe {
            dwarf_json::from_json(
                black_box(JSON),
                &mut e as *mut _ as *mut u8,
            );
            e.assume_init()
        }
    });
}

/// The cranelift parser called directly — apples-to-apples vs
/// `serde_json::from_str`.
#[divan::bench]
#[inline(never)]
fn parser_pure(bencher: Bencher) {
    let mut warm: MaybeUninit<Endpoint> = MaybeUninit::uninit();
    let r = unsafe { dwarf_json::resolve(&mut warm as *mut _ as *mut u8) };
    let pf = *r
        .jit_parser
        .get_or_init(|| dwarf_json::jit::compile_parser(&r.ty));
    warmed(bencher, || -> Endpoint {
        let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
        unsafe {
            pf(&mut e as *mut _ as *mut u8, black_box(JSON).as_ptr(), JSON.len());
            e.assume_init()
        }
    });
}

/// Just our naive parser (no bind), for the breakdown.
#[divan::bench]
fn parse_only(bencher: Bencher) {
    warmed(bencher, || dwarf_json::json::parse(black_box(JSON)));
}

/// Just the interpreter bind step, type + JSON pre-resolved.
#[divan::bench]
#[inline(never)]
fn bind_only_interp(bencher: Bencher) {
    let mut warm: MaybeUninit<Endpoint> = MaybeUninit::uninit();
    let r = unsafe { dwarf_json::resolve(&mut warm as *mut _ as *mut u8) };
    let parsed = dwarf_json::json::parse(JSON);
    warmed(bencher, || -> Endpoint {
        let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
        unsafe {
            dwarf_json::interp::run(
                &mut e as *mut _ as *mut u8,
                &r.ty,
                &parsed,
            );
            e.assume_init()
        }
    });
}
