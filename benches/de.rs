//! `&str` -> `Endpoint`, several ways. serde_json / facet-json know the
//! type at compile time; we recover it from our own DWARF at runtime and
//! JIT a specialized parser.

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

#[divan::bench]
fn serde_json() -> Endpoint {
    serde_json::from_str(black_box(JSON)).unwrap()
}

#[divan::bench]
fn facet_json() -> Endpoint {
    facet_json::from_str(black_box(JSON)).unwrap()
}

/// Full ergonomic pipeline: frame-walk + resolve cache + parse + bind.
/// (`Bencher`-closure form so `from_json`'s one-frame unwind lands on the
/// closure that actually holds `e`, like criterion's `b.iter`.)
#[divan::bench]
#[inline(never)]
fn dwarf_json(bencher: Bencher) {
    bencher.bench(|| {
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
/// `serde_json::from_str` (both are bytes -> value; serde's type is
/// compile-time, ours a one-time resolve).
#[divan::bench]
#[inline(never)]
fn parser_pure(bencher: Bencher) {
    let mut warm: MaybeUninit<Endpoint> = MaybeUninit::uninit();
    let r = unsafe { dwarf_json::resolve(&mut warm as *mut _ as *mut u8) };
    let pf = *r
        .jit_parser
        .get_or_init(|| dwarf_json::jit::compile_parser(&r.ty));
    bencher.bench(|| {
        let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
        unsafe {
            pf(&mut e as *mut _ as *mut u8, black_box(JSON).as_ptr(), JSON.len());
            e.assume_init()
        }
    });
}

/// Just our naive parser (no bind), for the breakdown.
#[divan::bench]
fn parse_only() -> dwarf_json::json::Json {
    dwarf_json::json::parse(black_box(JSON))
}

/// Just the interpreter bind step, type + JSON pre-resolved.
#[divan::bench]
#[inline(never)]
fn bind_only_interp(bencher: Bencher) {
    let mut warm: MaybeUninit<Endpoint> = MaybeUninit::uninit();
    let r = unsafe { dwarf_json::resolve(&mut warm as *mut _ as *mut u8) };
    let parsed = dwarf_json::json::parse(JSON);
    bencher.bench(|| {
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
