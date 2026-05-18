//! `canada.json` (~2.3 MB GeoJSON) — a `FeatureCollection` that is, in
//! substance, `Vec<Vec<(f64, f64)>>`: nested arrays of coordinate pairs.
//! Exercises tuples-as-arrays and the `f64` path at scale.

#![allow(non_snake_case)]

use std::mem::MaybeUninit;

use divan::{Bencher, black_box};
use serde::Deserialize;

type Pos = (f64, f64);

#[derive(Deserialize, PartialEq, Debug)]
struct Canada {
    r#type: String,
    features: Vec<Feature>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Feature {
    r#type: String,
    properties: Properties,
    geometry: Geometry,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Properties {
    name: String,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Geometry {
    r#type: String,
    coordinates: Vec<Vec<Pos>>,
}

const J: &str = include_str!("../tests/data/canada.json");

fn main() {
    dwarf_json::debug_init();

    let want: Canada = serde_json::from_str(J).unwrap();
    let mut got: MaybeUninit<Canada> = MaybeUninit::uninit();
    let got = unsafe {
        dwarf_json::from_json_jit_parse(J, &mut got as *mut _ as *mut u8);
        got.assume_init()
    };
    // serde_json's *default* float parser is only best-effort (≈1 ULP);
    // the `float_roundtrip` feature makes it correctly-rounded, matching
    // our `str::parse::<f64>` so this is an exact, fair comparison.
    assert!(
        got == want,
        "JIT parser disagrees with serde_json on canada.json"
    );
    let pts: usize =
        want.features[0].geometry.coordinates.iter().map(Vec::len).sum();
    eprintln!(
        "canada OK — exact match: {} rings, {} points",
        want.features[0].geometry.coordinates.len(),
        pts
    );
    drop(got);

    divan::main();
}

/// Warm `f` once (discarded) so divan's Tune phase calibrates its sample
/// size on steady-state timing, not a cold first sample.
fn warmed<O>(bencher: Bencher, mut f: impl FnMut() -> O) {
    black_box(f());
    bencher.bench_local(f);
}

#[divan::bench]
fn serde_json(bencher: Bencher) {
    warmed(bencher, || -> Canada {
        serde_json::from_str(black_box(J)).unwrap()
    });
}

#[divan::bench]
#[inline(never)]
fn parser_pure(bencher: Bencher) {
    let mut warm: MaybeUninit<Canada> = MaybeUninit::uninit();
    let r = unsafe { dwarf_json::resolve(&mut warm as *mut _ as *mut u8) };
    let pf = *r
        .jit_parser
        .get_or_init(|| dwarf_json::jit::compile_parser(&r.ty));
    warmed(bencher, || -> Canada {
        let mut e: MaybeUninit<Canada> = MaybeUninit::uninit();
        unsafe {
            pf(&mut e as *mut _ as *mut u8, black_box(J).as_ptr(), J.len());
            e.assume_init()
        }
    });
}
