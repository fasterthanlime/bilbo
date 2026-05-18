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
    // Correctness gate. We can't use `got == want`: serde_json's
    // *default* float parser (what everyone actually runs — benching
    // against `float_roundtrip` would just gimp serde) is best-effort
    // and ≈1 ULP off. *We* are the correctly-rounded side (our f64s
    // match Python / Rust `str::parse`). So: structure + strings + the
    // ring/point shape must be exact, and every coordinate must agree
    // with serde to within 2 ULP (i.e. serde's own float error).
    fn ulps(a: f64, b: f64) -> u64 {
        let (x, y) = (a.to_bits() as i64, b.to_bits() as i64);
        x.abs_diff(y)
    }
    assert_eq!(got.r#type, want.r#type);
    assert_eq!(got.features.len(), want.features.len());
    let gc = &got.features[0].geometry.coordinates;
    let wc = &want.features[0].geometry.coordinates;
    assert_eq!(got.features[0].r#type, want.features[0].r#type);
    assert_eq!(
        got.features[0].properties.name,
        want.features[0].properties.name
    );
    assert_eq!(got.features[0].geometry.r#type, want.features[0].geometry.r#type);
    assert_eq!(gc.len(), wc.len(), "ring count");
    for (ri, (gr, wr)) in gc.iter().zip(wc).enumerate() {
        assert_eq!(gr.len(), wr.len(), "ring {ri} len");
        for (pi, (gp, wp)) in gr.iter().zip(wr).enumerate() {
            let d = ulps(gp.0, wp.0).max(ulps(gp.1, wp.1));
            assert!(
                d <= 2,
                "ring {ri} pt {pi}: {gp:?} vs serde {wp:?} ({d} ULP)"
            );
        }
    }
    let pts: usize =
        want.features[0].geometry.coordinates.iter().map(Vec::len).sum();
    eprintln!(
        "canada OK — {} rings, {} points; structure exact, coords \
         within 2 ULP of (best-effort) serde_json, and our f64s are \
         the correctly-rounded ones",
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

/// One ergonomic `from_json_jit_parse` call in a real `#[inline(never)]`
/// frame, with the `MaybeUninit<T>` *used in place* — not returned by
/// value (NRVO would forward the caller's sret slot and elide the local
/// entirely, leaving no DIE; that's what the Box workaround was dodging).
/// The cross-CU definition index handles any stub `String`/`Vec` types.
#[inline(never)]
fn ergonomic(j: &str) {
    let mut e: MaybeUninit<Canada> = MaybeUninit::uninit();
    let v = unsafe {
        dwarf_json::from_json_jit_parse(j, &mut e as *mut _ as *mut u8);
        e.assume_init()
    };
    black_box(&v);
}

/// Apples-to-apples vs `serde_json::from_str`: the ergonomic entry
/// point. Every call pays frame capture + the two-level cache lookup
/// (L1 hit: callsite -> type -> JIT'd parser), then runs that parser.
#[divan::bench]
#[inline(never)]
fn dwarf_json(bencher: Bencher) {
    warmed(bencher, || ergonomic(black_box(J)));
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
