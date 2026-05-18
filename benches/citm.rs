//! The real test: `citm_catalog.json` (~1.7 MB) — nested structs, `Vec`s,
//! `Option<String>`, `()` nulls, and `BTreeMap`s with dynamic string keys.
//! Same owned Rust types for both contenders; serde knows them at compile
//! time, we recover them from our own DWARF and JIT a specialized parser.

#![allow(non_snake_case)]

use std::collections::BTreeMap;
use std::mem::MaybeUninit;

use divan::{Bencher, black_box};
use serde::Deserialize;

type Map<V> = BTreeMap<String, V>;

#[derive(Deserialize, PartialEq, Debug)]
struct Citm {
    areaNames: Map<String>,
    audienceSubCategoryNames: Map<String>,
    blockNames: Map<String>,
    events: Map<Event>,
    performances: Vec<Performance>,
    seatCategoryNames: Map<String>,
    subTopicNames: Map<String>,
    subjectNames: Map<String>,
    topicNames: Map<String>,
    topicSubTopics: Map<Vec<u32>>,
    venueNames: Map<String>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Event {
    description: (),
    id: u32,
    logo: Option<String>,
    name: String,
    subTopicIds: Vec<u32>,
    subjectCode: (),
    subtitle: (),
    topicIds: Vec<u32>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Performance {
    eventId: u32,
    id: u32,
    logo: Option<String>,
    name: (),
    prices: Vec<Price>,
    seatCategories: Vec<SeatCategory>,
    seatMapImage: (),
    start: u64,
    venueCode: String,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Price {
    amount: u32,
    audienceSubCategoryId: u32,
    seatCategoryId: u32,
}

#[derive(Deserialize, PartialEq, Debug)]
struct SeatCategory {
    areas: Vec<Area>,
    seatCategoryId: u32,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Area {
    areaId: u32,
    blockIds: Vec<u32>,
}

const J: &str = include_str!("../tests/data/citm_catalog.json");

fn main() {
    dwarf_json::debug_init();
    // The one honest caveat: the real std BTreeMap is built via trampolines,
    // monomorphized once per value type.
    dwarf_json::tramp::force::<String>();
    dwarf_json::tramp::force::<Event>();
    dwarf_json::tramp::force::<Vec<u32>>();

    // Correctness gate: our parser must produce exactly what serde does.
    let want: Citm = serde_json::from_str(J).unwrap();
    let mut got: MaybeUninit<Citm> = MaybeUninit::uninit();
    let got = unsafe {
        dwarf_json::from_json_jit_parse(J, &mut got as *mut _ as *mut u8);
        got.assume_init()
    };
    assert!(
        got == want,
        "JIT parser disagrees with serde_json on citm_catalog.json"
    );
    eprintln!(
        "citm OK — exact match: {} events, {} performances",
        want.events.len(),
        want.performances.len()
    );
    drop(got);

    divan::main();
}

/// Warm `f` once (discarded) so divan's Tune phase calibrates its sample
/// size on steady-state timing, not a cold first sample (which makes Tune
/// pick sample_size=1 -> 100 noisy single-iter samples).
fn warmed<O>(bencher: Bencher, mut f: impl FnMut() -> O) {
    black_box(f());
    bencher.bench_local(f);
}

#[divan::bench]
fn serde_json(bencher: Bencher) {
    warmed(bencher, || -> Citm {
        serde_json::from_str(black_box(J)).unwrap()
    });
}

#[divan::bench]
#[inline(never)]
fn parser_pure(bencher: Bencher) {
    let mut warm: MaybeUninit<Citm> = MaybeUninit::uninit();
    let r = unsafe { dwarf_json::resolve(&mut warm as *mut _ as *mut u8) };
    let pf = *r
        .jit_parser
        .get_or_init(|| dwarf_json::jit::compile_parser(&r.ty));
    warmed(bencher, || -> Citm {
        let mut e: MaybeUninit<Citm> = MaybeUninit::uninit();
        unsafe {
            pf(&mut e as *mut _ as *mut u8, black_box(J).as_ptr(), J.len());
            e.assume_init()
        }
    });
}
