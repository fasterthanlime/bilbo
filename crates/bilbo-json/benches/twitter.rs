//! `twitter.json` (~632 KB) — the full-fidelity test. This is where every
//! capability lands at once: tagged `Option<bool>`/`Option<i64>` (A1),
//! `Box<Status>` heap pointers (A2), and a genuinely *recursive* type —
//! `Status::retweeted_status: Option<Box<Status>>` (A3). Plus the serde
//! semantic "an absent `Option` field is `None`", which both contenders
//! must agree on. Same owned Rust types for both: serde knows them at
//! compile time; we recover them from this binary's own DWARF and JIT a
//! parser specialized to exactly this shape.

use std::mem::MaybeUninit;

use divan::{Bencher, black_box};
use serde::Deserialize;

#[derive(Deserialize, PartialEq, Debug)]
struct Twitter {
    statuses: Vec<Status>,
    search_metadata: SearchMetadata,
}

#[derive(Deserialize, PartialEq, Debug)]
struct SearchMetadata {
    completed_in: f64,
    max_id: u64,
    max_id_str: String,
    next_results: String,
    query: String,
    refresh_url: String,
    count: u64,
    since_id: u64,
    since_id_str: String,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Status {
    // Always `null` in this corpus; typed `Option` so both sides agree.
    contributors: Option<u64>,
    coordinates: Option<u64>,
    created_at: String,
    entities: Entities,
    favorite_count: u64,
    favorited: bool,
    geo: Option<u64>,
    id: u64,
    id_str: String,
    in_reply_to_screen_name: Option<String>,
    in_reply_to_status_id: Option<u64>,
    in_reply_to_status_id_str: Option<String>,
    in_reply_to_user_id: Option<u64>,
    in_reply_to_user_id_str: Option<String>,
    lang: String,
    metadata: Metadata,
    place: Option<u64>,
    // Sometimes absent -> `None` (A1 tagged Option<bool>).
    possibly_sensitive: Option<bool>,
    retweet_count: u64,
    retweeted: bool,
    // Sometimes absent; recursive (A2 Box + A3 cycle).
    retweeted_status: Option<Box<Status>>,
    source: String,
    text: String,
    truncated: bool,
    user: User,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Metadata {
    iso_language_code: String,
    result_type: String,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Entities {
    hashtags: Vec<Hashtag>,
    // Always `[]` here; element type is never instantiated.
    symbols: Vec<String>,
    urls: Vec<UrlEntity>,
    user_mentions: Vec<UserMention>,
    // Sometimes absent.
    media: Option<Vec<Media>>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Hashtag {
    text: String,
    indices: (u32, u32),
}

#[derive(Deserialize, PartialEq, Debug)]
struct UrlEntity {
    url: String,
    expanded_url: String,
    display_url: String,
    indices: (u32, u32),
}

#[derive(Deserialize, PartialEq, Debug)]
struct UserMention {
    screen_name: String,
    name: String,
    id: u64,
    id_str: String,
    indices: (u32, u32),
}

#[derive(Deserialize, PartialEq, Debug)]
struct Media {
    id: u64,
    id_str: String,
    indices: (u32, u32),
    media_url: String,
    media_url_https: String,
    url: String,
    display_url: String,
    expanded_url: String,
    #[serde(rename = "type")]
    r#type: String,
    sizes: Sizes,
    // Sometimes absent.
    source_status_id: Option<u64>,
    source_status_id_str: Option<String>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Sizes {
    medium: Size,
    large: Size,
    thumb: Size,
    small: Size,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Size {
    w: u32,
    h: u32,
    resize: String,
}

#[derive(Deserialize, PartialEq, Debug)]
struct User {
    id: u64,
    id_str: String,
    name: String,
    screen_name: String,
    location: String,
    description: String,
    url: Option<String>,
    entities: UserEntities,
    protected: bool,
    followers_count: u64,
    friends_count: u64,
    listed_count: u64,
    created_at: String,
    favourites_count: u64,
    utc_offset: Option<i64>,
    time_zone: Option<String>,
    geo_enabled: bool,
    verified: bool,
    statuses_count: u64,
    lang: String,
    contributors_enabled: bool,
    is_translator: bool,
    is_translation_enabled: bool,
    profile_background_color: String,
    profile_background_image_url: String,
    profile_background_image_url_https: String,
    profile_background_tile: bool,
    profile_image_url: String,
    profile_image_url_https: String,
    // Sometimes absent.
    profile_banner_url: Option<String>,
    profile_link_color: String,
    profile_sidebar_border_color: String,
    profile_sidebar_fill_color: String,
    profile_text_color: String,
    profile_use_background_image: bool,
    default_profile: bool,
    default_profile_image: bool,
    following: bool,
    follow_request_sent: bool,
    notifications: bool,
}

#[derive(Deserialize, PartialEq, Debug)]
struct UserEntities {
    description: UrlList,
    // Sometimes absent.
    url: Option<UrlList>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct UrlList {
    urls: Vec<UrlEntity>,
}

const J: &str = include_str!("../tests/data/twitter.json");

fn main() {
    bilbo_json::debug_init();

    // Correctness gate: our parser must produce exactly what serde does,
    // including the recursive `retweeted_status` and every absent-Option.
    let want: Twitter = serde_json::from_str(J).unwrap();
    let mut got: MaybeUninit<Twitter> = MaybeUninit::uninit();
    let got = unsafe {
        bilbo_json::from_json_jit_parse(J, &mut got as *mut _ as *mut u8);
        got.assume_init()
    };
    assert!(
        got == want,
        "JIT parser disagrees with serde_json on twitter.json"
    );
    let retweets = want
        .statuses
        .iter()
        .filter(|s| s.retweeted_status.is_some())
        .count();
    eprintln!(
        "twitter OK — exact match: {} statuses, {} with retweeted_status",
        want.statuses.len(),
        retweets
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
    warmed(bencher, || -> Twitter {
        serde_json::from_str(black_box(J)).unwrap()
    });
}

/// One ergonomic `from_json_jit_parse` call in a real `#[inline(never)]`
/// frame, with the `MaybeUninit<T>` *used in place* — not returned by
/// value. (Returning it by value lets NRVO forward the caller's sret
/// slot and elide the local entirely, leaving no DIE to resolve; that's
/// the case the Box workaround was dodging. Using it in-frame, exactly
/// like the correctness gate's `main`, keeps a real materialized local
/// and the cross-CU definition index handles any stub types.)
#[inline(never)]
fn ergonomic(j: &str) {
    let mut e: MaybeUninit<Twitter> = MaybeUninit::uninit();
    let v = unsafe {
        bilbo_json::from_json_jit_parse(j, &mut e as *mut _ as *mut u8);
        e.assume_init()
    };
    black_box(&v);
}

/// Apples-to-apples vs `serde_json::from_str`: the ergonomic entry
/// point. Every call pays frame capture + the two-level cache lookup
/// (L1 hit: callsite -> type -> JIT'd parser), then runs that parser.
#[divan::bench]
#[inline(never)]
fn bilbo_json(bencher: Bencher) {
    warmed(bencher, || ergonomic(black_box(J)));
}

#[divan::bench]
#[inline(never)]
fn parser_pure(bencher: Bencher) {
    let mut warm: MaybeUninit<Twitter> = MaybeUninit::uninit();
    let r = unsafe { bilbo_json::resolve(&mut warm as *mut _ as *mut u8) };
    let pf = *r.ext(bilbo_json::jit::compile_parser);
    warmed(bencher, || -> Twitter {
        let mut e: MaybeUninit<Twitter> = MaybeUninit::uninit();
        unsafe {
            pf(&mut e as *mut _ as *mut u8, black_box(J).as_ptr(), J.len());
            e.assume_init()
        }
    });
}
