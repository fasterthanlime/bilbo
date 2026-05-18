//! Demo. `from_json` figures out `Endpoint`'s layout from this binary's own
//! DWARF, guided by the stack frame, then binds the JSON into raw memory.

use tracing::info;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_target(false)
        .without_time()
        .init();

    #[derive(Debug)]
    struct Endpoint {
        host: String,
        port: u16,
        tags: Vec<String>,
    }

    let mut e: std::mem::MaybeUninit<Endpoint> = std::mem::MaybeUninit::uninit();
    // Safety: `e` is exactly the local `ptr` will be matched to.
    unsafe {
        dwarf_json::from_json(
            r#"
            {
              "host": "rustweek.org",
              "port": 443,
              "tags": ["conf", "rust", "🦀"],
            }
        "#,
            &mut e as *mut _ as *mut u8,
        );
    }
    // Safety: yolo
    let e = unsafe { e.assume_init() };
    info!(
        "🎉 interp: host={:?} port={} tags={:?}",
        e.host, e.port, e.tags
    );

    // The cranelift parser that eats raw bytes (no Json tree).
    let mut k: std::mem::MaybeUninit<Endpoint> = std::mem::MaybeUninit::uninit();
    // Safety: `k` is exactly the local `ptr` will be matched to.
    unsafe {
        dwarf_json::from_json_jit_parse(
            r#"{ "host": "parse.jit", "port": 9001, "tags": ["raw","bytes","🥖"], }"#,
            &mut k as *mut _ as *mut u8,
        );
    }
    let k = unsafe { k.assume_init() };
    info!(
        "🎉 jitp:   host={:?} port={} tags={:?}",
        k.host, k.port, k.tags
    );

    // Tagged Option<scalar> round-trip (the new A1 capability).
    #[derive(Debug, PartialEq)]
    struct Opt {
        a: Option<u64>,
        b: Option<i32>,
        c: Option<bool>,
        n: Option<u64>,
        s: Option<String>,
    }
    let mut o: std::mem::MaybeUninit<Opt> = std::mem::MaybeUninit::uninit();
    unsafe {
        dwarf_json::from_json_jit_parse(
            r#"{"a":42,"b":-7,"c":true,"n":null,"s":null}"#,
            &mut o as *mut _ as *mut u8,
        );
    }
    let o = unsafe { o.assume_init() };
    let want = Opt {
        a: Some(42),
        b: Some(-7),
        c: Some(true),
        n: None,
        s: None,
    };
    assert_eq!(o, want, "tagged Option<scalar> round-trip");
    info!("🎉 opt:    {o:?}  (tagged Option OK)");

    // Box<T> round-trip (the new A2 capability): owned heap pointers,
    // through both backends.
    #[derive(Debug, PartialEq)]
    struct Inner {
        x: u32,
        label: String,
    }
    #[derive(Debug, PartialEq)]
    struct BoxDemo {
        id: u64,
        inner: Box<Inner>,
        tail: Box<u32>,
    }
    let want = || BoxDemo {
        id: 7,
        inner: Box::new(Inner {
            x: 99,
            label: "deep".into(),
        }),
        tail: Box::new(1234),
    };
    const BJ: &str =
        r#"{"id":7,"inner":{"x":99,"label":"deep"},"tail":1234}"#;

    let mut bi: std::mem::MaybeUninit<BoxDemo> =
        std::mem::MaybeUninit::uninit();
    unsafe {
        dwarf_json::from_json(BJ, &mut bi as *mut _ as *mut u8);
    }
    let bi = unsafe { bi.assume_init() };
    assert_eq!(bi, want(), "Box<T> round-trip (interp)");
    info!("🎉 box i:  {bi:?}  (Box<T> interp OK)");

    let mut bj: std::mem::MaybeUninit<BoxDemo> =
        std::mem::MaybeUninit::uninit();
    unsafe {
        dwarf_json::from_json_jit_parse(BJ, &mut bj as *mut _ as *mut u8);
    }
    let bj = unsafe { bj.assume_init() };
    assert_eq!(bj, want(), "Box<T> round-trip (jit)");
    info!("🎉 box j:  {bj:?}  (Box<T> jit OK)");

    // Recursive type round-trip (the new A3 capability): a cons-list
    // where `Node` reaches itself through `Option<Box<Node>>`.
    #[derive(Debug, PartialEq)]
    struct Node {
        val: u64,
        next: Option<Box<Node>>,
    }
    let chain = || Node {
        val: 1,
        next: Some(Box::new(Node {
            val: 2,
            next: Some(Box::new(Node { val: 3, next: None })),
        })),
    };
    const NJ: &str =
        r#"{"val":1,"next":{"val":2,"next":{"val":3,"next":null}}}"#;

    let mut ni: std::mem::MaybeUninit<Node> =
        std::mem::MaybeUninit::uninit();
    unsafe {
        dwarf_json::from_json(NJ, &mut ni as *mut _ as *mut u8);
    }
    let ni = unsafe { ni.assume_init() };
    assert_eq!(ni, chain(), "recursive type round-trip (interp)");
    info!("🎉 rec i:  {ni:?}  (recursive interp OK)");

    let mut nj: std::mem::MaybeUninit<Node> =
        std::mem::MaybeUninit::uninit();
    unsafe {
        dwarf_json::from_json_jit_parse(NJ, &mut nj as *mut _ as *mut u8);
    }
    let nj = unsafe { nj.assume_init() };
    assert_eq!(nj, chain(), "recursive type round-trip (jit)");
    info!("🎉 rec j:  {nj:?}  (recursive jit OK)");

    // Profiling mode: print the JIT'd parser's code address and hammer it
    // forever so `stax` can sample + disassemble it.
    //   DWARF_JSON_PROFILE=1 cargo run --release
    if std::env::var_os("DWARF_JSON_PROFILE").is_some() {
        const J: &str =
            r#"{"host":"rustweek.org","port":443,"tags":["conf","rust","crab"]}"#;
        let mut warm: std::mem::MaybeUninit<Endpoint> =
            std::mem::MaybeUninit::uninit();
        let r = unsafe {
            dwarf_json::resolve(&mut warm as *mut _ as *mut u8)
        };
        let f = *r
            .jit_parser
            .get_or_init(|| dwarf_json::jit::compile_parser(&r.ty));
        info!("PARSER @ {:#x}  (stax annotate that address)", f as usize);
        loop {
            let mut e: std::mem::MaybeUninit<Endpoint> =
                std::mem::MaybeUninit::uninit();
            unsafe {
                f(
                    &mut e as *mut _ as *mut u8,
                    J.as_ptr(),
                    J.len(),
                );
                let v = e.assume_init();
                std::hint::black_box(&v);
            }
        }
    }
}
