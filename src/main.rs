use std::mem::MaybeUninit;

fn main() {
    #[derive(Debug)]
    struct Endpoint {
        host: String,
        port: u16,
    }

    let mut e: MaybeUninit<Endpoint> = MaybeUninit::uninit();
    from_json(
        r#"
        {
          "host": "rustweek.org",
          "port": 443,
        }
    "#,
        &mut e as *mut _ as *mut u8,
    );
    // Safety: yolo
    let e = unsafe { e.assume_init() };
    eprintln!("e = {:#?}", e)
}

fn from_json(s: &str, ptr: *mut u8) {
    // use DWARF info to deserialize from JSON since we know
    // what the type being passed in (from the stack trace) is.
    //
    // panics if anything goes wrong. prints every step of the way
    // to stdout since this is a very silly idea
}
