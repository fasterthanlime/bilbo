//! Emit a perf-style `/tmp/jit-<pid>.dump` so profilers (stax, perf,
//! `perf inject`) can name and disassemble our cranelift-compiled code
//! instead of showing `<unresolved>` for anonymous JIT memory.
//!
//! Format: linux/tools/perf/Documentation/jitdump-specification.txt.
//! stax's `JitdumpTailer` reads exactly this; it also has a DYLD-insert
//! preload that notices the target `open()` this path and starts tailing.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const JITDUMP_MAGIC: u32 = 0x4A69_5444; // "JiTD", host-endian
const JIT_CODE_LOAD: u32 = 0;
/// `e_machine` for the jitdump header — perf/stax use it to pick a
/// disassembler, so it must match the host we JIT for.
#[cfg(target_arch = "aarch64")]
const E_MACHINE: u32 = 183; // EM_AARCH64
#[cfg(target_arch = "x86_64")]
const E_MACHINE: u32 = 62; // EM_X86_64

struct State {
    file: File,
    index: u64,
    base: Instant,
}

static STATE: OnceLock<Option<Mutex<State>>> = OnceLock::new();

fn state() -> Option<&'static Mutex<State>> {
    STATE
        .get_or_init(|| {
            let pid = std::process::id();
            let path = format!("/tmp/jit-{pid}.dump");
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .ok()?;

            // 40-byte global header.
            let mut h = Vec::with_capacity(40);
            h.extend_from_slice(&JITDUMP_MAGIC.to_ne_bytes());
            h.extend_from_slice(&1u32.to_ne_bytes()); // version
            h.extend_from_slice(&40u32.to_ne_bytes()); // header size
            h.extend_from_slice(&E_MACHINE.to_ne_bytes());
            h.extend_from_slice(&0u32.to_ne_bytes()); // pad1
            h.extend_from_slice(&pid.to_ne_bytes());
            h.extend_from_slice(&0u64.to_ne_bytes()); // timestamp
            h.extend_from_slice(&0u64.to_ne_bytes()); // flags
            file.write_all(&h).ok()?;

            Some(Mutex::new(State {
                file,
                index: 0,
                base: Instant::now(),
            }))
        })
        .as_ref()
}

/// Announce one JIT'd function: its load address, machine code, and name.
pub fn register(name: &str, addr: u64, code: &[u8]) {
    let Some(lock) = state() else { return };
    let mut st = lock.lock().unwrap();
    let ts = st.base.elapsed().as_nanos() as u64;
    let idx = st.index;
    st.index += 1;

    let pid = std::process::id();
    let name_bytes = name.as_bytes();
    // prefix(16) + pid/tid/vma/code_addr/code_size/code_index(40)
    //            + name + NUL + code
    let total = 16 + 40 + name_bytes.len() + 1 + code.len();

    let mut rec = Vec::with_capacity(total);
    rec.extend_from_slice(&JIT_CODE_LOAD.to_ne_bytes());
    rec.extend_from_slice(&(total as u32).to_ne_bytes());
    rec.extend_from_slice(&ts.to_ne_bytes());
    rec.extend_from_slice(&pid.to_ne_bytes());
    rec.extend_from_slice(&0u32.to_ne_bytes()); // tid
    rec.extend_from_slice(&addr.to_ne_bytes()); // vma
    rec.extend_from_slice(&addr.to_ne_bytes()); // code_addr
    rec.extend_from_slice(&(code.len() as u64).to_ne_bytes());
    rec.extend_from_slice(&idx.to_ne_bytes());
    rec.extend_from_slice(name_bytes);
    rec.push(0);
    rec.extend_from_slice(code);

    let _ = st.file.write_all(&rec);
    let _ = st.file.flush();
}
