//! The cranelift backend. Given a [`Ty`], compile a *specialized* native
//! function `extern "C" fn(*mut u8, *const Json)` with every offset, size and
//! field key baked in as a constant. Structural work (field dispatch, the
//! `Vec` loop) is emitted as machine code; the leaf operations that need Rust
//! (JSON access, allocation) are `extern "C"` shims. Compiled once per call
//! site, then it's just an indirect call.

use std::cell::RefCell;
use std::collections::HashMap;

use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};

use crate::json::Json;
use crate::plan::{SeqLayout, Ty};

/// `extern "C" fn(dst: *mut u8, json: *const Json)`.
pub type Compiled = unsafe extern "C" fn(*mut u8, *const Json);

thread_local! {
    static JIT: RefCell<Jit> = RefCell::new(Jit::new());
}

/// Compile a specialized binder for `ty`. Caching by type lives in
/// `resolve` (one `Resolved` per type holds the result), so this just
/// emits code; the thread-local `JITModule` keeps it mapped.
pub fn compile(ty: &Ty) -> Compiled {
    JIT.with(|j| j.borrow_mut().compile(ty))
}

struct Jit {
    module: JITModule,
    shims: Shims,
    seq: u32,
}

/// `FuncId`s of the runtime shims, declared once in the module.
#[derive(Clone, Copy)]
struct Shims {
    obj_get: FuncId,
    arr_len: FuncId,
    arr_get: FuncId,
    as_u64: FuncId,
    as_i64: FuncId,
    as_f64: FuncId,
    as_bool: FuncId,
    as_char: FuncId,
    str_ptr: FuncId,
    str_len: FuncId,
    dup: FuncId,
    alloc: FuncId,
}

impl Jit {
    fn new() -> Self {
        let mut b = JITBuilder::new(default_libcall_names())
            .expect("jit builder");
        b.symbol("rt_obj_get", rt_obj_get as *const u8);
        b.symbol("rt_arr_len", rt_arr_len as *const u8);
        b.symbol("rt_arr_get", rt_arr_get as *const u8);
        b.symbol("rt_as_u64", rt_as_u64 as *const u8);
        b.symbol("rt_as_i64", rt_as_i64 as *const u8);
        b.symbol("rt_as_f64", rt_as_f64 as *const u8);
        b.symbol("rt_as_bool", rt_as_bool as *const u8);
        b.symbol("rt_as_char", rt_as_char as *const u8);
        b.symbol("rt_str_ptr", rt_str_ptr as *const u8);
        b.symbol("rt_str_len", rt_str_len as *const u8);
        b.symbol("rt_dup", rt_dup as *const u8);
        b.symbol("rt_alloc", rt_alloc as *const u8);
        let mut module = JITModule::new(b);

        let p = types::I64;
        let sig = |m: &JITModule, params: &[_], ret: Option<_>| {
            let mut s = m.make_signature();
            for &t in params {
                s.params.push(AbiParam::new(t));
            }
            if let Some(r) = ret {
                s.returns.push(AbiParam::new(r));
            }
            s
        };
        let decl = |m: &mut JITModule, name: &str, ps: &[_], r| {
            let s = sig(m, ps, r);
            m.declare_function(name, Linkage::Import, &s).unwrap()
        };
        let shims = Shims {
            obj_get: decl(&mut module, "rt_obj_get", &[p, p, p], Some(p)),
            arr_len: decl(&mut module, "rt_arr_len", &[p], Some(p)),
            arr_get: decl(&mut module, "rt_arr_get", &[p, p], Some(p)),
            as_u64: decl(&mut module, "rt_as_u64", &[p], Some(p)),
            as_i64: decl(&mut module, "rt_as_i64", &[p], Some(p)),
            as_f64: decl(&mut module, "rt_as_f64", &[p], Some(types::F64)),
            as_bool: decl(&mut module, "rt_as_bool", &[p], Some(p)),
            as_char: decl(&mut module, "rt_as_char", &[p], Some(p)),
            str_ptr: decl(&mut module, "rt_str_ptr", &[p], Some(p)),
            str_len: decl(&mut module, "rt_str_len", &[p], Some(p)),
            dup: decl(&mut module, "rt_dup", &[p, p], Some(p)),
            alloc: decl(&mut module, "rt_alloc", &[p, p], Some(p)),
        };

        Jit {
            module,
            shims,
            seq: 0,
        }
    }

    fn compile(&mut self, ty: &Ty) -> Compiled {
        let p = types::I64;
        let mut ctx = self.module.make_context();
        ctx.func.signature.params.push(AbiParam::new(p)); // dst
        ctx.func.signature.params.push(AbiParam::new(p)); // json

        self.seq += 1;
        let name = format!("bind_{}", self.seq);
        let fid = self
            .module
            .declare_function(&name, Linkage::Local, &ctx.func.signature)
            .unwrap();

        let mut fctx = FunctionBuilderContext::new();
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            let dst = b.block_params(entry)[0];
            let json = b.block_params(entry)[1];

            let mut e = Emit {
                module: &mut self.module,
                shims: &self.shims,
                b: &mut b,
            };
            e.value(ty, dst, json);

            b.ins().return_(&[]);
            b.seal_all_blocks();
            b.finalize();
        }

        self.module.define_function(fid, &mut ctx).unwrap();
        self.module.clear_context(&mut ctx);
        self.module.finalize_definitions().unwrap();
        let code = self.module.get_finalized_function(fid);
        unsafe { std::mem::transmute::<*const u8, Compiled>(code) }
    }
}

/// Carries the cranelift builder + shim refs through the recursive emit.
struct Emit<'a, 'b> {
    module: &'a mut JITModule,
    shims: &'a Shims,
    b: &'a mut FunctionBuilder<'b>,
}

impl Emit<'_, '_> {
    fn call1(&mut self, f: FuncId, a: Value) -> Value {
        let r = self.module.declare_func_in_func(f, self.b.func);
        let c = self.b.ins().call(r, &[a]);
        self.b.inst_results(c)[0]
    }
    fn call2(&mut self, f: FuncId, a: Value, b: Value) -> Value {
        let r = self.module.declare_func_in_func(f, self.b.func);
        let c = self.b.ins().call(r, &[a, b]);
        self.b.inst_results(c)[0]
    }
    fn call3(&mut self, f: FuncId, a: Value, b: Value, c: Value) -> Value {
        let r = self.module.declare_func_in_func(f, self.b.func);
        let ci = self.b.ins().call(r, &[a, b, c]);
        self.b.inst_results(ci)[0]
    }
    fn iconst(&mut self, v: i64) -> Value {
        self.b.ins().iconst(types::I64, v)
    }

    /// Emit code to bind `json` into `dst` according to `ty`.
    fn value(&mut self, ty: &Ty, dst: Value, json: Value) {
        let trusted = MemFlags::trusted();
        match ty {
            Ty::Bool => {
                let v = self.call1(self.shims.as_bool, json);
                let v = self.b.ins().ireduce(types::I8, v);
                self.b.ins().store(trusted, v, dst, 0);
            }
            Ty::Char => {
                let v = self.call1(self.shims.as_char, json);
                let v = self.b.ins().ireduce(types::I32, v);
                self.b.ins().store(trusted, v, dst, 0);
            }
            Ty::U(n) | Ty::I(n) => {
                let signed = matches!(ty, Ty::I(_));
                let v = if signed {
                    self.call1(self.shims.as_i64, json)
                } else {
                    self.call1(self.shims.as_u64, json)
                };
                let t = match n {
                    1 => types::I8,
                    2 => types::I16,
                    4 => types::I32,
                    _ => types::I64,
                };
                let v = if t == types::I64 {
                    v
                } else {
                    self.b.ins().ireduce(t, v)
                };
                self.b.ins().store(trusted, v, dst, 0);
            }
            Ty::F32 => {
                let v = self.call1(self.shims.as_f64, json);
                let v = self.b.ins().fdemote(types::F32, v);
                self.b.ins().store(trusted, v, dst, 0);
            }
            Ty::F64 => {
                let v = self.call1(self.shims.as_f64, json);
                self.b.ins().store(trusted, v, dst, 0);
            }
            Ty::Str(seq) => self.emit_str(*seq, dst, json),
            Ty::StrRef { ptr_off, len_off } => {
                let p = self.call1(self.shims.str_ptr, json);
                let n = self.call1(self.shims.str_len, json);
                let dup = self.call2(self.shims.dup, p, n);
                self.store_at(dst, *ptr_off, dup);
                self.store_at(dst, *len_off, n);
            }
            Ty::Struct { fields, .. } => {
                for f in fields {
                    let fdst = {
                        let off = self.iconst(f.offset as i64);
                        self.b.ins().iadd(dst, off)
                    };
                    let (kp, kl) = leak_key(&f.name);
                    let kp = self.iconst(kp as i64);
                    let kl = self.iconst(kl as i64);
                    let jn = self.call3(self.shims.obj_get, json, kp, kl);
                    self.value(&f.ty, fdst, jn);
                }
            }
            Ty::Vec {
                elem,
                elem_size,
                elem_align,
                seq,
            } => self.emit_vec(elem, *elem_size, *elem_align, seq, dst, json),
            Ty::Unknown(w) => panic!("jit: unknown type `{w}`"),
        }
    }

    fn store_at(&mut self, base: Value, off: usize, v: Value) {
        // off is small and constant; fold it into the store.
        self.b.ins().store(MemFlags::trusted(), v, base, off as i32);
    }

    fn emit_str(&mut self, seq: SeqLayout, dst: Value, json: Value) {
        let p = self.call1(self.shims.str_ptr, json);
        let n = self.call1(self.shims.str_len, json);
        let buf = self.call2(self.shims.dup, p, n);
        self.store_at(dst, seq.ptr_off, buf);
        self.store_at(dst, seq.cap_off, n);
        self.store_at(dst, seq.len_off, n);
    }

    fn emit_vec(
        &mut self,
        elem: &Ty,
        elem_size: u64,
        elem_align: u64,
        seq: &SeqLayout,
        dst: Value,
        json: Value,
    ) {
        let esz = self.iconst(elem_size as i64);
        let n = self.call1(self.shims.arr_len, json);

        // base = n == 0 ? align : alloc(n*esz, align)
        let total = self.b.ins().imul(n, esz);
        let align = self.iconst(elem_align.max(1) as i64);
        let buf = self.call2(self.shims.alloc, total, align);
        // alloc() returns the aligned dangling pointer when size == 0, so we
        // don't even need a branch here.

        // for i in 0..n { bind elem at buf + i*esz }
        let header = self.b.create_block();
        self.b.append_block_param(header, types::I64); // i
        let body = self.b.create_block();
        let cont = self.b.create_block();

        let zero = self.iconst(0);
        self.b.ins().jump(header, &[zero.into()]);

        self.b.switch_to_block(header);
        let i = self.b.block_params(header)[0];
        let done =
            self.b
                .ins()
                .icmp(cranelift_codegen::ir::condcodes::IntCC::UnsignedGreaterThanOrEqual, i, n);
        self.b.ins().brif(done, cont, &[], body, &[]);

        self.b.switch_to_block(body);
        let off = self.b.ins().imul(i, esz);
        let edst = self.b.ins().iadd(buf, off);
        let ej = self.call2(self.shims.arr_get, json, i);
        self.value(elem, edst, ej);
        let one = self.iconst(1);
        let next = self.b.ins().iadd(i, one);
        self.b.ins().jump(header, &[next.into()]);

        self.b.switch_to_block(cont);
        self.store_at(dst, seq.ptr_off, buf);
        self.store_at(dst, seq.cap_off, n);
        self.store_at(dst, seq.len_off, n);
    }
}

/// Field keys must outlive every call into the JIT'd code; leak them once.
fn leak_key(name: &str) -> (*const u8, usize) {
    use std::sync::Mutex;
    static KEYS: Mutex<Option<HashMap<String, &'static [u8]>>> =
        Mutex::new(None);
    let mut g = KEYS.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    let s = map
        .entry(name.to_string())
        .or_insert_with(|| {
            Box::leak(name.as_bytes().to_vec().into_boxed_slice())
        });
    (s.as_ptr(), s.len())
}

// --- runtime shims -------------------------------------------------------

unsafe extern "C" fn rt_obj_get(
    j: *const Json,
    k: *const u8,
    klen: usize,
) -> *const Json {
    let j = unsafe { &*j };
    let key = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(k, klen)) };
    j.get(key).expect("jit: missing JSON key") as *const Json
}

unsafe extern "C" fn rt_arr_len(j: *const Json) -> usize {
    match unsafe { &*j } {
        Json::Array(v) => v.len(),
        other => panic!("jit: expected array, got {other:?}"),
    }
}

unsafe extern "C" fn rt_arr_get(j: *const Json, i: usize) -> *const Json {
    match unsafe { &*j } {
        Json::Array(v) => &v[i] as *const Json,
        other => panic!("jit: expected array, got {other:?}"),
    }
}

unsafe extern "C" fn rt_as_u64(j: *const Json) -> u64 {
    unsafe { &*j }.as_u128() as u64
}
unsafe extern "C" fn rt_as_i64(j: *const Json) -> i64 {
    unsafe { &*j }.as_i128() as i64
}
unsafe extern "C" fn rt_as_f64(j: *const Json) -> f64 {
    unsafe { &*j }.as_f64()
}
unsafe extern "C" fn rt_as_bool(j: *const Json) -> u64 {
    matches!(unsafe { &*j }, Json::Bool(true)) as u64
}
unsafe extern "C" fn rt_as_char(j: *const Json) -> u64 {
    unsafe { &*j }.as_str().chars().next().unwrap_or('\0') as u64
}
unsafe extern "C" fn rt_str_ptr(j: *const Json) -> *const u8 {
    unsafe { &*j }.as_str().as_ptr()
}
unsafe extern "C" fn rt_str_len(j: *const Json) -> usize {
    unsafe { &*j }.as_str().len()
}

/// Allocate + copy `n` bytes (matches `Global`; `Drop` frees with cap==len).
unsafe extern "C" fn rt_dup(src: *const u8, n: usize) -> *mut u8 {
    if n == 0 {
        return std::ptr::without_provenance_mut(1);
    }
    let layout = std::alloc::Layout::from_size_align(n, 1).unwrap();
    let p = unsafe { std::alloc::alloc(layout) };
    assert!(!p.is_null());
    unsafe { std::ptr::copy_nonoverlapping(src, p, n) };
    p
}

unsafe extern "C" fn rt_alloc(size: usize, align: usize) -> *mut u8 {
    let align = align.max(1);
    if size == 0 {
        return std::ptr::without_provenance_mut(align);
    }
    let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
    let p = unsafe { std::alloc::alloc(layout) };
    assert!(!p.is_null());
    p
}
