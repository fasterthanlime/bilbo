//! The cranelift backend. Given a [`Ty`], compile a *specialized* native
//! function `extern "C" fn(*mut u8, *const Json)` with every offset, size and
//! field key baked in as a constant. Structural work (field dispatch, the
//! `Vec` loop) is emitted as machine code; the leaf operations that need Rust
//! (JSON access, allocation) are `extern "C"` shims. Compiled once per call
//! site, then it's just an indirect call.

use std::cell::RefCell;
use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value, types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};

use crate::json::Json;
use crate::plan::{SeqLayout, Ty};

/// `extern "C" fn(dst: *mut u8, json: *const Json)` — binds a pre-parsed
/// `Json` tree (the tree-walking JIT).
pub type Compiled = unsafe extern "C" fn(*mut u8, *const Json);

/// `extern "C" fn(dst: *mut u8, input: *const u8, len: usize)` — parses raw
/// JSON bytes straight into the struct, no `Json` tree at all.
pub type Parser = unsafe extern "C" fn(*mut u8, *const u8, usize);

thread_local! {
    static JIT: RefCell<Jit> = RefCell::new(Jit::new());
}

/// Compile a specialized binder for `ty`. Caching by type lives in
/// `resolve` (one `Resolved` per type holds the result), so this just
/// emits code; the thread-local `JITModule` keeps it mapped.
pub fn compile(ty: &Ty) -> Compiled {
    JIT.with(|j| j.borrow_mut().compile(ty))
}

/// Compile a specialized raw-bytes parser for `ty`.
pub fn compile_parser(ty: &Ty) -> Parser {
    JIT.with(|j| j.borrow_mut().compile_parser(ty))
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
    // parser-JIT shims. Byte scanning is emitted inline in cranelift now;
    // only the genuinely-Rust bits remain: `skip` (unknown key / array
    // count) and `f64v` (float parsing).
    skip: FuncId,
    f64v: FuncId,
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
        b.symbol("rt_skip", rt_skip as *const u8);
        b.symbol("rt_f64v", rt_f64v as *const u8);
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
            skip: decl(&mut module, "rt_skip", &[p, p], Some(p)),
            f64v: decl(&mut module, "rt_f64v", &[p, p, p], Some(p)),
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

    fn compile_parser(&mut self, ty: &Ty) -> Parser {
        let p = types::I64;
        let mut ctx = self.module.make_context();
        ctx.func.signature.params.push(AbiParam::new(p)); // dst
        ctx.func.signature.params.push(AbiParam::new(p)); // input ptr
        ctx.func.signature.params.push(AbiParam::new(p)); // len

        self.seq += 1;
        let name = format!("parse_{}", self.seq);
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
            let ptr = b.block_params(entry)[1];
            let len = b.block_params(entry)[2];
            let end = b.ins().iadd(ptr, len);
            // 16-byte scratch for shims that return an extra value.
            let slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                3,
            ));
            let sc = b.ins().stack_addr(types::I64, slot, 0);

            let mut e = Emit {
                module: &mut self.module,
                shims: &self.shims,
                b: &mut b,
            };
            let p = Pctx { end, sc };
            e.parse(ty, dst, ptr, &p);

            b.ins().return_(&[]);
            b.seal_all_blocks();
            b.finalize();
        }

        self.module.define_function(fid, &mut ctx).unwrap();
        self.module.clear_context(&mut ctx);
        self.module.finalize_definitions().unwrap();
        let code = self.module.get_finalized_function(fid);
        unsafe { std::mem::transmute::<*const u8, Parser>(code) }
    }
}

/// Constants available to every node of the parser emit: the end-of-input
/// pointer and the address of the 16-byte shim scratch area.
#[derive(Clone, Copy)]
struct Pctx {
    end: Value,
    sc: Value,
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

    // --- parser JIT: raw bytes -> struct, no Json tree ------------------

    fn load_u8(&mut self, addr: Value) -> Value {
        let v = self.b.ins().load(types::I8, MemFlags::trusted(), addr, 0);
        self.b.ins().uextend(types::I64, v)
    }
    fn scratch(&mut self, sc: Value, off: i32) -> Value {
        self.b.ins().load(types::I64, MemFlags::trusted(), sc, off)
    }
    fn byte_eq(&mut self, b: Value, ch: u8) -> Value {
        self.b.ins().icmp_imm(IntCC::Equal, b, ch as i64)
    }

    /// Inline whitespace skip — no shim call. Emits the bounded loop
    /// `while cur < end && is_ws(*cur) { cur += 1 }` directly in IR.
    fn ws(&mut self, cur: Value, p: &Pctx) -> Value {
        let head = self.b.create_block();
        self.b.append_block_param(head, types::I64);
        let chk = self.b.create_block();
        self.b.append_block_param(chk, types::I64);
        let cont = self.b.create_block();
        self.b.append_block_param(cont, types::I64);
        self.b.ins().jump(head, &[cur.into()]);

        self.b.switch_to_block(head);
        let c = self.b.block_params(head)[0];
        let inb = self.b.ins().icmp(IntCC::UnsignedLessThan, c, p.end);
        self.b.ins().brif(inb, chk, &[c.into()], cont, &[c.into()]);

        self.b.switch_to_block(chk);
        let c = self.b.block_params(chk)[0];
        let b = self.load_u8(c);
        let e1 = self.byte_eq(b, b' ');
        let e2 = self.byte_eq(b, b'\n');
        let e3 = self.byte_eq(b, b'\t');
        let e4 = self.byte_eq(b, b'\r');
        let o12 = self.b.ins().bor(e1, e2);
        let o34 = self.b.ins().bor(e3, e4);
        let is_ws = self.b.ins().bor(o12, o34);
        let c1 = self.b.ins().iadd_imm(c, 1);
        self.b
            .ins()
            .brif(is_ws, head, &[c1.into()], cont, &[c.into()]);

        self.b.switch_to_block(cont);
        self.b.block_params(cont)[0]
    }

    /// Inline string scan. Returns `(start_ptr, len, cursor_after_quote)`.
    /// Skips leading ws; handles `\"` so we don't stop early (escapes are
    /// not unescaped — naive fast path, same as before).
    fn strspan(&mut self, cur: Value, p: &Pctx) -> (Value, Value, Value) {
        let cur = self.ws(cur, p);
        let start = self.b.ins().iadd_imm(cur, 1); // past opening quote

        let head = self.b.create_block();
        self.b.append_block_param(head, types::I64);
        let chk = self.b.create_block();
        self.b.append_block_param(chk, types::I64);
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.b.ins().jump(head, &[start.into()]);

        self.b.switch_to_block(head);
        let q = self.b.block_params(head)[0];
        let inb = self.b.ins().icmp(IntCC::UnsignedLessThan, q, p.end);
        self.b.ins().brif(inb, chk, &[q.into()], done, &[q.into()]);

        self.b.switch_to_block(chk);
        let q = self.b.block_params(chk)[0];
        let b = self.load_u8(q);
        let is_quote = self.byte_eq(b, b'"');
        let is_bs = self.byte_eq(b, b'\\');
        let q1 = self.b.ins().iadd_imm(q, 1);
        let q2 = self.b.ins().iadd_imm(q, 2);
        let qn = self.b.ins().select(is_bs, q2, q1);
        self.b
            .ins()
            .brif(is_quote, done, &[q.into()], head, &[qn.into()]);

        self.b.switch_to_block(done);
        let qend = self.b.block_params(done)[0];
        let len = self.b.ins().isub(qend, start);
        let after = self.b.ins().iadd_imm(qend, 1);
        (start, len, after)
    }

    /// Inline integer scan. Returns `(value, cursor_after)`.
    fn intval(&mut self, cur: Value, p: &Pctx, signed: bool) -> (Value, Value) {
        let cur = self.ws(cur, p);
        // optional sign
        let b0 = self.load_u8(cur);
        let neg = if signed {
            self.byte_eq(b0, b'-')
        } else {
            self.b.ins().iconst(types::I8, 0)
        };
        let is_plus = self.byte_eq(b0, b'+');
        let is_sign = self.b.ins().bor(neg, is_plus);
        let cur1 = self.b.ins().iadd_imm(cur, 1);
        let cur = self.b.ins().select(is_sign, cur1, cur);

        let head = self.b.create_block();
        self.b.append_block_param(head, types::I64); // cursor
        self.b.append_block_param(head, types::I64); // acc
        let chk = self.b.create_block();
        self.b.append_block_param(chk, types::I64);
        self.b.append_block_param(chk, types::I64);
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.b.append_block_param(done, types::I64);
        let zero = self.iconst(0);
        self.b.ins().jump(head, &[cur.into(), zero.into()]);

        self.b.switch_to_block(head);
        let c = self.b.block_params(head)[0];
        let v = self.b.block_params(head)[1];
        let inb = self.b.ins().icmp(IntCC::UnsignedLessThan, c, p.end);
        self.b
            .ins()
            .brif(inb, chk, &[c.into(), v.into()], done, &[c.into(), v.into()]);

        self.b.switch_to_block(chk);
        let c = self.b.block_params(chk)[0];
        let v = self.b.block_params(chk)[1];
        let b = self.load_u8(c);
        let d = self.b.ins().iadd_imm(b, -(b'0' as i64));
        let is_dig = self.b.ins().icmp_imm(IntCC::UnsignedLessThan, d, 10);
        let v10 = self.b.ins().imul_imm(v, 10);
        let vn = self.b.ins().iadd(v10, d);
        let c1 = self.b.ins().iadd_imm(c, 1);
        self.b.ins().brif(
            is_dig,
            head,
            &[c1.into(), vn.into()],
            done,
            &[c.into(), v.into()],
        );

        self.b.switch_to_block(done);
        let cend = self.b.block_params(done)[0];
        let val = self.b.block_params(done)[1];
        let val = if signed {
            let negd = self.b.ins().ineg(val);
            self.b.ins().select(neg, negd, val)
        } else {
            val
        };
        (val, cend)
    }

    /// Inline, fully-unrolled comparison of the scanned key bytes against a
    /// compile-time-constant field name. No shim, no loop.
    fn keyeq(&mut self, kp: Value, kl: Value, name: &[u8]) -> Value {
        let mut acc =
            self.b.ins().icmp_imm(IntCC::Equal, kl, name.len() as i64);
        for (i, &ch) in name.iter().enumerate() {
            let byte =
                self.b
                    .ins()
                    .load(types::I8, MemFlags::trusted(), kp, i as i32);
            let byte = self.b.ins().uextend(types::I64, byte);
            let eqi = self.byte_eq(byte, ch);
            acc = self.b.ins().band(acc, eqi);
        }
        acc
    }

    /// Emit code to parse one JSON value at `cur` into `dst`; returns the
    /// cursor just past it.
    fn parse(&mut self, ty: &Ty, dst: Value, cur: Value, p: &Pctx) -> Value {
        match ty {
            Ty::Bool => {
                let c = self.ws(cur, p);
                let b = self.load_u8(c);
                let is_t = self.byte_eq(b, b't');
                let is_f = self.byte_eq(b, b'f');
                let v = self.b.ins().ireduce(types::I8, is_t);
                self.b.ins().store(MemFlags::trusted(), v, dst, 0);
                // "true"=4, "false"=5, "null"=4
                let five = self.iconst(5);
                let four = self.iconst(4);
                let adv = self.b.ins().select(is_f, five, four);
                self.b.ins().iadd(c, adv)
            }
            Ty::U(n) | Ty::I(n) => {
                let signed = matches!(ty, Ty::I(_));
                let (v, c) = self.intval(cur, p, signed);
                self.store_sized(dst, v, *n);
                c
            }
            Ty::F32 | Ty::F64 => {
                let c = self.call3(self.shims.f64v, cur, p.end, p.sc);
                let bits = self.scratch(p.sc, 0);
                let f = self.b.ins().bitcast(
                    types::F64,
                    MemFlags::new(),
                    bits,
                );
                if matches!(ty, Ty::F32) {
                    let f = self.b.ins().fdemote(types::F32, f);
                    self.b.ins().store(MemFlags::trusted(), f, dst, 0);
                } else {
                    self.b.ins().store(MemFlags::trusted(), f, dst, 0);
                }
                c
            }
            Ty::Char => {
                // naive: a one-char string
                let (sp, _sl, c) = self.strspan(cur, p);
                let ch = self.load_u8(sp);
                let ch = self.b.ins().ireduce(types::I32, ch);
                self.b.ins().store(MemFlags::trusted(), ch, dst, 0);
                c
            }
            Ty::Str(seq) => {
                let (sp, sl, c) = self.strspan(cur, p);
                let buf = self.call2(self.shims.dup, sp, sl);
                self.store_at(dst, seq.ptr_off, buf);
                self.store_at(dst, seq.cap_off, sl);
                self.store_at(dst, seq.len_off, sl);
                c
            }
            Ty::StrRef { ptr_off, len_off } => {
                let (sp, sl, c) = self.strspan(cur, p);
                let buf = self.call2(self.shims.dup, sp, sl);
                self.store_at(dst, *ptr_off, buf);
                self.store_at(dst, *len_off, sl);
                c
            }
            Ty::Struct { fields, .. } => self.parse_struct(fields, dst, cur, p),
            Ty::Vec {
                elem,
                elem_size,
                elem_align,
                seq,
            } => self.parse_vec(elem, *elem_size, *elem_align, seq, dst, cur, p),
            Ty::Unknown(w) => panic!("jit parser: unknown type `{w}`"),
        }
    }

    fn store_sized(&mut self, dst: Value, v: Value, n: u8) {
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
        self.b.ins().store(MemFlags::trusted(), v, dst, 0);
    }

    fn parse_struct(
        &mut self,
        fields: &[crate::plan::FieldTy],
        dst: Value,
        cur: Value,
        p: &Pctx,
    ) -> Value {
        // skip ws, step past '{'
        let cur = self.ws(cur, p);
        let cur = self.b.ins().iadd_imm(cur, 1);

        let header = self.b.create_block();
        self.b.append_block_param(header, types::I64);
        let cont = self.b.create_block();
        self.b.append_block_param(cont, types::I64);
        self.b.ins().jump(header, &[cur.into()]);

        self.b.switch_to_block(header);
        let hc = self.b.block_params(header)[0];
        let hc = self.ws(hc, p);
        let bch = self.load_u8(hc);
        let is_end = self.b.ins().icmp_imm(IntCC::Equal, bch, b'}' as i64);
        let body = self.b.create_block();
        self.b.append_block_param(body, types::I64);
        let hc_past = self.b.ins().iadd_imm(hc, 1);
        self.b
            .ins()
            .brif(is_end, cont, &[hc_past.into()], body, &[hc.into()]);

        self.b.switch_to_block(body);
        let bc = self.b.block_params(body)[0];
        // key span (inline)
        let (kp, kl, mut c) = self.strspan(bc, p);
        // skip ws, past ':'
        c = self.ws(c, p);
        c = self.b.ins().iadd_imm(c, 1);

        let after = self.b.create_block();
        self.b.append_block_param(after, types::I64);

        for f in fields {
            let cond = self.keyeq(kp, kl, f.name.as_bytes());
            let then_b = self.b.create_block();
            let else_b = self.b.create_block();
            self.b.ins().brif(cond, then_b, &[], else_b, &[]);

            self.b.switch_to_block(then_b);
            let off = self.iconst(f.offset as i64);
            let fdst = self.b.ins().iadd(dst, off);
            let cc = self.parse(&f.ty, fdst, c, p);
            self.b.ins().jump(after, &[cc.into()]);

            self.b.switch_to_block(else_b);
        }
        // no field matched: skip the value
        let skipped = self.call2(self.shims.skip, c, p.end);
        self.b.ins().jump(after, &[skipped.into()]);

        self.b.switch_to_block(after);
        let mut ac = self.b.block_params(after)[0];
        ac = self.ws(ac, p);
        let bcm = self.load_u8(ac);
        let is_comma = self.b.ins().icmp_imm(IntCC::Equal, bcm, b',' as i64);
        let ac1 = self.b.ins().iadd_imm(ac, 1);
        let ac = self.b.ins().select(is_comma, ac1, ac);
        self.b.ins().jump(header, &[ac.into()]);

        self.b.switch_to_block(cont);
        self.b.block_params(cont)[0]
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_vec(
        &mut self,
        elem: &Ty,
        elem_size: u64,
        elem_align: u64,
        seq: &SeqLayout,
        dst: Value,
        cur: Value,
        p: &Pctx,
    ) -> Value {
        let esz = self.iconst(elem_size as i64);
        let cur = self.ws(cur, p);
        let c0 = self.b.ins().iadd_imm(cur, 1); // past '['

        // count pass
        let ch = self.b.create_block();
        self.b.append_block_param(ch, types::I64); // cur
        self.b.append_block_param(ch, types::I64); // n
        let cdone = self.b.create_block();
        self.b.append_block_param(cdone, types::I64); // cur
        self.b.append_block_param(cdone, types::I64); // n
        let zero = self.iconst(0);
        self.b.ins().jump(ch, &[c0.into(), zero.into()]);

        self.b.switch_to_block(ch);
        let cc = self.b.block_params(ch)[0];
        let cn = self.b.block_params(ch)[1];
        let cc = self.ws(cc, p);
        let bch = self.load_u8(cc);
        let is_end = self.b.ins().icmp_imm(IntCC::Equal, bch, b']' as i64);
        let cbody = self.b.create_block();
        self.b.append_block_param(cbody, types::I64);
        self.b.append_block_param(cbody, types::I64);
        self.b
            .ins()
            .brif(is_end, cdone, &[cc.into(), cn.into()], cbody, &[cc.into(), cn.into()]);

        self.b.switch_to_block(cbody);
        let mut bc = self.b.block_params(cbody)[0];
        let bn = self.b.block_params(cbody)[1];
        bc = self.call2(self.shims.skip, bc, p.end);
        bc = self.ws(bc, p);
        let bcm = self.load_u8(bc);
        let is_comma = self.b.ins().icmp_imm(IntCC::Equal, bcm, b',' as i64);
        let bc1 = self.b.ins().iadd_imm(bc, 1);
        let bc = self.b.ins().select(is_comma, bc1, bc);
        let bn1 = self.b.ins().iadd_imm(bn, 1);
        self.b.ins().jump(ch, &[bc.into(), bn1.into()]);

        self.b.switch_to_block(cdone);
        let n = self.b.block_params(cdone)[1];
        let total = self.b.ins().imul(n, esz);
        let al = self.iconst(elem_align.max(1) as i64);
        let base = self.call2(self.shims.alloc, total, al);

        // parse pass
        let ph = self.b.create_block();
        self.b.append_block_param(ph, types::I64); // cur
        self.b.append_block_param(ph, types::I64); // i
        let pdone = self.b.create_block();
        self.b.append_block_param(pdone, types::I64); // cur
        let zero2 = self.iconst(0);
        self.b.ins().jump(ph, &[c0.into(), zero2.into()]);

        self.b.switch_to_block(ph);
        let pc = self.b.block_params(ph)[0];
        let pi = self.b.block_params(ph)[1];
        let ge = self.b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, pi, n);
        let pbody = self.b.create_block();
        self.b.append_block_param(pbody, types::I64);
        self.b.append_block_param(pbody, types::I64);
        self.b
            .ins()
            .brif(ge, pdone, &[pc.into()], pbody, &[pc.into(), pi.into()]);

        self.b.switch_to_block(pbody);
        let mut ec = self.b.block_params(pbody)[0];
        let ei = self.b.block_params(pbody)[1];
        ec = self.ws(ec, p);
        let eoff = self.b.ins().imul(ei, esz);
        let edst = self.b.ins().iadd(base, eoff);
        ec = self.parse(elem, edst, ec, p);
        ec = self.ws(ec, p);
        let ecm = self.load_u8(ec);
        let isc = self.b.ins().icmp_imm(IntCC::Equal, ecm, b',' as i64);
        let ec1 = self.b.ins().iadd_imm(ec, 1);
        let ec = self.b.ins().select(isc, ec1, ec);
        let ei1 = self.b.ins().iadd_imm(ei, 1);
        self.b.ins().jump(ph, &[ec.into(), ei1.into()]);

        self.b.switch_to_block(pdone);
        let mut fc = self.b.block_params(pdone)[0];
        fc = self.ws(fc, p);
        let fc = self.b.ins().iadd_imm(fc, 1); // past ']'
        self.store_at(dst, seq.ptr_off, base);
        self.store_at(dst, seq.cap_off, n);
        self.store_at(dst, seq.len_off, n);
        fc
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

// --- parser-JIT shims (byte scanning) ------------------------------------

#[inline]
fn isws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\t' | b'\r')
}

#[inline]
unsafe fn slice<'a>(cur: *const u8, end: *const u8) -> &'a [u8] {
    let len = (end as usize).saturating_sub(cur as usize);
    unsafe { std::slice::from_raw_parts(cur, len) }
}

unsafe extern "C" fn rt_ws(mut cur: *const u8, end: *const u8) -> *const u8 {
    while cur < end && isws(unsafe { *cur }) {
        cur = unsafe { cur.add(1) };
    }
    cur
}

/// At a string: write (ptr,len) of the bytes between the quotes into `sc`,
/// return the cursor past the closing quote. Escapes are not unescaped
/// (naive fast path); `\"` is still handled so we don't stop early.
/// Skip one JSON value; return the cursor just past it.
unsafe extern "C" fn rt_skip(cur: *const u8, end: *const u8) -> *const u8 {
    let mut cur = unsafe { rt_ws(cur, end) };
    if cur >= end {
        return cur;
    }
    match unsafe { *cur } {
        b'"' => {
            cur = unsafe { cur.add(1) };
            while cur < end && unsafe { *cur } != b'"' {
                cur = if unsafe { *cur } == b'\\' {
                    unsafe { cur.add(2) }
                } else {
                    unsafe { cur.add(1) }
                };
            }
            unsafe { cur.add(1) }
        }
        b'{' | b'[' => {
            let mut depth = 0i32;
            let mut in_str = false;
            while cur < end {
                let c = unsafe { *cur };
                if in_str {
                    if c == b'\\' {
                        cur = unsafe { cur.add(2) };
                        continue;
                    }
                    if c == b'"' {
                        in_str = false;
                    }
                    cur = unsafe { cur.add(1) };
                    continue;
                }
                match c {
                    b'"' => in_str = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return unsafe { cur.add(1) };
                        }
                    }
                    _ => {}
                }
                cur = unsafe { cur.add(1) };
            }
            cur
        }
        _ => {
            while cur < end {
                let c = unsafe { *cur };
                if c == b',' || c == b'}' || c == b']' || isws(c) {
                    break;
                }
                cur = unsafe { cur.add(1) };
            }
            cur
        }
    }
}

unsafe extern "C" fn rt_f64v(
    cur: *const u8,
    end: *const u8,
    sc: *mut u64,
) -> *const u8 {
    let cur = unsafe { rt_ws(cur, end) };
    let mut p = cur;
    while p < end {
        let c = unsafe { *p };
        if c.is_ascii_digit()
            || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E')
        {
            p = unsafe { p.add(1) };
        } else {
            break;
        }
    }
    let bytes = unsafe { slice(cur, p) };
    let f: f64 = std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    unsafe { *sc = f.to_bits() };
    p
}
