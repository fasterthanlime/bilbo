//! The cranelift backend. Given a [`Ty`] recovered from DWARF, compile a
//! function `extern "C" fn(*mut u8, *const u8, usize)` specialized to that
//! type: it walks the raw JSON bytes straight into the struct. Every offset,
//! size and field-name is baked in as a constant; whitespace/string/number/
//! key scanning is emitted *inline* as machine code. The only shims are the
//! ones that genuinely need Rust: the allocator (`rt_dup`, `rt_realloc`),
//! float parsing (`rt_f64v`), and skipping an unknown object value
//! (`rt_skip`).

use std::cell::RefCell;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value, types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};

use crate::plan::{SeqLayout, Ty};

/// `extern "C" fn(dst: *mut u8, input: *const u8, len: usize)` — parses raw
/// JSON bytes straight into the struct, no `Json` tree at all.
pub type Parser = unsafe extern "C" fn(*mut u8, *const u8, usize);

thread_local! {
    static JIT: RefCell<Jit> = RefCell::new(Jit::new());
}

/// Compile a specialized parser for `ty`. Caching by type lives in
/// `resolve` (one `Resolved` per type holds the result); this just emits
/// code, and the thread-local `JITModule` keeps it mapped.
pub fn compile_parser(ty: &Ty) -> Parser {
    JIT.with(|j| j.borrow_mut().compile_parser(ty))
}

struct Jit {
    module: JITModule,
    shims: Shims,
    seq: u32,
}

/// `FuncId`s of the runtime shims, declared once in the module. Byte
/// scanning is emitted inline in cranelift; only these remain.
#[derive(Clone, Copy)]
struct Shims {
    dup: FuncId,     // alloc + memcpy a string's bytes
    realloc: FuncId, // grow a Vec buffer
    skip: FuncId,    // skip an unknown object value
    f64v: FuncId,    // parse a float
}

impl Jit {
    fn new() -> Self {
        let mut b =
            JITBuilder::new(default_libcall_names()).expect("jit builder");
        b.symbol("rt_dup", rt_dup as *const u8);
        b.symbol("rt_realloc", rt_realloc as *const u8);
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
            dup: decl(&mut module, "rt_dup", &[p, p], Some(p)),
            realloc: decl(&mut module, "rt_realloc", &[p, p, p, p], Some(p)),
            skip: decl(&mut module, "rt_skip", &[p, p], Some(p)),
            f64v: decl(&mut module, "rt_f64v", &[p, p, p], Some(p)),
        };

        Jit {
            module,
            shims,
            seq: 0,
        }
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
        let size = ctx
            .compiled_code()
            .map(|c| c.code_buffer().len())
            .unwrap_or(0);
        self.module.clear_context(&mut ctx);
        self.module.finalize_definitions().unwrap();
        let code = self.module.get_finalized_function(fid);

        // Tell profilers what this anonymous JIT memory is. We dump the
        // *finalized* (relocated) bytes so disassembly shows real targets.
        if size > 0 {
            let bytes = unsafe { std::slice::from_raw_parts(code, size) };
            crate::jitdump::register(
                &format!("dwarf_json::jit::{name}"),
                code as u64,
                bytes,
            );
        }
        unsafe { std::mem::transmute::<*const u8, Parser>(code) }
    }
}

/// Constants available to every node of the emit: the end-of-input pointer
/// and the address of the 16-byte shim scratch area.
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
    fn call4(
        &mut self,
        f: FuncId,
        a: Value,
        b: Value,
        c: Value,
        d: Value,
    ) -> Value {
        let r = self.module.declare_func_in_func(f, self.b.func);
        let ci = self.b.ins().call(r, &[a, b, c, d]);
        self.b.inst_results(ci)[0]
    }
    fn iconst(&mut self, v: i64) -> Value {
        self.b.ins().iconst(types::I64, v)
    }
    fn load_u8(&mut self, addr: Value) -> Value {
        // Single extending load: `ldrb` already zero-extends, so this is
        // one instruction (vs `load I8` + `uextend`, which adds a `uxtb`
        // in every hot scan loop).
        self.b.ins().uload8(types::I64, MemFlags::trusted(), addr, 0)
    }
    fn load_u8_off(&mut self, addr: Value, off: i32) -> Value {
        self.b.ins().uload8(types::I64, MemFlags::trusted(), addr, off)
    }
    fn scratch(&mut self, sc: Value, off: i32) -> Value {
        self.b.ins().load(types::I64, MemFlags::trusted(), sc, off)
    }
    fn byte_eq(&mut self, b: Value, ch: u8) -> Value {
        self.b.ins().icmp_imm(IntCC::Equal, b, ch as i64)
    }
    fn store_at(&mut self, base: Value, off: usize, v: Value) {
        self.b.ins().store(MemFlags::trusted(), v, base, off as i32);
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

    /// Inline whitespace skip — `while cur < end && is_ws(*cur) { cur += 1 }`.
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
        // JSON ws is 0x20, and 0x09..=0x0d. `(b - 9) <= 4` (unsigned)
        // covers \t\n\v\f\r; \v\f can't legally appear between tokens, so
        // skipping them too is harmless leniency and halves the test.
        let is_space = self.byte_eq(b, b' ');
        let d = self.b.ins().iadd_imm(b, -9);
        let in_ctrl = self.b.ins().icmp_imm(IntCC::UnsignedLessThanOrEqual, d, 4);
        let is_ws = self.b.ins().bor(is_space, in_ctrl);
        let c1 = self.b.ins().iadd_imm(c, 1);
        self.b
            .ins()
            .brif(is_ws, head, &[c1.into()], cont, &[c.into()]);

        self.b.switch_to_block(cont);
        self.b.block_params(cont)[0]
    }

    /// Inline string scan. Returns `(start_ptr, len, cursor_after_quote)`.
    /// Skips leading ws; handles `\"` so we don't stop early (escapes are
    /// not unescaped — naive fast path).
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
    fn intval(
        &mut self,
        cur: Value,
        p: &Pctx,
        signed: bool,
    ) -> (Value, Value) {
        let cur = self.ws(cur, p);
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
        self.b.ins().brif(
            inb,
            chk,
            &[c.into(), v.into()],
            done,
            &[c.into(), v.into()],
        );

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
            let byte = self.load_u8_off(kp, i as i32);
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
                let f = self.b.ins().bitcast(types::F64, MemFlags::new(), bits);
                if matches!(ty, Ty::F32) {
                    let f = self.b.ins().fdemote(types::F32, f);
                    self.b.ins().store(MemFlags::trusted(), f, dst, 0);
                } else {
                    self.b.ins().store(MemFlags::trusted(), f, dst, 0);
                }
                c
            }
            Ty::Char => {
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
            Ty::Struct { fields, .. } => {
                self.parse_struct(fields, dst, cur, p)
            }
            Ty::Vec {
                elem,
                elem_size,
                elem_align,
                seq,
            } => self.parse_vec(elem, *elem_size, *elem_align, seq, dst, cur, p),
            Ty::Unknown(w) => panic!("jit parser: unknown type `{w}`"),
        }
    }

    fn parse_struct(
        &mut self,
        fields: &[crate::plan::FieldTy],
        dst: Value,
        cur: Value,
        p: &Pctx,
    ) -> Value {
        let cur = self.ws(cur, p);
        let cur = self.b.ins().iadd_imm(cur, 1); // past '{'

        let header = self.b.create_block();
        self.b.append_block_param(header, types::I64);
        let cont = self.b.create_block();
        self.b.append_block_param(cont, types::I64);
        self.b.ins().jump(header, &[cur.into()]);

        self.b.switch_to_block(header);
        let hc = self.b.block_params(header)[0];
        let hc = self.ws(hc, p);
        let bch = self.load_u8(hc);
        let is_end = self.byte_eq(bch, b'}');
        let body = self.b.create_block();
        self.b.append_block_param(body, types::I64);
        let hc_past = self.b.ins().iadd_imm(hc, 1);
        self.b
            .ins()
            .brif(is_end, cont, &[hc_past.into()], body, &[hc.into()]);

        self.b.switch_to_block(body);
        let bc = self.b.block_params(body)[0];
        let (kp, kl, mut c) = self.strspan(bc, p); // key
        c = self.ws(c, p);
        c = self.b.ins().iadd_imm(c, 1); // past ':'

        let after = self.b.create_block();
        self.b.append_block_param(after, types::I64);

        // One then-block per field, reachable from both the fast and slow
        // dispatch chains; and one skip block for an unknown key.
        let then_blocks: Vec<_> =
            fields.iter().map(|_| self.b.create_block()).collect();
        let skip_b = self.b.create_block();

        // Fast path: if the key has >=8 bytes of slack before EOF, load it
        // as a single word and compare against the constant field names
        // (serde-style) instead of re-reading it byte-by-byte per field.
        let kp8 = self.b.ins().iadd_imm(kp, 8);
        let can_word =
            self.b.ins().icmp(IntCC::UnsignedLessThanOrEqual, kp8, p.end);
        let fast_b = self.b.create_block();
        let slow_b = self.b.create_block();
        self.b.ins().brif(can_word, fast_b, &[], slow_b, &[]);

        self.b.switch_to_block(fast_b);
        let w = self.b.ins().load(types::I64, MemFlags::trusted(), kp, 0);
        for (f, &then_b) in fields.iter().zip(&then_blocks) {
            let name = f.name.as_bytes();
            let cond = if name.len() <= 8 {
                let mut wb = [0u8; 8];
                wb[..name.len()].copy_from_slice(name);
                let word = u64::from_le_bytes(wb) as i64;
                let mask = if name.len() == 8 {
                    -1i64
                } else {
                    ((1u64 << (8 * name.len())) - 1) as i64
                };
                let lc =
                    self.b.ins().icmp_imm(IntCC::Equal, kl, name.len() as i64);
                let wm = self.b.ins().band_imm(w, mask);
                let we = self.b.ins().icmp_imm(IntCC::Equal, wm, word);
                self.b.ins().band(lc, we)
            } else {
                self.keyeq(kp, kl, name)
            };
            let next = self.b.create_block();
            self.b.ins().brif(cond, then_b, &[], next, &[]);
            self.b.switch_to_block(next);
        }
        self.b.ins().jump(skip_b, &[]);

        // Slow path (key within 8 bytes of EOF): byte-wise compare.
        self.b.switch_to_block(slow_b);
        for (f, &then_b) in fields.iter().zip(&then_blocks) {
            let cond = self.keyeq(kp, kl, f.name.as_bytes());
            let next = self.b.create_block();
            self.b.ins().brif(cond, then_b, &[], next, &[]);
            self.b.switch_to_block(next);
        }
        self.b.ins().jump(skip_b, &[]);

        for (f, &then_b) in fields.iter().zip(&then_blocks) {
            self.b.switch_to_block(then_b);
            let off = self.iconst(f.offset as i64);
            let fdst = self.b.ins().iadd(dst, off);
            let cc = self.parse(&f.ty, fdst, c, p);
            self.b.ins().jump(after, &[cc.into()]);
        }

        // no field matched: skip the value (rare; never hit for a known
        // schema, but keeps us correct on extra keys like serde does).
        self.b.switch_to_block(skip_b);
        let skipped = self.call2(self.shims.skip, c, p.end);
        self.b.ins().jump(after, &[skipped.into()]);

        self.b.switch_to_block(after);
        let ac = self.b.block_params(after)[0];
        let ac = self.ws(ac, p);
        let bcm = self.load_u8(ac);
        let is_comma = self.byte_eq(bcm, b',');
        let ac1 = self.b.ins().iadd_imm(ac, 1);
        let ac = self.b.ins().select(is_comma, ac1, ac);
        self.b.ins().jump(header, &[ac.into()]);

        self.b.switch_to_block(cont);
        self.b.block_params(cont)[0]
    }

    /// Single pass. Grow a buffer geometrically (like `Vec`), parsing each
    /// element straight into place. No count pass, no `rt_skip`.
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
        let align = self.iconst(elem_align.max(1) as i64);
        let cur = self.ws(cur, p);
        let c0 = self.b.ins().iadd_imm(cur, 1); // past '['

        // loop state: (cursor, base, cap, len). Empty Vec => base is the
        // aligned dangling pointer and cap 0, so Drop won't free it.
        let zero = self.iconst(0);
        let head = self.b.create_block();
        for _ in 0..4 {
            self.b.append_block_param(head, types::I64);
        }
        let body = self.b.create_block();
        for _ in 0..4 {
            self.b.append_block_param(body, types::I64);
        }
        let grow = self.b.create_block();
        for _ in 0..4 {
            self.b.append_block_param(grow, types::I64);
        }
        let afterg = self.b.create_block();
        for _ in 0..4 {
            self.b.append_block_param(afterg, types::I64);
        }
        let done = self.b.create_block();
        for _ in 0..4 {
            self.b.append_block_param(done, types::I64);
        }
        self.b
            .ins()
            .jump(head, &[c0.into(), align.into(), zero.into(), zero.into()]);

        // head: at ']' ? -> done : body
        self.b.switch_to_block(head);
        let hc = self.b.block_params(head)[0];
        let hbase = self.b.block_params(head)[1];
        let hcap = self.b.block_params(head)[2];
        let hlen = self.b.block_params(head)[3];
        let hc = self.ws(hc, p);
        let bch = self.load_u8(hc);
        let is_end = self.byte_eq(bch, b']');
        self.b.ins().brif(
            is_end,
            done,
            &[hc.into(), hbase.into(), hcap.into(), hlen.into()],
            body,
            &[hc.into(), hbase.into(), hcap.into(), hlen.into()],
        );

        // body: ensure capacity, then either grow or go straight to afterg
        self.b.switch_to_block(body);
        let bc = self.b.block_params(body)[0];
        let bbase = self.b.block_params(body)[1];
        let bcap = self.b.block_params(body)[2];
        let blen = self.b.block_params(body)[3];
        let full = self.b.ins().icmp(IntCC::Equal, blen, bcap);
        self.b.ins().brif(
            full,
            grow,
            &[bc.into(), bbase.into(), bcap.into(), blen.into()],
            afterg,
            &[bc.into(), bbase.into(), bcap.into(), blen.into()],
        );

        // grow: newcap = cap==0 ? 4 : cap*2; realloc
        self.b.switch_to_block(grow);
        let gc = self.b.block_params(grow)[0];
        let gbase = self.b.block_params(grow)[1];
        let gcap = self.b.block_params(grow)[2];
        let glen = self.b.block_params(grow)[3];
        let is0 = self.b.ins().icmp_imm(IntCC::Equal, gcap, 0);
        let cap2 = self.b.ins().imul_imm(gcap, 2);
        let four = self.iconst(4);
        let newcap = self.b.ins().select(is0, four, cap2);
        let oldbytes = self.b.ins().imul(gcap, esz);
        let newbytes = self.b.ins().imul(newcap, esz);
        let nbase =
            self.call4(self.shims.realloc, gbase, oldbytes, newbytes, align);
        self.b.ins().jump(
            afterg,
            &[gc.into(), nbase.into(), newcap.into(), glen.into()],
        );

        // afterg: parse element at base + len*esz, then ',' and loop
        self.b.switch_to_block(afterg);
        let ac = self.b.block_params(afterg)[0];
        let abase = self.b.block_params(afterg)[1];
        let acap = self.b.block_params(afterg)[2];
        let alen = self.b.block_params(afterg)[3];
        let aoff = self.b.ins().imul(alen, esz);
        let edst = self.b.ins().iadd(abase, aoff);
        let ac = self.parse(elem, edst, ac, p);
        let ac = self.ws(ac, p);
        let bcm = self.load_u8(ac);
        let is_comma = self.byte_eq(bcm, b',');
        let ac1 = self.b.ins().iadd_imm(ac, 1);
        let ac = self.b.ins().select(is_comma, ac1, ac);
        let alen1 = self.b.ins().iadd_imm(alen, 1);
        self.b.ins().jump(
            head,
            &[ac.into(), abase.into(), acap.into(), alen1.into()],
        );

        // done: write the three words, step past ']'
        self.b.switch_to_block(done);
        let fc = self.b.block_params(done)[0];
        let fbase = self.b.block_params(done)[1];
        let fcap = self.b.block_params(done)[2];
        let flen = self.b.block_params(done)[3];
        let fc = self.ws(fc, p);
        let fc = self.b.ins().iadd_imm(fc, 1); // past ']'
        self.store_at(dst, seq.ptr_off, fbase);
        self.store_at(dst, seq.cap_off, fcap);
        self.store_at(dst, seq.len_off, flen);
        fc
    }
}

// --- runtime shims -------------------------------------------------------

/// Allocate + copy `n` bytes (matches `Global`, so `String`'s `Drop` frees
/// it correctly with `cap == len`).
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

/// Grow a `Vec` buffer. `old`/`new` are byte sizes; matches `Global` so the
/// reconstructed `Vec`'s `Drop` frees `cap * size_of::<T>()` correctly.
unsafe extern "C" fn rt_realloc(
    ptr: *mut u8,
    old: usize,
    new: usize,
    align: usize,
) -> *mut u8 {
    let align = align.max(1);
    if old == 0 {
        if new == 0 {
            return std::ptr::without_provenance_mut(align);
        }
        let l = std::alloc::Layout::from_size_align(new, align).unwrap();
        let p = unsafe { std::alloc::alloc(l) };
        assert!(!p.is_null());
        return p;
    }
    let ol = std::alloc::Layout::from_size_align(old, align).unwrap();
    let p = unsafe { std::alloc::realloc(ptr, ol, new) };
    assert!(!p.is_null());
    p
}

#[inline]
fn isws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\t' | b'\r')
}

#[inline]
unsafe fn slice<'a>(cur: *const u8, end: *const u8) -> &'a [u8] {
    let len = (end as usize).saturating_sub(cur as usize);
    unsafe { std::slice::from_raw_parts(cur, len) }
}

unsafe fn rt_ws(mut cur: *const u8, end: *const u8) -> *const u8 {
    while cur < end && isws(unsafe { *cur }) {
        cur = unsafe { cur.add(1) };
    }
    cur
}

/// Skip one JSON value; return the cursor just past it. Only used for keys
/// that don't match any struct field (extra keys), like serde ignores.
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

/// Parse a float token; writes the bits into `sc`, returns cursor after.
unsafe extern "C" fn rt_f64v(
    cur: *const u8,
    end: *const u8,
    sc: *mut u64,
) -> *const u8 {
    let cur = unsafe { rt_ws(cur, end) };
    let mut q = cur;
    while q < end {
        let c = unsafe { *q };
        if c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E')
        {
            q = unsafe { q.add(1) };
        } else {
            break;
        }
    }
    let bytes = unsafe { slice(cur, q) };
    let f: f64 = std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    unsafe { *sc = f.to_bits() };
    q
}
