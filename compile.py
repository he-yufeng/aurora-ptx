#!/usr/bin/env python3
"""Aurora PTX: an independent, typed Kernel-IR to PTX visitor.

Usage: python compile.py path/to/kernel.ir.json > out.ptx

Emmy is used only for its mandated Graph.from_dict IR deserializer. No renderer,
CUDA compiler, reference source, expected outputs, or problem-name dispatch is
used. Launch metadata supplies the ABI, never the implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import struct
import sys

from emmy.compiler.graph import Graph


class UnsupportedIR(ValueError):
    pass


def static(dim):
    value = dim if isinstance(dim, int) else getattr(dim, "value", None)
    if not isinstance(value, int):
        raise UnsupportedIR(f"non-static dimension: {dim!r}")
    return value


def typ(dtype):
    name = getattr(dtype, "name", dtype)
    table = {"f32": "f32", "float": "f32", "f64": "f64", "double": "f64",
             "int": "s32", "i32": "s32", "int32_t": "s32", "i64": "s64", "int64_t": "s64",
             "f16": "f16", "half": "f16", "u32": "u32", "bool": "pred"}
    if name not in table:
        raise UnsupportedIR(f"unsupported scalar dtype {name!r}")
    return table[name]


@dataclass(frozen=True)
class Value:
    operand: str
    kind: str


class PTX:
    def __init__(self, graph, node, metadata):
        self.graph, self.node, self.kernel = graph, node, node.op
        self.metadata = metadata
        self.lines, self.shared_decls = [], []
        self.counts = {"f32": 0, "f64": 0, "s32": 0, "s64": 0,
                       "u32": 0, "u64": 0, "pred": 0, "b16": 0}
        self.env, self.buffers = {}, {}
        self.label_count = 0
        self.smem = {}
        for n in graph.nodes.values():
            for tensor in n.outputs:
                self.buffers[tensor.name] = {
                    "shape": tuple(static(d) for d in tensor.shape),
                    "dtype": typ(tensor.dtype), "space": "global"}
        for stmt in self.walk(self.kernel.body):
            if type(stmt).__name__ == "Smem":
                if stmt.name in self.smem and self.smem[stmt.name] != stmt:
                    raise UnsupportedIR("conflicting shared-memory declarations")
                self.smem[stmt.name] = stmt

    @staticmethod
    def walk(body):
        for stmt in body:
            yield stmt
            for attr in ("body", "else_body"):
                child = getattr(stmt, attr, None)
                if child:
                    yield from PTX.walk(child)

    def emit(self, line):
        self.lines.append("    " + line)

    def label(self, stem):
        self.label_count += 1
        return f"L_{stem}_{self.label_count}"

    def fresh(self, kind):
        prefix = {"f32": "f", "f64": "d", "s32": "r", "s64": "l",
                  "u32": "u", "u64": "rd", "pred": "p", "b16": "h"}[kind]
        n = self.counts[kind]
        self.counts[kind] += 1
        return Value(f"%{prefix}{n}", kind)

    def literal(self, value, kind):
        if kind == "f32":
            return Value("0f" + struct.pack(">f", float(value)).hex(), kind)
        if kind == "f64":
            return Value("0d" + struct.pack(">d", float(value)).hex(), kind)
        return Value(str(int(value)), kind)

    def convert(self, value, kind):
        if value.kind == kind:
            return value
        out = self.fresh(kind)
        rounding = ".rn" if kind.startswith("f") and value.kind.startswith(("s", "u")) else ""
        if kind.startswith(("s", "u")) and value.kind.startswith("f"):
            rounding = ".rzi"
        if kind.startswith("f") and value.kind.startswith("f"):
            rounding = ".rn" if kind == "f32" else ""
        self.emit(f"cvt{rounding}.{kind}.{value.kind} {out.operand}, {value.operand};")
        return out

    def reg(self, value):
        if value.operand.startswith("%"):
            return value
        out = self.fresh(value.kind)
        self.emit(f"mov.{value.kind} {out.operand}, {value.operand};")
        return out

    def bind(self, name, value):
        dst = self.env.get(name)
        if dst is None:
            dst = self.fresh(value.kind)
            self.env[name] = dst
        value = self.convert(value, dst.kind)
        if value.operand != dst.operand:
            self.emit(f"mov.{dst.kind} {dst.operand}, {value.operand};")
        return dst

    def binary(self, op, left, right):
        if op in ("&&", "and", "||", "or"):
            out = self.fresh("pred")
            self.emit(f"{'and' if op in ('&&','and') else 'or'}.pred {out.operand}, "
                      f"{self.predicate(left).operand}, {self.predicate(right).operand};")
            return out
        right = self.convert(right, left.kind)
        if op in ("==", "!=", "<", "<=", ">", ">="):
            out = self.fresh("pred")
            cmp = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[op]
            self.emit(f"setp.{cmp}.{left.kind} {out.operand}, {left.operand}, {right.operand};")
            return out
        out = self.fresh(left.kind)
        mnemonic = {"+": "add", "-": "sub", "*": "mul", "/": "div", "//": "div",
                    "%": "rem", "&": "and", "|": "or", "^": "xor", ">>": "shr", "<<": "shl"}[op]
        modifier = ".rn" if left.kind.startswith("f") and op in ("+", "-", "*", "/") else ""
        if op == "*" and not left.kind.startswith("f"):
            modifier = ".lo"
        instruction_kind = left.kind
        if op in ("&", "|", "^", "<<"):
            instruction_kind = "b64" if left.kind.endswith("64") else "b32"
        self.emit(f"{mnemonic}{modifier}.{instruction_kind} {out.operand}, {left.operand}, {right.operand};")
        return out

    def predicate(self, value):
        return value if value.kind == "pred" else self.binary("!=", value, self.literal(0, value.kind))

    def expr(self, expr):
        if isinstance(expr, str):
            if expr not in self.env:
                raise UnsupportedIR(f"unbound typed IR value: {expr}")
            return self.env[expr]
        if isinstance(expr, (int, float)):
            return self.literal(expr, "s32" if isinstance(expr, int) else "f32")
        name = type(expr).__name__
        if name == "Var":
            return self.expr(expr.name)
        if name == "Literal":
            return self.literal(expr.value, typ(expr.dtype))
        if name == "BinaryExpr":
            return self.binary(expr.op, self.expr(expr.left), self.expr(expr.right))
        if name == "TernaryExpr":
            cond = self.predicate(self.expr(expr.cond))
            a, b = self.expr(expr.if_true), self.expr(expr.if_false)
            b = self.convert(b, a.kind)
            dst = self.fresh(a.kind)
            self.emit(f"selp.{a.kind} {dst.operand}, {a.operand}, {b.operand}, {cond.operand};")
            return dst
        raise UnsupportedIR(f"unimplemented expression {name}")

    def arithmetic(self, name, args, kind="f32"):
        args = [self.convert(self.expr(x), kind) if not isinstance(x, Value)
                else self.convert(x, kind) for x in args]
        if name == "copy":
            return args[0]
        if name in ("add", "subtract", "multiply", "divide"):
            return self.binary({"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[name], *args)
        dst = self.fresh(kind)
        if name in ("maximum", "minimum"):
            self.emit(f"{'max' if name == 'maximum' else 'min'}.{kind} {dst.operand}, {args[0].operand}, {args[1].operand};")
        elif name in ("relu", "abs", "negative", "negate"):
            if name == "relu":
                self.emit(f"max.{kind} {dst.operand}, {args[0].operand}, {self.literal(0,kind).operand};")
            else:
                self.emit(f"{'abs' if name=='abs' else 'neg'}.{kind} {dst.operand}, {args[0].operand};")
        elif name in ("sqrt", "reciprocal", "rsqrt"):
            mnemonic = {"sqrt": "sqrt.rn", "reciprocal": "rcp.rn", "rsqrt": "rsqrt.approx"}[name]
            self.emit(f"{mnemonic}.{kind} {dst.operand}, {args[0].operand};")
        elif name in ("exp", "exp2", "log", "log2"):
            if kind != "f32":
                raise UnsupportedIR("transcendentals currently require f32")
            value = args[0]
            if name == "exp":
                value = self.binary("*", value, self.literal(math.log2(math.e), kind))
            self.emit(f"{'lg2' if name.startswith('log') else 'ex2'}.approx.f32 {dst.operand}, {value.operand};")
            if name == "log":
                return self.binary("*", dst, self.literal(math.log(2), kind))
        else:
            raise UnsupportedIR(f"unsupported elementwise operation {name}")
        return dst

    def flat(self, buffer, index):
        shape = self.buffers[buffer]["shape"]
        if len(index) != len(shape):
            raise UnsupportedIR(f"rank mismatch for {buffer}")
        off = self.literal(0, "s32")
        for i, dim in zip(index, shape):
            off = self.binary("+", self.binary("*", off, self.literal(dim, "s32")),
                              self.convert(self.expr(i), "s32"))
        return off

    def address(self, buffer, index=None, flat=None, extra=0, swizzle="NONE"):
        info = self.buffers[buffer]
        off = self.flat(buffer, index) if flat is None else flat
        if extra:
            off = self.binary("+", off, self.literal(extra, "s32"))
        if swizzle != "NONE":
            raise UnsupportedIR(f"unsupported shared swizzle: {swizzle}")
        size = {"f32": 4, "f64": 8, "s32": 4, "s64": 8, "u32": 4, "f16": 2}[info["dtype"]]
        if info["space"] == "shared":
            byteoff = self.convert(self.binary("*", off, self.literal(size, "s32")), "u32")
            dst = self.fresh("u32")
            self.emit(f"add.u32 {dst.operand}, {info['ptr'].operand}, {byteoff.operand};")
        else:
            byteoff, dst = self.fresh("s64"), self.fresh("u64")
            self.emit(f"mul.wide.s32 {byteoff.operand}, {off.operand}, {size};")
            self.emit(f"add.u64 {dst.operand}, {info['ptr'].operand}, {byteoff.operand};")
        return dst

    def load(self, buffer, index=None, flat=None, extra=0):
        info = self.buffers[buffer]
        addr = self.address(buffer, index, flat, extra)
        dtype = info["dtype"]
        dst = self.fresh("b16" if dtype == "f16" else dtype)
        self.emit(f"ld.{info['space']}.{'b16' if dtype=='f16' else dtype} {dst.operand}, [{addr.operand}];")
        if dtype == "f16":
            out = self.fresh("f32")
            self.emit(f"cvt.f32.f16 {out.operand}, {dst.operand};")
            return out
        return dst

    def store(self, buffer, index, value, extra=0, atomic=False, flat=None, swizzle="NONE"):
        info = self.buffers[buffer]
        addr = self.address(buffer, index, flat, extra, swizzle)
        dtype = info["dtype"]
        if dtype == "f16":
            out = self.fresh("b16")
            value = self.convert(value, "f32")
            self.emit(f"cvt.rn.f16.f32 {out.operand}, {value.operand};")
            value, dtype = out, "b16"
        else:
            value = self.convert(value, dtype)
        if atomic:
            if dtype != "f32" or info["space"] != "global":
                raise UnsupportedIR("only global f32 atomic addition is implemented")
            self.emit(f"red.global.add.f32 [{addr.operand}], {value.operand};")
        else:
            self.emit(f"st.{info['space']}.{dtype} [{addr.operand}], {value.operand};")

    def barrier(self, barrier_id=0, count=None, warp=False):
        if warp:
            self.emit("bar.warp.sync 0xffffffff;")
        elif count is None:
            self.emit(f"bar.sync {barrier_id};")
        else:
            self.emit(f"bar.sync {barrier_id}, {count};")

    def loop(self, stmt):
        axis = stmt.axis.name
        # A reduction is seeded where its own reduction axis enters, not at
        # every enclosing loop. This also handles coupled online-softmax state.
        if stmt.seed:
            seen = set()
            for acc in self.walk(stmt.body):
                if type(acc).__name__ == "Accum" and axis in acc.axes and acc.name not in seen:
                    self.bind(acc.name, self.literal(acc.op.identity, typ(acc.dtype)))
                    seen.add(acc.name)
        strided = type(stmt).__name__ == "StridedLoop"
        start = self.expr(stmt.start) if strided else self.literal(0,"s32")
        step = self.expr(stmt.step) if strided else self.literal(1,"s32")
        end = self.expr(stmt.end) if strided and stmt.end is not None else self.literal(static(stmt.axis.extent),"s32")
        it = self.bind(axis, self.convert(start,"s32"))
        head, tail = self.label("loop"), self.label("loop_end")
        self.lines.append(head + ":")
        done = self.binary(">=", it, self.convert(end,"s32"))
        self.emit(f"@{done.operand} bra {tail};")
        self.body(stmt.body)
        nxt = self.binary("+", it, self.convert(step,"s32"))
        self.bind(axis,nxt)
        self.emit(f"bra {head};")
        self.lines.append(tail + ":")

    def conditional(self, condition, body, else_body=()):
        p = self.predicate(condition)
        other, end = self.label("else"), self.label("if_end")
        self.emit(f"@!{p.operand} bra {other};")
        self.body(body)
        self.emit(f"bra {end};")
        self.lines.append(other + ":")
        self.body(else_body)
        self.lines.append(end + ":")

    def tile(self, stmt):
        if stmt.aux_threads:
            raise UnsupportedIR("auxiliary producer bands are not in this contest dialect")
        tid = self.env["threadIdx.x"]
        block = self.env["blockIdx.x"]
        width = self.literal(stmt.block_threads,"s32") if stmt.block_threads else self.env["blockDim.x"]
        gid = self.binary("+", self.binary("*", block,width),tid)
        extents = [static(a.extent) for a in stmt.axes]
        n = math.prod(extents)
        end = self.label("tile_end")
        invalid = self.binary(">=",gid,self.literal(n,"s32"))
        self.emit(f"@{invalid.operand} bra {end};")
        for i, axis in enumerate(stmt.axes):
            stride = math.prod(extents[i+1:])
            v = self.binary("/",gid,self.literal(stride,"s32"))
            v = self.binary("%",v,self.literal(extents[i],"s32"))
            self.bind(axis.name,v)
        if stmt.raster_group and stmt.raster_axes:
            names = [a.name for a in stmt.axes]
            mpos,npos = [names.index(x) for x in stmt.raster_axes]
            if npos != mpos+1:
                raise UnsupportedIR("nonadjacent raster axes")
            m,n = extents[mpos],extents[npos]
            inner = math.prod(extents[npos+1:])
            sub = self.binary("%",self.binary("/",gid,self.literal(inner,"s32")),self.literal(m*n,"s32"))
            group = stmt.raster_group
            major,minor = (m,n) if stmt.raster_orient == "m" else (n,m)
            group_id = self.binary("/",sub,self.literal(group*minor,"s32"))
            base = self.binary("*",group_id,self.literal(group,"s32"))
            remain = self.binary("-",self.literal(major,"s32"),base)
            active = self.fresh("s32")
            self.emit(f"min.s32 {active.operand}, {remain.operand}, {group};")
            within = self.binary("%",sub,self.literal(group*minor,"s32"))
            major_v = self.binary("+",base,self.binary("%",within,active))
            minor_v = self.binary("/",within,active)
            values = (major_v,minor_v) if stmt.raster_orient == "m" else (minor_v,major_v)
            for name,value in zip(stmt.raster_axes,values):
                self.bind(name,value)
        self.body(stmt.body)
        self.lines.append(end + ":")

    def statement(self, stmt):
        tag = type(stmt).__name__
        if tag == "Tile":
            self.tile(stmt)
        elif tag in ("Loop","StridedLoop"):
            self.loop(stmt)
        elif tag == "Cond":
            self.conditional(self.expr(stmt.cond),stmt.body,stmt.else_body)
        elif tag == "Smem":
            pass  # allocated once in the module prologue
        elif tag == "Sync":
            self.barrier(stmt.barrier_id,stmt.count,stmt.warp)
        elif tag == "Init":
            self.bind(stmt.name,self.literal(stmt.identity,typ(stmt.dtype)))
        elif tag == "Let":
            self.bind(stmt.name,self.convert(self.expr(stmt.value),typ(stmt.dtype)))
        elif tag == "Assign":
            kind = typ(stmt.dtype) if stmt.dtype is not None else self.expr(stmt.args[0]).kind
            self.bind(stmt.name,self.arithmetic(stmt.op.name,stmt.args,kind))
        elif tag == "Accum":
            lhs = self.expr(stmt.base if stmt.base is not None else stmt.name)
            self.bind(stmt.name,self.arithmetic(stmt.op.name,[lhs,self.expr(stmt.value)],typ(stmt.dtype)))
        elif tag == "Load":
            for i,name in enumerate(stmt.names):
                self.bind(name,self.load(stmt.input,stmt.index,extra=i))
        elif tag == "Write":
            for i,name in enumerate(stmt.values):
                self.store(stmt.output,stmt.index,self.expr(name),extra=i,atomic=stmt.atomic,swizzle=stmt.swizzle)
        elif tag == "CpAsyncCopy":
            dest = self.flat(stmt.smem,stmt.smem_index)
            if stmt.lane_index is not None:
                dest = self.binary("+",dest,self.flat(stmt.smem,stmt.lane_index))
            da = self.address(stmt.smem,flat=dest,swizzle=stmt.swizzle)
            sa = self.address(stmt.src,stmt.src_index)
            if stmt.nbytes not in (4,8,16):
                raise UnsupportedIR("invalid cp.async width")
            cache = "cg" if stmt.nbytes == 16 else "ca"
            self.emit(f"cp.async.{cache}.shared.global [{da.operand}], [{sa.operand}], {stmt.nbytes};")
        elif tag == "CpAsyncCommit":
            self.emit("cp.async.commit_group;")
        elif tag == "CpAsyncWait":
            self.emit(f"cp.async.wait_group {stmt.group};")
        elif tag == "WarpShuffle":
            offset = stmt.length//2
            while offset:
                for a,b in zip(stmt.state,stmt.state_b):
                    src = self.reg(self.expr(a))
                    dst = self.fresh(src.kind)
                    if src.kind in ("f64", "s64"):
                        # shfl moves 32 bits; preserve both words without conversion.
                        lo, hi = self.fresh("u32"), self.fresh("u32")
                        shuffled_lo, shuffled_hi = self.fresh("u32"), self.fresh("u32")
                        self.emit(f"mov.b64 {{{lo.operand}, {hi.operand}}}, {src.operand};")
                        self.emit(f"shfl.sync.bfly.b32 {shuffled_lo.operand}, {lo.operand}, {offset}, 31, 0xffffffff;")
                        self.emit(f"shfl.sync.bfly.b32 {shuffled_hi.operand}, {hi.operand}, {offset}, 31, 0xffffffff;")
                        self.emit(f"mov.b64 {dst.operand}, {{{shuffled_lo.operand}, {shuffled_hi.operand}}};")
                    else:
                        self.emit(f"shfl.sync.bfly.b32 {dst.operand}, {src.operand}, {offset}, 31, 0xffffffff;")
                    self.bind(b,dst)
                self.body(stmt.combine_states)
                offset //= 2
        elif tag == "TreeHalve":
            tid = self.expr(stmt.tid_var)
            scale = stmt.inner[1] if stmt.inner else 1
            inner = self.expr(stmt.inner[0]) if stmt.inner else self.literal(0,"s32")
            offset = stmt.length//2
            while offset:
                cond = self.binary("<",tid,self.literal(offset,"s32"))
                skip = self.label("tree_skip")
                self.emit(f"@!{cond.operand} bra {skip};")
                left = self.binary("+",self.binary("*",tid,self.literal(scale,"s32")),inner)
                right = self.binary("+",left,self.literal(offset*scale,"s32"))
                for buf,a,b in zip(stmt.bufs,stmt.state,stmt.state_b):
                    self.bind(a,self.load(buf,flat=left))
                    self.bind(b,self.load(buf,flat=right))
                self.body(stmt.combine_states)
                for buf,a in zip(stmt.bufs,stmt.state):
                    self.store(buf,None,self.expr(a),flat=left)
                self.lines.append(skip + ":")
                self.barrier(stmt.barrier_id,stmt.barrier_count)
                offset //= 2
            for buf,a in zip(stmt.bufs,stmt.state):
                self.bind(a,self.load(buf,flat=inner))
        else:
            raise UnsupportedIR(f"unsupported statement kind {tag}")

    def body(self, statements):
        for stmt in statements:
            self.statement(stmt)

    def compile(self):
        if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*",self.kernel.name):
            raise UnsupportedIR("invalid PTX entry identifier")
        if self.kernel.serial:
            raise UnsupportedIR("serial launch parameters not in this contest dialect")
        for i,name in enumerate(self.metadata["arg_order"]):
            if name not in self.buffers:
                raise UnsupportedIR(f"ABI buffer not in graph: {name}")
            ptr = self.fresh("u64")
            generic = self.fresh("u64")
            self.emit(f"ld.param.u64 {generic.operand}, [arg_{i}];")
            self.emit(f"cvta.to.global.u64 {ptr.operand}, {generic.operand};")
            self.buffers[name]["ptr"] = ptr
        for name,special in [("threadIdx.x","%tid.x"),("blockIdx.x","%ctaid.x"),("blockDim.x","%ntid.x")]:
            v = self.fresh("s32")
            self.emit(f"mov.u32 {v.operand}, {special};")
            self.env[name] = v
        self.env["lane"] = self.binary("&",self.env["threadIdx.x"],self.literal(31,"s32"))
        self.env["warp"] = self.binary(">>",self.env["threadIdx.x"],self.literal(5,"s32"))
        offset,layout = 0,[]
        for name,s in self.smem.items():
            kind = typ(s.dtype)
            size = {"f32":4,"f64":8,"s32":4,"s64":8,"f16":2}[kind]
            align = max(s.align,size)
            offset = (offset+align-1)//align*align
            layout.append((name,s,kind,offset,math.prod(s.extents)*size,align))
            offset += math.prod(s.extents)*size
        dynamic = offset > 48*1024
        if dynamic:
            if offset > self.metadata["smem_bytes"]:
                raise UnsupportedIR("dynamic shared footprint exceeds prescribed launch")
            self.shared_decls.append("    .extern .shared .align 16 .b8 dynamic_smem[];")
        for i,(name,s,kind,off,size,align) in enumerate(layout):
            ptr = self.fresh("u32")
            if dynamic:
                base = self.fresh("u32")
                self.emit(f"mov.u32 {base.operand}, dynamic_smem;")
                self.emit(f"add.u32 {ptr.operand}, {base.operand}, {off};")
            else:
                self.shared_decls.append(f"    .shared .align {align} .b8 shared_{i}[{size}];")
                self.emit(f"mov.u32 {ptr.operand}, shared_{i};")
            self.buffers[name] = {"shape":s.extents,"dtype":kind,"space":"shared","ptr":ptr}
        self.body(self.kernel.body)
        self.emit("ret;")
        params = ",\n".join(f"    .param .u64 arg_{i}" for i in range(len(self.metadata["arg_order"])))
        declarations = []
        for kind,count in self.counts.items():
            if count:
                prefix = {"f32":"f","f64":"d","s32":"r","s64":"l","u32":"u","u64":"rd","pred":"p","b16":"h"}[kind]
                declarations.append(f"    .reg .{kind} %{prefix}<{count}>;")
        module_shared = [s for s in self.shared_decls if s.strip().startswith(".extern")]
        local_shared = [s for s in self.shared_decls if not s.strip().startswith(".extern")]
        return (".version 8.0\n.target sm_89\n.address_size 64\n\n"
                + "\n".join(module_shared) + "\n"
                + f".visible .entry {self.kernel.name}(\n{params}\n)\n{{\n"
                + "\n".join(declarations+local_shared+self.lines) + "\n}\n")


def compile_file(path):
    path = Path(path)
    graph = Graph.from_dict(json.loads(path.read_text()))
    kernels = [n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]
    if len(kernels) != 1:
        raise UnsupportedIR("one KernelOp per input is required")
    meta = json.loads((path.parent / "meta.json").read_text())
    matching = [k for k in meta["kernels"] if k["entry"] == kernels[0].op.name]
    if len(matching) != 1:
        raise UnsupportedIR("entry is not uniquely described by launch metadata")
    return PTX(graph,kernels[0],matching[0]).compile()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python compile.py kernel.ir.json")
    try:
        print(compile_file(sys.argv[1]),end="")
    except (UnsupportedIR,KeyError,ValueError) as exc:
        print(f"Aurora PTX: {type(exc).__name__}: {exc}",file=sys.stderr)
        raise SystemExit(2)
