"""Four original predicate-operand fixtures; no compiler, renderer, or network."""
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import struct

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, F64, I32, I64, U32, BOOL
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Let, Load, Write

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
IDS = ("predicate_pair_comparisons", "predicate_pair_arithmetic",
       "predicate_numeric_peers", "predicate_left_shifts")
COMPARISONS = ("==", "!=", "<", "<=", ">", ">=")


def binary(op, a, b):
    return BinaryExpr(op, a, b)


def total(values):
    result = values[0]
    for value in values[1:]:
        result = binary("+", result, value)
    return result


def make_case(case_id):
    assert case_id in IDS
    original = PROJECT / "upstream/problems/p00_scalar_add"
    meta = json.loads((original / "meta.json").read_text())
    km = meta["kernels"][0]
    graph = Graph.from_dict(json.loads((original / (km["entry"] + ".ir.json")).read_text()))
    node = next(n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
    x0 = [-2.0, 3.0, -2.0, 3.0] * 32
    x1 = [-5.0, -5.0, 7.0, 7.0] * 32
    body = [Load(names=("a",), input="x0", index=(Var("cell"),), dtype=F32),
            Load(names=("b",), input="x1", index=(Var("cell"),), dtype=F32)]
    values = []

    def term(name, expr, weight):
        body.append(Let(name, expr, F32))
        values.append(binary("*", Var(name), Literal(float(weight), "float")))

    if case_id in IDS[:2]:
        body += [Let("p", binary(">", Var("a"), Literal(0.0, "float")), BOOL),
                 Let("q", binary(">", Var("b"), Literal(0.0, "float")), BOOL)]
        if case_id == IDS[0]:
            for j, op in enumerate(COMPARISONS):
                term(f"cmp{j}", binary(op, Var("p"), Var("q")), 1 << j)
            def oracle(a, b):
                p, q = a > 0, b > 0
                return sum(int(t) << j for j, t in enumerate(
                    (p == q, p != q, p < q, p <= q, p > q, p >= q)))
        else:
            # All a values are >= -2, so this *input-derived* predicate is nonzero.
            body.append(Let("nonzero", binary(">=", Var("a"), Literal(-2.0, "float")), BOOL))
            for j, op in enumerate(("+", "-", "*", "&", "|", "^", "/", "//", "%")):
                right = Var("q") if j < 6 else Var("nonzero")
                term(f"arith{j}", binary(op, Var("p"), right), 1 << (2 * j))
            def oracle(a, b):
                p, q = int(a > 0), int(b > 0)
                assert a >= -2
                terms = (p + q, p - q, p * q, p & q, p | q, p ^ q, p // 1, p // 1, p % 1)
                return sum(t << (2 * j) for j, t in enumerate(terms))
        expected = [float(oracle(a, b)) for a, b in zip(x0, x1)]
        combinations = 4
    elif case_id == IDS[2]:
        x0 = [float(i % 32) for i in range(128)]
        x1 = [(.5, 1.0, 2.0, 4.0)[i // 32] for i in range(128)]
        body.append(Let("mask", Var("a"), I32))
        for j, (dtype, op) in enumerate(zip((F32, F64, I32, I64, U32), ("/", "*", "+", "-", "%"))):
            flag = binary("!=", binary("&", Var("mask"), Literal(1 << j, "i32")), Literal(0, "i32"))
            peer = Var("b") if j < 2 else binary("+", Var("b"), Literal(2.0, "float"))
            body += [Let(f"flag{j}", flag, BOOL), Let(f"peer{j}", peer, dtype),
                     Let(f"left{j}", binary(op, Var(f"flag{j}"), Var(f"peer{j}")), F32),
                     Let(f"right{j}", binary("+", Var(f"peer{j}"), Var(f"flag{j}")), F32)]
            subtotal = binary("+", Var(f"left{j}"), binary("*", Var(f"right{j}"), Literal(8.0, "float")))
            term(f"mixed{j}", subtotal, 1 << (4 * j))
        expected = []
        for a, b in zip(x0, x1):
            p = [(int(a) >> j) & 1 for j in range(5)]
            f, n = Fraction(b), int(b + 2)
            left = (Fraction(p[0]) / f, p[1] * f, p[2] + n, p[3] - n, p[4] % n)
            peers = (f, f, n, n, n)
            value = sum((left[j] + 8 * (peers[j] + p[j])) * (1 << (4 * j)) for j in range(5))
            output = float(value)
            assert Fraction(output) == value
            assert struct.unpack("<f", struct.pack("<f", output))[0] == output
            expected.append(output)
        combinations = 128  # 32 independent Boolean states times four numeric peers.
    else:
        x0 = [float(i % 124) for i in range(128)]
        x1 = [1.0 if (i % 124 // 2) % 2 else -1.0 for i in range(128)]
        body += [Let("mask", Var("a"), I32),
                 Let("p", binary("!=", binary("&", Var("mask"), Literal(1, "i32")), Literal(0, "i32")), BOOL),
                 Let("q", binary(">", Var("b"), Literal(0.0, "float")), BOOL)]
        for j, dtype in enumerate((I32, I64, U32)):
            body += [Let(f"count{j}", binary("//", Var("mask"), Literal(4, "i32")), dtype),
                     Let(f"shifted{j}", binary("<<", Var("p"), Var(f"count{j}")), I32)]
            # Numeric data here is only s32; this is not the sealed R04 s64-data fix.
            term(f"roundtrip{j}", binary(">>", Var(f"shifted{j}"), Var(f"count{j}")), 1 << (2 * j))
            term(f"rightshift{j}", binary(">>", Var("p"), Var(f"count{j}")), 1 << (2 * j + 1))
        term("bool_leftshift", binary("<<", Var("p"), Var("q")), 64)
        term("bool_rightshift", binary(">>", Var("p"), Var("q")), 256)
        expected = []
        for a, b in zip(x0, x1):
            p, q, count = int(a) & 1, int(b > 0), int(a) // 4
            assert 0 <= count <= 30 and 0 <= (p << count) <= 2**30
            output = sum((((p << count) >> count) << (2 * j)) + ((p >> count) << (2 * j + 1)) for j in range(3))
            expected.append(float(output + (p << q) * 64 + (p >> q) * 256))
        combinations = 124
    body += [Let("answer", total(values), F32),
             Write(output="add", index=(Var("cell"),), values=("answer",), value_dtype=F32)]
    node.op = replace(node.op, name="k_r07_" + case_id,
                      body=(Tile(axes=(Axis("cell", Dim(128)),), body=tuple(body)),))
    km["entry"] = node.op.name
    assert km["arg_order"] == ["x0", "x1", "add"]
    assert km["grid"] == [1, 1, 1] and km["block"] == [256, 1, 1] and km["smem_bytes"] == 0
    for arg in km["args"]:
        assert arg["shape"] == [128] and arg["dtype"] == "f32"
        if not arg.get("output"):
            arg.pop("seed", None)
            arg["data"] = x0 if arg["name"] == "x0" else x1
    meta.update(id="local_r07_" + case_id, code="Original predicate-operand fixture, not hidden input", atol=0, rtol=0)
    return graph, meta, expected, combinations


def write_case(directory, case_id):
    directory = Path(directory)
    directory.mkdir(parents=True)
    graph, meta, expected, combinations = make_case(case_id)
    ir = directory / (meta["kernels"][0]["entry"] + ".ir.json")
    for p, obj in ((ir, graph.to_dict()), (directory / "meta.json", meta),
                   (directory / "expected.json", {"add": expected})):
        with p.open("x") as f:
            json.dump(obj, f, indent=2, allow_nan=False, default=repr)
            f.write("\n")
    wire = json.loads(ir.read_text())
    restored = Graph.from_dict(wire)
    assert json.loads(json.dumps(restored.to_dict(), default=repr)) == wire
    return ir, combinations


if __name__ == "__main__":
    for case_id in IDS:
        ir, _ = write_case(HERE / "predicate_binary_cases" / case_id, case_id)
        print(ir)
