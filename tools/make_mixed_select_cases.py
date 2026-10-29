"""Original mixed-selection fixtures; no compiler, renderer, GPU or network."""
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import struct
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, F64, I32, I64, U32, BOOL
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Let, Load, Write

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
IDS = ("predicate_true_numeric_false", "numeric_true_predicate_false",
       "nested_literals_and_aliases", "selected_nonfinite_truth")

def bit(n):
    return BinaryExpr("!=", BinaryExpr("&", Var("mask"), Literal(n, "i32")), Literal(0, "i32"))

def total(values):
    value = values[0]
    for other in values[1:]:
        value = BinaryExpr("+", value, other)
    return value

def make_case(case_id):
    assert case_id in IDS
    original = PROJECT / "upstream/problems/p00_scalar_add"
    meta = json.loads((original / "meta.json").read_text())
    km = meta["kernels"][0]
    graph = Graph.from_dict(json.loads((original / (km["entry"] + ".ir.json")).read_text()))
    node = next(n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
    body = [Load(names=("a",), input="x0", index=(Var("cell"),), dtype=F32),
            Load(names=("b",), input="x1", index=(Var("cell"),), dtype=F32)]
    values = []
    types = (F32, F64, I32, I64, U32)
    if case_id in IDS[:2]:
        x0 = [float(i % 32) for i in range(128)]
        x1 = [(-2.5, -.5, .5, 2.5)[i // 32] for i in range(128)]
        body += [Let("mask", Var("a"), I32),
                 Let("condition", BinaryExpr(">", Var("b"), Literal(0.0, "float")), BOOL)]
        for j, dtype in enumerate(types):
            peer = Var("b") if j < 2 else BinaryExpr("+", Var("b"), Literal(8.0, "float"))
            body += [Let(f"flag{j}", bit(1 << j), BOOL), Let(f"peer{j}", peer, dtype)]
            a, b = Var(f"flag{j}"), Var(f"peer{j}")
            if case_id == IDS[1]:
                a, b = b, a
            body += [Let(f"selected{j}", TernaryExpr(Var("condition"), a, b), dtype),
                     Let(f"out{j}", Var(f"selected{j}"), F32)]
            values.append(BinaryExpr("*", Var(f"out{j}"), Literal(float(16**j), "float")))
        expected = []
        for a, b in zip(x0, x1):
            result = Fraction(0)
            for j in range(5):
                flag, peer = (int(a) >> j) & 1, Fraction(b) if j < 2 else int(b + 8)
                if case_id == IDS[0]:
                    selected = flag if b > 0 else peer
                else:
                    selected = peer if b > 0 else flag
                result += selected * 16**j
            expected.append(float(result))
        combinations = 128
    elif case_id == IDS[2]:
        x0 = [float(i % 16) for i in range(128)]
        peers = (-3.5, -2.5, -1.5, -.5, .5, 1.5, 2.5, 3.5)
        x1 = [peers[i // 16] for i in range(128)]
        body += [Let("mask", Var("a"), I32), Let("on", Literal(1, "bool"), BOOL),
                 Let("off", Literal(0, "bool"), BOOL), Let("alias", Var("on"), BOOL),
                 Let("inner_predicate", TernaryExpr(bit(2), Var("alias"), Var("off")), BOOL)]
        for j, dtype in enumerate(types):
            peer = Var("b") if j < 2 else BinaryExpr("+", Var("b"), Literal(8.0, "float"))
            body += [Let(f"peer{j}", peer, dtype),
                     Let(f"inner{j}", TernaryExpr(bit(1), Var("inner_predicate"), Var(f"peer{j}")), dtype),
                     Let(f"outer{j}", TernaryExpr(bit(4), Literal(0, "bool"),
                         TernaryExpr(bit(8), Var(f"inner{j}"), Literal(1, "bool"))), dtype),
                     Let(f"out{j}", Var(f"outer{j}"), F32)]
            values.append(BinaryExpr("*", Var(f"out{j}"), Literal(float(16**j), "float")))
        expected = []
        for a, b in zip(x0, x1):
            flags = [bool(int(a) & (1 << j)) for j in range(4)]
            result = Fraction(0)
            for j in range(5):
                peer = Fraction(b) if j < 2 else int(b + 8)
                inside = int(flags[1]) if flags[0] else peer
                middle = inside if flags[3] else 1
                result += (0 if flags[2] else middle) * 16**j
            expected.append(float(result))
        combinations = 128
    else:
        pairs = ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (-0.0, 1.0),
                 (.5, 1.0), (-.5, 1.0), (0.0, 1.0), (-0.0, -1.0))
        x0 = [pairs[i // 4 % 8][0] for i in range(128)]
        x1 = [pairs[i // 4 % 8][1] for i in range(128)]
        body += [Let("mask", Var("cell"), I32), Let("condition", bit(1), BOOL), Let("flag", bit(2), BOOL),
                 Let("ratio32", BinaryExpr("/", Var("a"), Var("b")), F32),
                 Let("a64", Var("a"), F64), Let("b64", Var("b"), F64),
                 Let("ratio64", BinaryExpr("/", Var("a64"), Var("b64")), F64)]
        for j, (dtype, ratio) in enumerate(((F32, "ratio32"), (F64, "ratio64"))):
            body += [Let(f"forward{j}", TernaryExpr(Var("condition"), Var("flag"), Var(ratio)), dtype),
                     Let(f"reverse{j}", TernaryExpr(Var("condition"), Var(ratio), Var("flag")), dtype),
                     Let(f"ftruth{j}", Var(f"forward{j}"), BOOL), Let(f"rtruth{j}", Var(f"reverse{j}"), BOOL)]
            values.append(TernaryExpr(Var(f"ftruth{j}"), Literal(float(1 << j), "float"), Literal(0.0, "float")))
            values.append(TernaryExpr(Var(f"rtruth{j}"), Literal(float(4 << j), "float"), Literal(0.0, "float")))
        expected = []
        for i, (a, b) in enumerate(zip(x0, x1)):
            condition, flag = bool(i & 1), bool(i & 2)
            ratio_truth = b == 0 or a != 0  # NaN and either infinity are true.
            forward = flag if condition else ratio_truth
            reverse = ratio_truth if condition else flag
            expected.append(float(3 * forward + 12 * reverse))
        combinations = 32
    assert len(expected) == 128
    assert all(struct.unpack("<f", struct.pack("<f", x))[0] == x for x in expected)
    body += [Let("answer", total(values), F32),
             Write(output="add", index=(Var("cell"),), values=("answer",), value_dtype=F32)]
    node.op = replace(node.op, name="k_r08_" + case_id,
                      body=(Tile(axes=(Axis("cell", Dim(128)),), body=tuple(body)),))
    km["entry"] = node.op.name
    assert km["arg_order"] == ["x0", "x1", "add"]
    assert km["grid"] == [1, 1, 1] and km["block"] == [256, 1, 1] and km["smem_bytes"] == 0
    for arg in km["args"]:
        assert arg["shape"] == [128] and arg["dtype"] == "f32"
        if not arg.get("output"):
            arg.pop("seed", None)
            arg["data"] = x0 if arg["name"] == "x0" else x1
    meta.update(id="local_r08_" + case_id, code="Original mixed-selection fixture, not hidden input", atol=0, rtol=0)
    return graph, meta, expected, combinations

def write_case(directory, case_id):
    directory = Path(directory)
    directory.mkdir(parents=True)
    graph, meta, expected, combinations = make_case(case_id)
    ir = directory / (meta["kernels"][0]["entry"] + ".ir.json")
    for p, obj in ((ir, graph.to_dict()), (directory / "meta.json", meta), (directory / "expected.json", {"add": expected})):
        with p.open("x") as f:
            json.dump(obj, f, indent=2, allow_nan=False, default=repr)
            f.write("\n")
    wire = json.loads(ir.read_text())
    assert json.loads(json.dumps(Graph.from_dict(wire).to_dict(), default=repr)) == wire
    return ir, combinations

if __name__ == "__main__":
    for case_id in IDS:
        print(write_case(HERE / "mixed_select_cases" / case_id, case_id)[0])
