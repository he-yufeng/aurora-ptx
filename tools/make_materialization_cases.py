"""Four original scalar-materialization fixtures; no compiler or network."""
from dataclasses import replace
import json
from pathlib import Path
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, F64, I32, I64, U32, BOOL
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Let, Load, Write

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
IDS = ("pred_to_numeric_5types", "numeric_to_pred_5types",
       "nonfinite_numeric_truth", "literal_alias_predicate")

def bit(n):
    return BinaryExpr("!=", BinaryExpr("&", Var("cell"), Literal(n, "i32")), Literal(0, "i32"))

def total(values):
    result = values[0]
    for value in values[1:]:
        result = BinaryExpr("+", result, value)
    return result

def make_case(case_id):
    assert case_id in IDS
    original = PROJECT / "upstream/problems/p00_scalar_add"
    meta = json.loads((original / "meta.json").read_text())
    km = meta["kernels"][0]
    graph = Graph.from_dict(json.loads((original / (km["entry"] + ".ir.json")).read_text()))
    node = next(n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
    x0 = [0.5 if i & 16 else -0.5 for i in range(128)]
    x1 = [0.25 if i & 8 else -0.25 for i in range(128)]
    body = [
        Load(names=("a",), input="x0", index=(Var("cell"),), dtype=F32),
        Load(names=("b",), input="x1", index=(Var("cell"),), dtype=F32),
    ]
    types = (F32, F64, I32, I64, U32)
    if case_id == IDS[0]:
        predicates = [BinaryExpr(">", Var("a"), Literal(0.0, "float")),
                      BinaryExpr(">", Var("b"), Literal(0.0, "float")), bit(4), bit(2), bit(1)]
        values = []
        for j, (dtype, predicate) in enumerate(zip(types, predicates)):
            body += [Let(f"number{j}", predicate, dtype), Let(f"as_float{j}", Var(f"number{j}"), F32)]
            values.append(BinaryExpr("*", Var(f"as_float{j}"), Literal(float(1 << j), "float")))
        result = total(values)
        expected = [float(sum((1 << j) for j, truth in enumerate(
                    (x0[i] > 0, x1[i] > 0, bool(i & 4), bool(i & 2), bool(i & 1))) if truth))
                    for i in range(128)]
        combinations = 32
    elif case_id == IDS[1]:
        x0 = [0.5 if i & 16 else (-0.0 if i & 1 else 0.0) for i in range(128)]
        x1 = [-0.25 if i & 8 else 0.0 for i in range(128)]
        expressions = [Var("a"), Var("b"),
                       BinaryExpr("-", Literal(0, "i32"), BinaryExpr("&", Var("cell"), Literal(4, "i32"))),
                       BinaryExpr("-", Literal(0, "i32"), BinaryExpr("&", Var("cell"), Literal(2, "i32"))),
                       BinaryExpr("&", Var("cell"), Literal(1, "i32"))]
        values = []
        for j, (dtype, expr) in enumerate(zip(types, expressions)):
            body += [Let(f"input{j}", expr, dtype), Let(f"flag{j}", Var(f"input{j}"), BOOL)]
            values.append(TernaryExpr(Var(f"flag{j}"), Literal(float(1 << j), "float"), Literal(0.0, "float")))
        result = total(values)
        expected = [float(sum((1 << j) for j, value in enumerate(
                    (x0[i], x1[i], -(i & 4), -(i & 2), i & 1)) if value != 0))
                    for i in range(128)]
        combinations = 32
    elif case_id == IDS[2]:
        pairs = ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (-0.0, 1.0),
                 (0.5, 1.0), (-0.5, 1.0), (0.0, 1.0), (-0.0, -1.0))
        x0 = [pairs[i % 8][0] for i in range(128)]
        x1 = [pairs[i % 8][1] for i in range(128)]
        body += [Let("ratio32", BinaryExpr("/", Var("a"), Var("b")), F32),
                 Let("a64", Var("a"), F64), Let("b64", Var("b"), F64),
                 Let("ratio64", BinaryExpr("/", Var("a64"), Var("b64")), F64),
                 Let("truth32", Var("ratio32"), BOOL), Let("truth64", Var("ratio64"), BOOL)]
        result = BinaryExpr("+",
                    TernaryExpr(Var("truth32"), Literal(1.0, "float"), Literal(0.0, "float")),
                    TernaryExpr(Var("truth64"), Literal(2.0, "float"), Literal(0.0, "float")))
        # 0/0 is NaN; nonzero/0 is infinity; all are truthy. Outputs stay finite.
        expected = [3.0 if y == 0 or x != 0 else 0.0 for x, y in zip(x0, x1)]
        combinations = 8
    else:
        body += [Let("on", Literal(1, "bool"), BOOL), Let("off", Literal(0, "bool"), BOOL),
                 Let("alias", Var("on"), BOOL),
                 Let("one", Literal(1, "bool"), F32), Let("zero", Literal(0, "bool"), F32),
                 Let("choose", TernaryExpr(bit(4), Var("alias"), Var("off")), BOOL),
                 Let("picked", TernaryExpr(Var("choose"), Var("a"), Var("b")), F32),
                 Let("plus_one", BinaryExpr("+", Var("picked"), Var("one")), F32)]
        result = TernaryExpr(Literal(1, "bool"), Var("plus_one"), Var("zero"))
        expected = [(a if i & 4 else b) + 1.0 for i, (a, b) in enumerate(zip(x0, x1))]
        combinations = 8
    body += [Let("answer", result, F32),
             Write(output="add", index=(Var("cell"),), values=("answer",), value_dtype=F32)]
    node.op = replace(node.op, name="k_r06_" + case_id,
                      body=(Tile(axes=(Axis("cell", Dim(128)),), body=tuple(body)),))
    km["entry"] = node.op.name
    assert km["arg_order"] == ["x0", "x1", "add"]
    assert km["grid"] == [1, 1, 1] and km["block"] == [256, 1, 1] and km["smem_bytes"] == 0
    for arg in km["args"]:
        assert arg["shape"] == [128] and arg["dtype"] == "f32"
        if not arg.get("output"):
            arg.pop("seed", None)
            arg["data"] = x0 if arg["name"] == "x0" else x1
    meta.update(id="local_r06_" + case_id, code="Original scalar materialization fixture, not hidden input",
                atol=0, rtol=0)
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
        ir, _ = write_case(HERE / "materialization_cases" / case_id, case_id)
        print(ir)
