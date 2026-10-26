"""Five typed 64-bit reduction fixtures; independent oracles, no renderer/network."""
from dataclasses import fields, is_dataclass, replace
import json
import math
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import DataType, F64, I64
from emmy.compiler.graph import Graph

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / "upstream/problems/r04_row_sum"
DESIGN = (("f64", 64, 512), ("f64", 7, 257), ("f64", 3, 33),
          ("i64", 5, 257), ("i64", 3, 33))


def rewrite(obj, dtype, sizes, field=None):
    if isinstance(obj, DataType):
        return dtype if obj.name == "f32" else obj
    if isinstance(obj, str):
        if field == "dtype" and obj == "float":
            return "double" if dtype == F64 else "int64_t"
        return obj
    if isinstance(obj, tuple):
        return tuple(rewrite(x, dtype, sizes) for x in obj)
    if type(obj).__name__ == "Axis":
        return replace(obj, extent=Dim(sizes[obj.name])) if obj.name in sizes else obj
    if not is_dataclass(obj) or type(obj).__name__ == "Dim":
        return obj
    return replace(obj, **{
        f.name: rewrite(getattr(obj, f.name), dtype, sizes, f.name)
        for f in fields(obj) if f.name not in {"op", "source", "knobs"}
    })


def make_case(kind, rows, cols):
    dtype = F64 if kind == "f64" else I64
    name = f"{kind}_{rows}_{cols}"
    meta = json.loads((SOURCE / "meta.json").read_text())
    km = meta["kernels"][0]
    graph = Graph.from_dict(json.loads((SOURCE / (km["entry"] + ".ir.json")).read_text()))
    node = next(n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
    node.op = replace(node.op, body=rewrite(node.op.body, dtype, {"a0": rows, "a1": cols}),
                      name="k_r03_" + name)
    for n in graph.nodes.values():
        n.outputs = tuple(replace(t, dtype=dtype,
            shape=tuple(Dim(x) for x in ((rows, cols) if t.name == "x" else (rows,))))
            for t in n.outputs)
    km["entry"] = node.op.name
    km["grid"] = [rows, 1, 1]
    assert km["block"] == [128, 1, 1]
    if kind == "f64":
        values = np.random.default_rng(0).standard_normal((rows, cols)).tolist()
        expected = [math.fsum(row) for row in values]
        meta["rtol"] = meta["atol"] = 1e-10
    else:
        assert kind == "i64"
        values = [[(-1 if (i + j) % 2 else 1) *
                   ((1 << (34 + (i * cols + j) % 10)) + (i * cols + j) * 65537 + 3)
                   for j in range(cols)] for i in range(rows)]
        values[0][:6] = [-(1 << 40), (1 << 32) + 1, -1, 0, (1 << 40) + 7, -(1 << 32) - 9]
        expected = [sum(row) for row in values]
        assert all(-(1 << 63) <= v < (1 << 63) for v in expected)
        meta["rtol"] = meta["atol"] = 0
    for a in km["args"]:
        a["dtype"] = kind
        a["shape"] = [rows, cols] if a["name"] == "x" else [rows]
        if a["name"] == "x":
            a.pop("seed")
            a["data"] = [v for row in values for v in row]
    meta.update(id=name, code="Local typed wide-value reduction; not organizer hidden input")
    return graph.to_dict(), meta, {"sum_1": expected}


def main():
    out = ROOT / "wide_cases"
    out.mkdir(exist_ok=False)
    for kind, rows, cols in DESIGN:
        wire, meta, expected = make_case(kind, rows, cols)
        directory = out / meta["id"]
        directory.mkdir()
        for filename, data in ((meta["kernels"][0]["entry"] + ".ir.json", wire),
                               ("meta.json", meta), ("expected.json", expected)):
            with (directory / filename).open("x", encoding="utf-8") as f:
                json.dump(data, f, indent=2, allow_nan=False, default=repr)
                f.write("\n")
        print(meta["id"], "GENERATED_LOCAL_NOT_HIDDEN")


if __name__ == "__main__":
    main()
