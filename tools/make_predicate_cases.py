"""Generate four original predicate-select fixtures offline; never runs a GPU.

Requires the organizer checkout at ../upstream; refuses existing output folders.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, I32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Cond, Let, Load, Write

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
IDS=("predicate_truth8_f32","predicate_nested16_f32","predicate_cond_write_f32","predicate_to_s32_f32")


def save(p,x):
    with Path(p).open("x") as f:
        json.dump(x,f,indent=2,allow_nan=False,default=repr)
        f.write("\n")


def bit(n):
    return BinaryExpr("!=",BinaryExpr("&",Var("cell"),Literal(n,"i32")),Literal(0,"i32"))


def make_case(case_id):
    src=PROJECT / "upstream/problems/p00_scalar_add"
    meta=json.loads((src / "meta.json").read_text())
    km=meta["kernels"][0]
    graph=Graph.from_dict(json.loads((src / (km["entry"]+".ir.json")).read_text()))
    node=next(n for n in graph.nodes.values() if type(n.op).__name__=="KernelOp")
    x0=[1.0 if i&2 else -1.0 for i in range(128)]
    x1=[2.0 if i&1 else -2.0 for i in range(128)]
    c=bit(4)
    p=BinaryExpr(">",Var("x"),Literal(0.0,"float"))
    q=BinaryExpr(">",Var("y"),Literal(0.0,"float"))
    selected=TernaryExpr(c,p,q)
    body=[Load(names=("x",),input="x0",index=(Var("cell"),),dtype=F32),
          Load(names=("y",),input="x1",index=(Var("cell"),),dtype=F32)]
    if case_id=="predicate_nested16_f32":
        a,b=bit(8),bit(4)
        selected=TernaryExpr(TernaryExpr(a,b,p),TernaryExpr(q,a,b),TernaryExpr(b,p,q))
        expected=[]
        for i in range(128):
            av,bv,cv,dv=bool(i&8),bool(i&4),x0[i]>0,x1[i]>0
            take=(av if dv else bv) if (bv if av else cv) else (cv if bv else dv)
            expected.append(7.0 if take else -3.0)
        truth_combinations=16
    else:
        take=[(x0[i]>0) if (i&4) else (x1[i]>0) for i in range(128)]
        expected=[7.0 if t else -3.0 for t in take]
        truth_combinations=8
    if case_id=="predicate_cond_write_f32":
        body.append(Cond(selected,
            body=(Write(output="add",index=(Var("cell"),),values=("x",),value_dtype=F32),),
            else_body=(Write(output="add",index=(Var("cell"),),values=("y",),value_dtype=F32),)))
        expected=[x0[i] if take[i] else x1[i] for i in range(128)]
    else:
        integer=case_id=="predicate_to_s32_f32"
        value=TernaryExpr(selected,Literal(123 if integer else 7.0,"i32" if integer else "float"),
                          Literal(-45 if integer else -3.0,"i32" if integer else "float"))
        dtype=I32 if integer else F32
        body.extend((Let("chosen",value,dtype),Write(output="add",index=(Var("cell"),),values=("chosen",),value_dtype=dtype)))
        if integer:
            expected=[123.0 if t else -45.0 for t in take]
    assert len(expected)==128 and all(abs(x)<2**24 and x==int(x) for x in expected)
    node.op=replace(node.op,name="k_r05_"+case_id,body=(Tile(axes=(Axis("cell",Dim(128)),),body=tuple(body)),))
    km["entry"]=node.op.name
    assert km["arg_order"]==["x0","x1","add"] and km["grid"]==[1,1,1] and km["block"]==[256,1,1] and km["smem_bytes"]==0
    for a in km["args"]:
        assert a["dtype"]=="f32" and a["shape"]==[128]
        if not a.get("output"):
            a.pop("seed")
            a["data"]=x0 if a["name"]=="x0" else x1
    meta.update(id=case_id,code="Original predicate semantics case, not hidden evaluation",atol=0,rtol=0)
    return graph.to_dict(),meta,{"add":expected},truth_combinations


def write_case(directory, case_id):
    directory=Path(directory)
    directory.mkdir()
    wire,meta,expected,_=make_case(case_id)
    ir_name=meta["kernels"][0]["entry"]+".ir.json"
    for name,data in ((ir_name,wire),("meta.json",meta),("expected.json",expected)):
        save(directory / name,data)
    serialized=json.loads((directory / ir_name).read_text())
    assert json.loads(json.dumps(Graph.from_dict(serialized).to_dict(),default=repr))==serialized
    return directory / ir_name


def main():
    assert len(sys.argv)<=2,"Usage: python make_predicate_cases.py [new-output-directory]"
    output=Path(sys.argv[1]) if len(sys.argv)==2 else HERE / "predicate_cases"
    output.mkdir()
    for case_id in IDS:
        write_case(output / case_id,case_id)
    print("Generated four original cases; no GPU/network requests")


if __name__=="__main__":
    main()
