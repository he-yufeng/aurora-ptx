"""Original floating-comparison fixtures. Finite inputs; no compiler or network."""
from dataclasses import replace
import json
import math
import operator
from pathlib import Path
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, F64, I32, BOOL
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Cond, Let, Load, Write

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
IDS = ('f32_comparison_pairs','f64_comparison_pairs','mixed_boolean_inequality','inequality_control_flow')
COMPARISONS = ('==','!=','<','<=','>','>=')
OPS = (operator.eq,operator.ne,operator.lt,operator.le,operator.gt,operator.ge)
VALUES = (float('nan'),float('inf'),-float('inf'),0.0,-0.0,.5,-.5,1.0)
NUMERATORS = (0.0,1.0,-1.0,0.0,-0.0,.5,-.5,1.0)
DENOMINATORS = (0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0)

def select_code(code, numbers):
    expr = Literal(numbers[-1], 'float')
    for i in reversed(range(len(numbers)-1)):
        expr = TernaryExpr(BinaryExpr('==',Var(code),Literal(float(i),'float')),Literal(numbers[i],'float'),expr)
    return expr

def ratio(body, code, name, dtype):
    body += [Let(name+'_n',select_code(code,NUMERATORS),dtype),
             Let(name+'_d',select_code(code,DENOMINATORS),dtype),
             Let(name,BinaryExpr('/',Var(name+'_n'),Var(name+'_d')),dtype)]

def bit(n):
    return BinaryExpr('!=',BinaryExpr('&',Var('mask'),Literal(1<<n,'i32')),Literal(0,'i32'))

def sum_expr(values):
    expr = values[0]
    for other in values[1:]: expr = BinaryExpr('+',expr,other)
    return expr

def make_case(case_id):
    assert case_id in IDS
    original = PROJECT / 'upstream/problems/p00_scalar_add'
    meta = json.loads((original / 'meta.json').read_text())
    km = meta['kernels'][0]
    graph = Graph.from_dict(json.loads((original / (km['entry']+'.ir.json')).read_text()))
    node = next(n for n in graph.nodes.values() if type(n.op).__name__=='KernelOp')
    body = [Load(names=('a',),input='x0',index=(Var('cell'),),dtype=F32),
            Load(names=('b',),input='x1',index=(Var('cell'),),dtype=F32)]
    x0 = [float(i%8) for i in range(128)]
    expected = []
    if case_id in IDS[:2]:
        dtype = F32 if case_id == IDS[0] else F64
        x1 = [float(i//8%8) for i in range(128)]
        ratio(body,'a','left',dtype); ratio(body,'b','right',dtype)
        values = []
        for j,op in enumerate(COMPARISONS):
            body.append(Let(f'c{j}',BinaryExpr(op,Var('left'),Var('right')),BOOL))
            values.append(TernaryExpr(Var(f'c{j}'),Literal(float(1<<j),'float'),Literal(0.0,'float')))
        body += [Let('answer',sum_expr(values),F32),Write(output='add',index=(Var('cell'),),values=('answer',),value_dtype=F32)]
        expected = [float(sum((1<<j)*op(VALUES[int(a)],VALUES[int(b)]) for j,op in enumerate(OPS))) for a,b in zip(x0,x1)]
        combinations = 64
    else:
        x1 = [float(i//8) for i in range(128)]
        body.append(Let('mask',Var('b'),I32))
        for j in range(4): body.append(Let(f'flag{j}',bit(j),BOOL))
        ratio(body,'a','q32',F32); ratio(body,'a','q64',F64)
        if case_id == IDS[2]:
            values = []
            j = 0
            for q in ('q32','q64'):
                for k in range(4):
                    for left,right in ((Var(q),Var(f'flag{k}')),(Var(f'flag{k}'),Var(q))):
                        body.append(Let(f'c{j}',BinaryExpr('!=',left,right),BOOL))
                        values.append(TernaryExpr(Var(f'c{j}'),Literal(float(1<<j),'float'),Literal(0.0,'float')))
                        j += 1
            body += [Let('answer',sum_expr(values),F32),Write(output='add',index=(Var('cell'),),values=('answer',),value_dtype=F32)]
            for a,b in zip(x0,x1):
                v,mask = VALUES[int(a)],int(b)
                expected.append(float(sum((1<<(t*8+k*2+s))*(v!=bool(mask&(1<<k))) for t in range(2) for k in range(4) for s in range(2))))
        else:
            # Each of the four leaves writes; no initialization or partial-return assumption.
            body += [Let('c32',BinaryExpr('!=',Var('q32'),Var('flag0')),BOOL),
                     Let('c64',BinaryExpr('!=',Var('q64'),Var('flag1')),BOOL)]
            for name,offset in (('v11',100),('v10',200),('v01',300),('v00',400)):
                body.append(Let(name,BinaryExpr('+',Var('b'),Literal(float(offset),'float')),F32))
            def write(name): return (Write(output='add',index=(Var('cell'),),values=(name,),value_dtype=F32),)
            body.append(Cond(cond=Var('c32'),body=(Cond(cond=Var('c64'),body=write('v11'),else_body=write('v10')),),
                             else_body=(Cond(cond=Var('c64'),body=write('v01'),else_body=write('v00')),)))
            for a,b in zip(x0,x1):
                v,mask = VALUES[int(a)],int(b)
                c32,c64 = v!=bool(mask&1),v!=bool(mask&2)
                expected.append(float(mask + (100 if c32 and c64 else 200 if c32 else 300 if c64 else 400)))
        combinations = 128
    assert len(expected)==128 and all(math.isfinite(v) and 0<=v<2**24 and v==int(v) for v in expected)
    node.op=replace(node.op,name='k_r09_'+case_id,body=(Tile(axes=(Axis('cell',Dim(128)),),body=tuple(body)),))
    km['entry']=node.op.name
    assert km['arg_order']==['x0','x1','add'] and km['grid']==[1,1,1] and km['block']==[256,1,1] and km['smem_bytes']==0
    for arg in km['args']:
        assert arg['dtype']=='f32' and arg['shape']==[128]
        if not arg.get('output'):
            arg.pop('seed',None); arg['data']=x0 if arg['name']=='x0' else x1
    meta.update(id='local_r09_'+case_id,code='Original IEEE comparison fixture, not hidden input',atol=0,rtol=0)
    return graph,meta,expected,combinations

def write_case(directory,case_id):
    directory=Path(directory); directory.mkdir(parents=True)
    graph,meta,expected,combinations=make_case(case_id)
    ir=directory/(meta['kernels'][0]['entry']+'.ir.json')
    for p,obj in ((ir,graph.to_dict()),(directory/'meta.json',meta),(directory/'expected.json',{'add':expected})):
        with p.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False,default=repr); f.write('\n')
    wire=json.loads(ir.read_text())
    assert json.loads(json.dumps(Graph.from_dict(wire).to_dict(),default=repr))==wire
    return ir,combinations

if __name__=='__main__':
    for case_id in IDS: print(write_case(HERE/'float_comparison_cases'/case_id,case_id)[0])
