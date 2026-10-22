"""Twenty preregistered shape variants, generated from typed IR, not strings.

Oracles are independent float64 numerical expressions evaluated on the public
runner's documented seeded-input rule. Original problems are never written.
"""
from dataclasses import fields,is_dataclass,replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph

ROOT=Path(__file__).resolve().parent
OPEN=ROOT.parent.parent/'upstream'/'problems'
OUT=ROOT/'shape_cases'
COMPILER=ROOT.parent/'compile.py'
FROZEN='dfac9e389ba7775e3f1adb1fb4858e2434e43c1b6b28bfaa1b5df46a961d7ee3'


def rewrite(obj,axes,literals):
    if isinstance(obj,tuple):
        return tuple(rewrite(x,axes,literals) for x in obj)
    if not is_dataclass(obj):
        return obj
    tag=type(obj).__name__
    if tag in {'Dim','DataType'}:
        return obj
    change={}
    if tag=='Axis' and obj.name in axes:
        change['extent']=Dim(axes[obj.name])
    if tag=='Literal' and obj.dtype=='int' and obj.value in literals:
        change['value']=literals[obj.value]
    for f in fields(obj):
        if f.name not in change and f.name not in {'op','dtype','value_dtype','source','inputs','outputs','knobs'}:
            change[f.name]=rewrite(getattr(obj,f.name),axes,literals)
    return replace(obj,**change)


def softmax(x):
    ex=np.exp(x-np.max(x,axis=-1,keepdims=True))
    return ex/np.sum(ex,axis=-1,keepdims=True)


def reference(family,data,meta):
    if family=='row_sum':
        return {'sum_1':data['x'].astype(np.float64).sum(axis=-1)}
    if family=='mean':
        return {'mean':data['x'].astype(np.float64).mean()}
    if family=='rmsnorm':
        x=data['x'].astype(np.float64);w=data['w'].astype(np.float64)
        eps=next(a['data'][0] for a in meta['kernels'][0]['args'] if a['name']=='add_c1')
        return {'mul_1':x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*w}
    if family=='softmax':
        return {'softmax':softmax(data['x'].astype(np.float64))}
    if family=='l2norm':
        x=data['x'].astype(np.float64)
        return {'div':x/np.sqrt(np.sum(x*x,axis=-1,keepdims=True))}
    if family in {'pipeline_matmul','splitk_matmul'}:
        return {'matmul':data['x0'].astype(np.float64)@data['x1'].astype(np.float64)}
    if family in {'attention','tiled_attention'}:
        scale=next(a['data'][0] for a in meta['kernels'][0]['args'] if a['name']=='div_c1')
        scores=(data['q'].astype(np.float64)@data['kt'].astype(np.float64))/scale
        return {'matmul_1':softmax(scores)@data['v'].astype(np.float64)}
    raise ValueError(family)


design=[]
for rows,cols in [(3,33),(7,257),(31,1000)]:
    design.append(('row_sum','r04_row_sum',(rows,cols)))
for n in [511,8193]:
    design.append(('mean','r03_mean',(n,)))
for family,base in [('rmsnorm','f01_rmsnorm'),('softmax','f02_row_softmax'),('l2norm','f03_l2norm')]:
    for shape in [(3,37),(7,777)]:design.append((family,base,shape))
for shape in [(64,128,192),(96,64,128),(32,192,320)]:
    design.append(('pipeline_matmul','m01_matmul_512',shape))
for shape in [(32,96,48),(48,32,40)]:
    design.append(('splitk_matmul','m02_matmul_rect',shape))
for shape in [(3,17,16,64),(9,33,8,32)]:
    design.append(('attention','a02_attention_128',shape))
for shape in [(32,7,8,32),(96,31,16,96)]:
    design.append(('tiled_attention','a01_attention',shape))
assert len(design)==20
assert hashlib.sha256(COMPILER.read_bytes()).hexdigest()==FROZEN
OUT.mkdir(exist_ok=False)
manifest=[]
sp=importlib.util.spec_from_file_location('frozen_aurora',COMPILER)
compiler=importlib.util.module_from_spec(sp);sys.modules[sp.name]=compiler;sp.loader.exec_module(compiler)
for family,base,shape in design:
    p=OPEN/base;meta=json.loads((p/'meta.json').read_text());mk=meta['kernels'][0]
    g=Graph.from_dict(json.loads((p/f"{mk['entry']}.ir.json").read_text()))
    kn=next(n for n in g.nodes.values() if type(n.op).__name__=='KernelOp')
    axes={};shapes={};constants={};literals={}
    if family=='mean':
        n,=shape;axes={'a0':n};shapes={'x':(n,),'mean':()};constants={'mean_keepdim_count':float(n)}
    elif family in {'row_sum','rmsnorm','softmax','l2norm'}:
        rows,cols=shape;axes={'a0':rows,'a1':cols}
        shapes={'x':(rows,cols)}
        if family=='row_sum':shapes['sum_1']=(rows,)
        if family=='rmsnorm':
            axes['a2']=cols;shapes.update(w=(cols,),mul_1=(rows,cols));constants['mean_count']=float(cols)
        if family=='softmax':axes['a3']=cols;shapes['softmax']=(rows,cols)
        if family=='l2norm':axes['a2']=cols;shapes['div']=(rows,cols)
    elif family in {'pipeline_matmul','splitk_matmul'}:
        m,n,k=shape;shapes={'x0':(m,k),'x1':(k,n),'matmul':(m,n)}
        axes={'a0':m,'a1':n}
        if family=='pipeline_matmul':
            assert m%32==n%64==k%64==0 and k>=128
            axes.update(a0_b=m//32,a1_b=n//64,_ks=k)
            literals={512:k,448:k-64}
        else:
            assert m%16==n%32==k%8==0
            axes.update(a0_b=m//16,a1_b=n//32,_a2_ks=k//8)
    else:
        m,keys,d,dv=shape
        shapes={'q':(m,d),'kt':(d,keys),'v':(keys,dv),'matmul_1':(m,dv)}
        axes={'a0':m,'a1':keys,'a2':d,'a5':dv,'a5_b':dv//32}
        if family=='attention':axes['a0_b']=m
        else:
            assert m%32==dv%32==0
            axes['a0_b']=m//32
    kn.op=replace(kn.op,body=rewrite(kn.op.body,axes,literals),
                  name='r01_'+family+'_'+'_'.join(map(str,shape)))
    for node in g.nodes.values():
        node.outputs=tuple(replace(t,shape=tuple(Dim(x) for x in shapes[t.name]))
                           if t.name in shapes else t for t in node.outputs)
    wire=g.to_dict()
    for node in wire['nodes'].values():
        if node['op']=='ConstantOp':
            for tensor in node['outputs']:
                if tensor['name'] in constants:node['op_fields']['value']=constants[tensor['name']]
    for a in mk['args']:
        if a['name'] in shapes:a['shape']=list(shapes[a['name']]) or [1]
        if a['name'] in constants:a['data']=[constants[a['name']]]
    tile=kn.op.body[0]
    cells=math.prod(a.extent.value for a in tile.axes)
    mk['grid']=[math.ceil(cells/mk['block'][0]),1,1]
    mk['entry']=kn.op.name
    case=family+'_'+'_'.join(map(str,shape));meta['id']=case
    meta['code']='Local typed-IR shape variant, not an organizer-hidden problem'
    data={a['name']:np.random.default_rng(a['seed']).standard_normal(a['shape']).astype(np.float32)
          for a in mk['args'] if 'seed' in a}
    expected={n:np.asarray(v,dtype=np.float32) for n,v in reference(family,data,meta).items()}
    pd=OUT/case;pd.mkdir()
    (pd/f"{mk['entry']}.ir.json").write_text(json.dumps(wire,indent=2,default=repr)+'\n')
    (pd/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    np.savez(pd/'expected.npz',**expected)
    for a in mk['args']:
        if a.get('output'):
            assert math.prod(expected[a['name']].shape or (1,))==math.prod(a['shape'])
    ptx=compiler.compile_file(pd/f"{mk['entry']}.ir.json")
    assert ptx.isascii()
    manifest.append({'id':case,'family':family,'source_problem':base,'shape':shape,
                     'rtol':meta['rtol'],'atol':meta['atol'],'grid':mk['grid'],'block':mk['block'],
                     'entry':mk['entry'],'emitted_bytes':len(ptx),
                     'ir_sha256':hashlib.sha256((pd/f"{mk['entry']}.ir.json").read_bytes()).hexdigest(),
                     'meta_sha256':hashlib.sha256((pd/'meta.json').read_bytes()).hexdigest(),
                     'oracle_sha256':hashlib.sha256((pd/'expected.npz').read_bytes()).hexdigest()})
    print(case,'LOCAL_PREFLIGHT_OK',flush=True)
assert len(manifest)==20
(OUT/'MANIFEST.json').write_text(json.dumps({'schema':'r01-shape-cases/v1','compiler_sha256':FROZEN,
    'count':20,'not_official_hidden':True,'cases':manifest},indent=2)+'\n')
