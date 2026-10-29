"""IEEE inequality emission/fixture controls. Native proof is separate."""
import importlib.util
import itertools
import json
import math
import operator
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
compiler=module('r09_float_compiler',HERE/'compile.py')
generator=module('r09_float_generator',HERE/'tools/make_float_comparison_cases.py')
BASELINE=json.loads((HERE/'tests/predicate_binary_numeric_baseline.json').read_text())
KINDS=('f32','f64','s32','s64','u32','u64','pred')
PREFIX={'f32':'f','f64':'d','s32':'r','s64':'l','u32':'u','u64':'rd','pred':'p'}
def stub():
    p=compiler.PTX.__new__(compiler.PTX); p.lines=[]; p.env={}; p.counts={k:8 for k in (*KINDS,'b16')}; return p
def value(kind,n=0): return compiler.Value('%'+PREFIX[kind]+str(n),kind)

class FloatComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='aurora-r09-test-'); cls.root=Path(cls.temp.name)
        for case_id in generator.IDS: generator.write_case(cls.root/case_id,case_id)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def test_float_inequality_is_unordered(self):
        for kind in ('f32','f64'):
            with self.subTest(kind=kind):
                p=stub(); got=p.binary('!=',value(kind),value(kind,1)); self.assertEqual(got.kind,'pred'); self.assertIn('setp.neu.'+kind,p.lines[-1])
    def test_other_five_float_comparisons_stay_ordered(self):
        for kind,(op,cmp) in itertools.product(('f32','f64'),(('==','eq'),('<','lt'),('<=','le'),('>','gt'),('>=','ge'))):
            p=stub(); p.binary(op,value(kind),value(kind,1)); self.assertIn('setp.'+cmp+'.'+kind,p.lines[-1])
    def test_integer_inequality_keeps_ne(self):
        for kind in ('s32','s64','u32','u64'):
            p=stub(); p.binary('!=',value(kind),value(kind,1)); self.assertIn('setp.ne.'+kind,p.lines[-1]); self.assertNotIn('neu','\n'.join(p.lines))
    def test_mixed_predicate_float_both_orders_use_neu(self):
        for kind,reverse in itertools.product(('f32','f64'),(False,True)):
            with self.subTest(kind=kind,reverse=reverse):
                p=stub(); a,b=value('pred'),value(kind); a,b=(b,a) if reverse else (a,b); p.binary('!=',a,b)
                self.assertIn('setp.neu.'+kind,p.lines[-1]); self.assertIn('selp.'+kind,'\n'.join(p.lines))
    def test_predicate_pair_inequality_keeps_integer_comparison(self):
        p=stub(); p.binary('!=',value('pred'),value('pred',1)); self.assertIn('setp.ne.s32',p.lines[-1])
    def test_typed_float_literals_compare_unordered_in_both_positions(self):
        for kind,reverse,n in itertools.product(('f32','f64'),(False,True),(0,1)):
            with self.subTest(kind=kind,reverse=reverse,n=n):
                p=stub(); a,b=value(kind),p.literal(n,kind); a,b=(b,a) if reverse else (a,b); p.binary('!=',a,b); self.assertIn('setp.neu.'+kind,p.lines[-1])
    def test_numeric_truth_neu_is_preserved(self):
        for kind in ('f32','f64'):
            p=stub(); p.predicate(value(kind)); self.assertIn('setp.neu.'+kind,p.lines[-1])
    def test_original_160_rows_have_exactly_two_documented_corrections(self):
        self.assertEqual(len(BASELINE['rows']),160); corrections=0
        for row in BASELINE['rows']:
            p=stub(); got=p.binary(row['op'],value(row['left_kind']),value(row['right_kind'],1)); expected=row['lines']
            if row['op']=='!=' and row['left_kind'] in ('f32','f64'):
                self.assertEqual(row['left_kind'],row['right_kind']); expected=[s.replace('setp.ne.','setp.neu.') for s in expected]; corrections+=1
            with self.subTest(op=row['op'],kind=row['left_kind']): self.assertEqual((got.operand,got.kind,p.lines),(row['operand'],row['kind'],expected))
        self.assertEqual(corrections,2)
    def test_all_four_serialized_graphs_have_no_ordered_float_ne(self):
        for ir in sorted(self.root.glob('*/*.ir.json')):
            with self.subTest(case=ir.parent.name):
                text=compiler.compile_file(ir); self.assertNotRegex(text,r'setp\.ne\.f(?:32|64)\b'); self.assertRegex(text,r'setp\.neu\.f(?:32|64)\b')
    def test_pair_oracles_cover_every_special_value_pair(self):
        for case_id in generator.IDS[:2]:
            _,meta,expected,n=generator.make_case(case_id); self.assertEqual(n,64); self.assertEqual(expected[:64],expected[64:])
            args=meta['kernels'][0]['args']; a=next(x['data'] for x in args if x['name']=='x0'); b=next(x['data'] for x in args if x['name']=='x1')
            self.assertEqual(len(set(zip(a,b))),64)
            for i,y in enumerate(expected):
                self.assertEqual(bool(int(y)&2),not bool(int(y)&1))
                if int(a[i])==0 or int(b[i])==0: self.assertEqual(y,2.0)
    def test_nested_control_flow_contains_both_float_neu(self):
        text=compiler.compile_file(next((self.root/generator.IDS[3]).glob('*.ir.json')))
        self.assertIn('setp.neu.f32',text); self.assertIn('setp.neu.f64',text); self.assertRegex(text,r'@!?%p\d+ bra')
    def test_host_ieee_nan_and_signed_zero_oracle_controls(self):
        n=float('nan'); self.assertTrue(operator.ne(n,n)); self.assertTrue(operator.ne(n,0.0)); self.assertFalse(operator.eq(n,n))
        self.assertTrue(0.0==-0.0); self.assertFalse(0.0!=-0.0); self.assertEqual(math.copysign(1,generator.VALUES[4]),-1.0)

if __name__=='__main__': unittest.main()
