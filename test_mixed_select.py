"""Finite local mixed-selection emission tests, not a native correctness proof."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

compiler = module("r08_mixed_candidate", HERE / "compile.py")
generator = module("r08_mixed_generator", HERE / "tools/make_mixed_select_cases.py")
BASELINE = json.loads((HERE / "tests/mixed_select_numeric_baseline.json").read_text())
KINDS = ("f32", "f64", "s32", "s64", "u32", "u64")
PREFIX = {"f32":"f", "f64":"d", "s32":"r", "s64":"l", "u32":"u", "u64":"rd", "pred":"p"}

def stub(a="pred", b="f32", condition="pred"):
    p = compiler.PTX.__new__(compiler.PTX)
    p.lines = []
    p.env = {name:compiler.Value("%" + PREFIX[kind] + str(i), kind)
             for i,(name,kind) in enumerate((("c",condition),("a",a),("b",b)))}
    p.counts = {k:8 for k in (*KINDS,"pred","b16")}
    return p

def choose(p, a=None, b=None, condition=None):
    return p.expr(generator.TernaryExpr(condition or generator.Var("c"), a or generator.Var("a"), b or generator.Var("b")))

def valid(test,p):
    text = "\n".join(p.lines)
    test.assertNotIn("selp.pred",text)
    test.assertNotRegex(text,r"\bcvt\S*\.pred\b")

class MixedSelectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="aurora-r08-test-")
        cls.root = Path(cls.temp.name)
        for case_id in generator.IDS:
            generator.write_case(cls.root / case_id,case_id)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_predicate_true_numeric_false_all_register_types(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                p=stub(b=kind)
                out=choose(p)
                self.assertEqual(out.kind,kind)
                self.assertIn("selp."+kind,p.lines[-1])
                self.assertEqual(sum("selp."+kind+" " in x for x in p.lines),2)
                valid(self,p)

    def test_numeric_true_predicate_false_keeps_existing_numeric_result(self):
        for kind in KINDS:
            p=stub(a=kind,b="pred")
            self.assertEqual(choose(p).kind,kind)
            valid(self,p)

    def test_predicate_pair_keeps_four_logic_instructions(self):
        p=stub(a="pred",b="pred")
        out=choose(p)
        self.assertEqual(out.kind,"pred")
        self.assertEqual(len(p.lines),4)
        for line,op in zip(p.lines,("not.pred","and.pred","and.pred","or.pred")):
            self.assertTrue(line.strip().startswith(op))
        valid(self,p)

    def test_numeric_arm_and_condition_matrix_is_unchanged(self):
        self.assertEqual(BASELINE["source_sha256"],"5e6adddeedf59c4be833003b2277760b79db234f665f1f494c22bee228064687")
        self.assertEqual(len(BASELINE["rows"]),72)
        for row in BASELINE["rows"]:
            p=stub(a=row["a_kind"],b=row["b_kind"],condition=row["condition_kind"])
            got=choose(p)
            self.assertEqual((got.operand,got.kind,p.lines),(row["operand"],row["kind"],row["lines"]))

    def test_boolean_immediate_true_arm_materializes_for_every_numeric_type(self):
        for kind in KINDS:
            for bit in (0,1):
                with self.subTest(kind=kind,bit=bit):
                    p=stub(b=kind)
                    self.assertEqual(choose(p,a=generator.Literal(bit,"bool")).kind,kind)
                    self.assertIn("mov.pred","\n".join(p.lines))
                    valid(self,p)

    def test_numeric_false_literals_do_not_collapse_to_boolean(self):
        for literal,kind in ((.5,"f32"),(-2.5,"f32"),(2.5,"f64"),(7,"s32"),(9,"s64"),(3,"u32")):
            with self.subTest(kind=kind,literal=literal):
                p=stub()
                spell={"f32":"float","f64":"double","s32":"i32","s64":"i64","u32":"u32"}[kind]
                got=choose(p,b=generator.Literal(literal,spell))
                self.assertEqual(got.kind,kind)
                self.assertIn(p.literal(literal,kind).operand,p.lines[-1])
                valid(self,p)

    def test_each_numeric_condition_type_keeps_numeric_selected_data(self):
        for kind in (*KINDS,"pred"):
            with self.subTest(condition=kind):
                p=stub(condition=kind)
                self.assertEqual(choose(p).kind,"f32")
                self.assertIn("selp.f32",p.lines[-1])
                valid(self,p)

    def test_inner_predicate_pair_then_numeric_selection(self):
        p=stub()
        inner=generator.TernaryExpr(generator.Var("c"),generator.Var("a"),generator.Literal(0,"bool"))
        out=choose(p,a=inner)
        self.assertEqual(out.kind,"f32")
        self.assertIn("or.pred","\n".join(p.lines))
        valid(self,p)

    def test_nested_mixed_selections_propagate_numeric_peer_type(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                p=stub(b=kind)
                inner=generator.TernaryExpr(generator.Var("c"),generator.Var("a"),generator.Var("b"))
                outer=generator.TernaryExpr(generator.Var("c"),generator.Literal(0,"bool"),inner)
                self.assertEqual(p.expr(outer).kind,kind)
                valid(self,p)

    def test_alias_predicate_is_not_mutated_by_selection_peer_type(self):
        p=stub()
        original=p.env["a"]
        for kind in KINDS:
            p.env["b"]=compiler.Value("%"+PREFIX[kind]+"2",kind)
            self.assertEqual(choose(p).kind,kind)
            self.assertIs(p.env["a"],original)
            self.assertEqual(original.kind,"pred")
        valid(self,p)

    def test_all_four_serialized_graphs_lower_without_predicate_selp(self):
        files=sorted(self.root.glob("*/*.ir.json"))
        self.assertEqual(len(files),4)
        for ir in files:
            with self.subTest(case=ir.parent.name):
                text=compiler.compile_file(ir)
                self.assertTrue(text.isascii())
                self.assertNotIn("selp.pred",text)

    def test_fixed_fixture_coverage_and_finite_outputs(self):
        self.assertEqual(len(generator.IDS),4)
        for case_id,count in zip(generator.IDS,(128,128,128,32)):
            graph,meta,expected,combinations=generator.make_case(case_id)
            self.assertEqual(combinations,count)
            self.assertEqual(len(expected),128)
            self.assertTrue(all(math.isfinite(x) for x in expected))
            self.assertEqual(meta["kernels"][0]["arg_order"],["x0","x1","add"])
