"""Finite local predicate truth tables; separate from native GPU evidence."""
import importlib.util
import itertools
from pathlib import Path
import re
import sys
import tempfile
import unittest

from emmy.compiler.ir.expr import TernaryExpr, Var

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("aurora_r05_test", HERE / "compile.py")
compiler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compiler
spec.loader.exec_module(compiler)


def stub(kinds):
    p = compiler.PTX.__new__(compiler.PTX)
    p.lines = []
    p.counts = {k: 8 for k in ("f32", "f64", "s32", "s64", "u32", "u64", "pred", "b16")}
    prefix = {"pred": "p", "f32": "f", "f64": "d", "s32": "r", "s64": "l", "u32": "u"}
    p.env = {name: compiler.Value(f"%{prefix[kind]}{i}", kind) for i, (name, kind) in enumerate(kinds.items())}
    return p


def evaluate(lines, initial):
    # A tiny independent executor for only the four documented Boolean ops.
    values = dict(initial)
    for line in lines:
        tokens = line.strip().rstrip(";").replace(",", "").split()
        op, dst, *src = tokens
        args = [values[x] for x in src]
        if op == "not.pred":
            values[dst] = not args[0]
        elif op == "and.pred":
            values[dst] = args[0] and args[1]
        elif op == "or.pred":
            values[dst] = args[0] or args[1]
        else:
            raise AssertionError("Not a valid predicate-only lowering: " + op)
    return values


class PredicateSelectTests(unittest.TestCase):
    def check(self, expr, names, oracle):
        p = stub(dict.fromkeys(names, "pred"))
        result = p.expr(expr)
        self.assertEqual(result.kind, "pred")
        self.assertNotIn("selp.pred", "\n".join(p.lines))
        for bits in itertools.product((False, True), repeat=len(names)):
            initial = {p.env[n].operand: bit for n, bit in zip(names, bits)}
            actual = evaluate(p.lines, initial)
            self.assertEqual(actual[result.operand], oracle(*bits))
            for name, bit in zip(names, bits):
                self.assertEqual(actual[p.env[name].operand], bit)
        return p

    def test_all_eight_truth_combinations(self):
        self.check(TernaryExpr(Var("c"), Var("a"), Var("b")), ("c", "a", "b"), lambda c,a,b: a if c else b)

    def test_four_variable_nested_all_sixteen(self):
        a,b,c,d = map(Var, ("a", "b", "c", "d"))
        expr = TernaryExpr(TernaryExpr(a,b,c), TernaryExpr(d,a,b), TernaryExpr(b,c,d))
        self.check(expr, ("a","b","c","d"), lambda a,b,c,d: (a if d else b) if (b if a else c) else (c if b else d))

    def test_same_true_and_false_source(self):
        self.check(TernaryExpr(Var("c"),Var("a"),Var("a")), ("c","a"), lambda c,a:a)

    def test_true_branch_aliases_condition(self):
        self.check(TernaryExpr(Var("c"),Var("c"),Var("b")), ("c","b"), lambda c,b:True if c else b)

    def test_false_branch_aliases_condition(self):
        self.check(TernaryExpr(Var("c"),Var("a"),Var("c")), ("c","a"), lambda c,a:a if c else False)

    def test_both_children_are_nested(self):
        a,b,c=map(Var,("a","b","c"))
        self.check(TernaryExpr(c,TernaryExpr(a,b,c),TernaryExpr(b,a,c)), ("a","b","c"), lambda a,b,c:(b if a else c) if c else (a if b else c))

    def numeric(self, kind):
        p=stub({"c":"pred","a":kind,"b":kind})
        result=p.expr(TernaryExpr(Var("c"),Var("a"),Var("b")))
        self.assertEqual(result.kind,kind)
        self.assertEqual(p.lines,[f"    selp.{kind} {result.operand}, {p.env['a'].operand}, {p.env['b'].operand}, {p.env['c'].operand};"])

    def test_f32_numeric_path_unchanged(self):
        self.numeric("f32")

    def test_f64_numeric_path_unchanged(self):
        self.numeric("f64")

    def test_integer_numeric_paths_unchanged(self):
        for kind in ("s32","s64","u32"):
            with self.subTest(kind=kind):
                self.numeric(kind)

    def test_predicate_uses_only_legal_four_ops(self):
        p=self.check(TernaryExpr(Var("c"),Var("a"),Var("b")),("c","a","b"),lambda c,a,b:a if c else b)
        self.assertEqual([x.strip().split()[0] for x in p.lines],["not.pred","and.pred","and.pred","or.pred"])
        self.assertEqual(len({re.split(r"[ ,]",x.strip())[1] for x in p.lines}),4)

    def test_ssa_names_are_not_dispatch_keys(self):
        self.check(TernaryExpr(Var("omega"),Var("north"),Var("south")),("omega","north","south"),lambda c,a,b:a if c else b)

    def test_legal_serialized_probe_compiles_without_selp_pred(self):
        fixture_spec=importlib.util.spec_from_file_location("r05_public_fixture_generator",HERE / "tools/make_predicate_cases.py")
        generator=importlib.util.module_from_spec(fixture_spec)
        sys.modules[fixture_spec.name]=generator
        fixture_spec.loader.exec_module(generator)
        with tempfile.TemporaryDirectory(prefix="aurora-predicate-test-") as temporary:
            ir=generator.write_case(Path(temporary) / "truth8","predicate_truth8_f32")
            text=compiler.compile_file(ir)
        self.assertNotIn("selp.pred",text)
        self.assertIn("not.pred",text)
        self.assertIn("selp.f32",text)
