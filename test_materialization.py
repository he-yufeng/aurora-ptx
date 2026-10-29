"""Fixed local regressions for scalar materialization, not native proof."""
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from dataclasses import replace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r06_materialization_candidate", HERE / "compile.py")
compiler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compiler
spec.loader.exec_module(compiler)
NUMERIC_BASELINE = json.loads((HERE / "tests/materialization_numeric_baseline.json").read_text())
gen_spec = importlib.util.spec_from_file_location("aurora_r06_case_generator_test", HERE / "tools/make_materialization_cases.py")
generator = importlib.util.module_from_spec(gen_spec)
sys.modules[gen_spec.name] = generator
gen_spec.loader.exec_module(generator)
KINDS = ("f32", "f64", "s32", "s64", "u32", "u64")
PREFIX = {"f32":"f", "f64":"d", "s32":"r", "s64":"l", "u32":"u", "u64":"rd", "pred":"p"}

def stub(module=compiler):
    p = module.PTX.__new__(module.PTX)
    p.lines, p.env = [], {}
    p.counts = {k: 8 for k in ("f32", "f64", "s32", "s64", "u32", "u64", "pred", "b16")}
    return p

def no_pred_cvt(test, text):
    test.assertNotRegex(text, r"\bcvt\S*\.pred(?:\.|\s)")
    test.assertNotRegex(text, r"\bcvt\S*\.[a-z0-9]+\.pred\s")

class MaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory(prefix="aurora-r06-test-")
        cls.case_root = Path(cls._temporary.name)
        for case_id in generator.IDS:
            generator.write_case(cls.case_root / case_id, case_id)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_predicate_to_all_numeric_register_types(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                p = stub()
                v = p.convert(compiler.Value("%p0", "pred"), kind)
                self.assertEqual(v.kind, kind)
                self.assertEqual(p.lines, [f"    selp.{kind} {v.operand}, {p.literal(1,kind).operand}, {p.literal(0,kind).operand}, %p0;"])

    def test_all_numeric_register_types_to_predicate(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                p = stub()
                src = compiler.Value("%" + PREFIX[kind] + "0", kind)
                v = p.convert(src, "pred")
                cmp = "neu" if kind.startswith("f") else "ne"
                self.assertEqual(v.kind, "pred")
                self.assertEqual(p.lines, [f"    setp.{cmp}.{kind} {v.operand}, {src.operand}, {p.literal(0,kind).operand};"])

    def test_same_type_is_identity(self):
        for kind in (*KINDS, "pred"):
            p = stub()
            src = compiler.Value("%" + PREFIX[kind] + "0", kind)
            self.assertIs(p.convert(src, kind), src)
            self.assertEqual(p.lines, [])

    def test_all_original_numeric_conversion_paths_unchanged(self):
        self.assertEqual(NUMERIC_BASELINE["source_sha256"], "c4ee6ce22a59373cdd839b42c52a49b6d931db0984a4e2c5942a52a2df0a2a62")
        rows = NUMERIC_BASELINE["rows"]
        self.assertEqual(len(rows), 36)
        self.assertEqual({(r["source_kind"], r["target_kind"]) for r in rows}, set(itertools.product(KINDS, repeat=2)))
        for row in rows:
            p = stub()
            src = "%" + PREFIX[row["source_kind"]] + "0"
            got = p.convert(compiler.Value(src, row["source_kind"]), row["target_kind"])
            self.assertEqual((got.operand, got.kind, p.lines), (row["operand"], row["result_kind"], row["lines"]))

    def test_predicate_copy_remains_legal_mov(self):
        p = stub()
        v = p.bind("kept", compiler.Value("%p0", "pred"))
        self.assertEqual(p.lines, [f"    mov.pred {v.operand}, %p0;"])

    def test_boolean_immediate_materializes_as_predicate_register(self):
        for bit in (0, 1):
            p = stub()
            got = p.predicate(compiler.Value(str(bit), "pred"))
            self.assertTrue(got.operand.startswith("%p"))
            self.assertEqual(p.lines, [f"    mov.pred {got.operand}, {bit};"])

    def test_boolean_immediate_to_numeric(self):
        for bit, kind in itertools.product((0, 1), KINDS):
            p = stub()
            got = p.convert(compiler.Value(str(bit), "pred"), kind)
            self.assertEqual(len(p.lines), 2)
            self.assertIn("mov.pred", p.lines[0])
            self.assertIn(f"selp.{kind} {got.operand}", p.lines[1])
            no_pred_cvt(self, "\n".join(p.lines))

    def test_float_truth_uses_unordered_nonzero_not_ordered_ne(self):
        for kind in ("f32", "f64"):
            p = stub()
            p.predicate(compiler.Value("%" + PREFIX[kind] + "0", kind))
            self.assertIn(f"setp.neu.{kind}", "\n".join(p.lines))
            self.assertNotIn(f"setp.ne.{kind}", "\n".join(p.lines))

    def test_rebinding_numeric_name_from_predicate_keeps_destination(self):
        p = stub()
        p.env["dest"] = compiler.Value("%f0", "f32")
        v = p.bind("dest", compiler.Value("%p0", "pred"))
        self.assertEqual(v.operand, "%f0")
        self.assertIn("selp.f32", "\n".join(p.lines))
        self.assertTrue(p.lines[-1].startswith("    mov.f32 %f0,"))
        no_pred_cvt(self, "\n".join(p.lines))

    def test_serialized_let_materialization_emits_valid_conversion(self):
        graph, meta, _, _ = generator.make_case("pred_to_numeric_5types")
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
        tile = node.op.body[0]
        body = (tile.body[0],
                generator.Let("numeric_boolean", generator.BinaryExpr(">", generator.Var("a"), generator.Literal(0.0, "float")), generator.F32),
                generator.Write(output="add", index=(generator.Var("cell"),), values=("numeric_boolean",), value_dtype=generator.F32))
        node.op = replace(node.op, name="k_r06_public_let_probe", body=(replace(tile, body=body),))
        meta["kernels"][0]["entry"] = node.op.name
        directory = self.case_root / "single_let_probe"
        directory.mkdir()
        ir = directory / (node.op.name + ".ir.json")
        ir.write_text(json.dumps(graph.to_dict(), default=repr))
        (directory / "meta.json").write_text(json.dumps(meta))
        text = compiler.compile_file(ir)
        self.assertIn("selp.f32", text)
        no_pred_cvt(self, text)

    def test_all_four_fixed_serialized_cases_emit_without_predicate_cvt(self):
        files = sorted(self.case_root.glob("*/*.ir.json"))
        self.assertEqual(len(files), 4)
        for ir in files:
            with self.subTest(case=ir.parent.name):
                text = compiler.compile_file(ir)
                self.assertTrue(text.isascii())
                no_pred_cvt(self, text)
                self.assertNotIn("selp.pred", text)

    def test_fixed_cases_cover_both_directions_and_internal_nonfinite_truth(self):
        forward = compiler.compile_file(next((self.case_root / "pred_to_numeric_5types").glob("*.ir.json")))
        reverse = compiler.compile_file(next((self.case_root / "numeric_to_pred_5types").glob("*.ir.json")))
        special = compiler.compile_file(next((self.case_root / "nonfinite_numeric_truth").glob("*.ir.json")))
        for kind in ("f32", "f64", "s32", "s64", "u32"):
            self.assertIn(f"selp.{kind}", forward)
            self.assertIn(f"setp.{'neu' if kind.startswith('f') else 'ne'}.{kind}", reverse)
        for kind in ("f32", "f64"):
            self.assertIn(f"div.rn.{kind}", special)
            self.assertIn(f"setp.neu.{kind}", special)
        for name in ("pred_to_numeric_5types", "numeric_to_pred_5types"):
            expected = json.loads((self.case_root / name / "expected.json").read_text())["add"]
            self.assertEqual(len(expected), 128)
            self.assertEqual(set(expected), set(range(32)))
