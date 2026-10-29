"""Predicate-operand regressions. Local emission checks are not native proof."""
import importlib.util
import itertools
import json
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

compiler = module("r07_predicate_binary_candidate", HERE / "compile.py")
generator = module("r07_predicate_binary_generator", HERE / "tools/make_predicate_binary_cases.py")
BASELINE = json.loads((HERE / "tests/predicate_binary_numeric_baseline.json").read_text())
KINDS = ("f32", "f64", "s32", "s64", "u32", "u64")
PREFIX = {"f32":"f", "f64":"d", "s32":"r", "s64":"l", "u32":"u", "u64":"rd", "pred":"p"}

def stub():
    p = compiler.PTX.__new__(compiler.PTX)
    p.lines, p.env = [], {}
    p.counts = {k: 8 for k in (*KINDS, "pred", "b16")}
    return p

def value(kind, n=0):
    return compiler.Value("%" + PREFIX[kind] + str(n), kind)

def valid_predicate_use(test, text):
    test.assertNotRegex(text, r"\b(?:setp\.[\w.]+|add|sub|mul(?:\.lo)?|div|rem|shr|shl)\.pred\b")
    test.assertNotRegex(text, r"\b(?:add|sub|mul|div|rem|shl|shr|and\.b|or\.b|xor\.b)[^\n;]*%p")

class PredicateBinaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="aurora-r07-test-")
        cls.case_root = Path(cls.temp.name)
        for case_id in generator.IDS:
            generator.write_case(cls.case_root / case_id, case_id)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pair_comparisons_promote_both_predicates_to_int(self):
        for op, cmp in zip(generator.COMPARISONS, ("eq", "ne", "lt", "le", "gt", "ge")):
            with self.subTest(op=op):
                p = stub()
                got = p.binary(op, value("pred"), value("pred", 1))
                self.assertEqual(got.kind, "pred")
                self.assertEqual(sum("selp.s32 " in line for line in p.lines), 2)
                self.assertIn(f"setp.{cmp}.s32", p.lines[-1])
                valid_predicate_use(self, "\n".join(p.lines))

    def test_pair_arithmetic_returns_int_not_predicate(self):
        for op, instruction in (("+","add"),("-","sub"),("*","mul.lo"),("/","div"),("//","div"),("%","rem")):
            with self.subTest(op=op):
                p = stub()
                got = p.binary(op, value("pred"), value("pred", 1))
                self.assertEqual(got.kind, "s32")
                self.assertIn(instruction + ".s32", p.lines[-1])
                valid_predicate_use(self, "\n".join(p.lines))

    def test_pair_bitwise_uses_integer_data_registers(self):
        for op, instruction in (("&","and"),("|","or"),("^","xor")):
            with self.subTest(op=op):
                p = stub()
                got = p.binary(op, value("pred"), value("pred", 1))
                self.assertEqual(got.kind, "s32")
                self.assertIn(instruction + ".b32", p.lines[-1])
                valid_predicate_use(self, "\n".join(p.lines))

    def test_predicate_left_uses_numeric_peer_arithmetic_type(self):
        for kind, op in itertools.product(KINDS, ("+", "-", "*", "/")):
            with self.subTest(kind=kind, op=op):
                p = stub()
                got = p.binary(op, value("pred"), value(kind))
                self.assertEqual(got.kind, kind)
                self.assertIn(f"selp.{kind}", p.lines[0])
                valid_predicate_use(self, "\n".join(p.lines))

    def test_predicate_left_comparison_uses_numeric_peer(self):
        for kind, op in itertools.product(KINDS, generator.COMPARISONS):
            with self.subTest(kind=kind, op=op):
                p = stub()
                got = p.binary(op, value("pred"), value(kind))
                self.assertEqual(got.kind, "pred")
                self.assertIn(f".{kind} ", p.lines[-1])
                self.assertIn(f"selp.{kind} ", p.lines[0])
                valid_predicate_use(self, "\n".join(p.lines))

    def test_numeric_left_predicate_right_remains_numeric(self):
        for kind, op in itertools.product(KINDS, ("+", "-", "*", "/", "==", "!=")):
            with self.subTest(kind=kind, op=op):
                p = stub()
                got = p.binary(op, value(kind), value("pred"))
                self.assertEqual(got.kind, "pred" if op in ("==", "!=") else kind)
                self.assertIn(f"selp.{kind} ", p.lines[0])
                valid_predicate_use(self, "\n".join(p.lines))

    def test_predicate_data_shift_promotes_to_int32_with_32bit_count(self):
        for kind, op in itertools.product(("s32", "s64", "u32", "u64"), ("<<", ">>")):
            with self.subTest(kind=kind, op=op):
                p = stub()
                got = p.binary(op, value("pred"), value(kind))
                self.assertEqual(got.kind, "s32")
                self.assertIn("shl.b32" if op == "<<" else "shr.s32", p.lines[-1])
                self.assertNotRegex(p.lines[-1], r"%(?:l|rd)\d+")
                valid_predicate_use(self, "\n".join(p.lines))

    def test_predicate_pair_shift_uses_int32_data_and_count(self):
        for op in ("<<", ">>"):
            p = stub()
            got = p.binary(op, value("pred"), value("pred", 1))
            self.assertEqual(got.kind, "s32")
            self.assertEqual(sum("selp.s32 " in line for line in p.lines), 2)
            valid_predicate_use(self, "\n".join(p.lines))

    def test_logical_pairs_keep_predicate_instructions_without_promotion(self):
        for op in ("&&", "and", "||", "or"):
            p = stub()
            got = p.binary(op, value("pred"), value("pred", 1))
            self.assertEqual(got.kind, "pred")
            instruction = "and" if op in ("&&", "and") else "or"
            self.assertEqual(p.lines, [f"    {instruction}.pred {got.operand}, %p0, %p1;"])

    def test_boolean_immediates_and_nested_comparison_materialize(self):
        p = stub()
        number = p.binary("+", compiler.Value("1", "pred"), compiler.Value("0", "pred"))
        self.assertEqual(number.kind, "s32")
        predicate = p.binary(">", number, p.literal(0, "s32"))
        p.binary("==", predicate, compiler.Value("1", "pred"))
        self.assertIn("mov.pred", "\n".join(p.lines))
        self.assertIn("setp.eq.s32", p.lines[-1])
        valid_predicate_use(self, "\n".join(p.lines))

    def test_all_fixed_serialized_graphs_lower_without_predicate_data_mismatch(self):
        files = sorted(self.case_root.glob("*/*.ir.json"))
        self.assertEqual(len(files), 4)
        for ir in files:
            with self.subTest(case=ir.parent.name):
                text = compiler.compile_file(ir)
                self.assertTrue(text.isascii())
                valid_predicate_use(self, text)

    def test_original_numeric_only_emission_matrix_is_unchanged(self):
        self.assertEqual(BASELINE["source_sha256"], "4961d6459327fea3d3a65ce5bcb069f5ee1e8791cd1dacbc9739a42e8a65dea6")
        self.assertGreaterEqual(len(BASELINE["rows"]), 100)
        for row in BASELINE["rows"]:
            p = stub()
            got = p.binary(row["op"], value(row["left_kind"]), value(row["right_kind"], 1))
            self.assertEqual((got.operand, got.kind, p.lines), (row["operand"], row["kind"], row["lines"]))
