"""Local regressions only. Native runner outcomes remain separate evidence."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("aurora_ptx", ROOT / "compile.py")
compiler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compiler
spec.loader.exec_module(compiler)
PROBLEMS = ROOT.parent / "upstream" / "problems"


class LoweringTests(unittest.TestCase):
    def test_bitwise_uses_bit_type(self):
        ir = next((PROBLEMS / "p00_scalar_add").glob("*.ir.json"))
        ptx = compiler.compile_file(ir)
        self.assertIn("and.b32", ptx)
        self.assertNotIn("and.s32", ptx)

    def test_dynamic_shared_is_module_scoped(self):
        ir = next((PROBLEMS / "m01_matmul_512").glob("*.ir.json"))
        ptx = compiler.compile_file(ir)
        self.assertLess(ptx.index(".extern .shared"), ptx.index(".visible .entry"))
        self.assertIn("cp.async.cg.shared.global", ptx)

    def test_all_fifteen_emit_ascii(self):
        files = sorted(PROBLEMS.glob("*/*.ir.json"))
        self.assertEqual(len(files), 15)
        for ir in files:
            with self.subTest(problem=ir.parent.name):
                ptx = compiler.compile_file(ir)
                self.assertTrue(ptx.isascii())
                self.assertTrue(ptx.startswith(".version"))


if __name__ == "__main__":
    unittest.main()
