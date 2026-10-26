"""Focused transport regressions, not native GPU or hidden-evaluation tests."""
import importlib.util
from pathlib import Path
import random
import re
import struct
import sys
import unittest

from emmy.compiler.dtype import F32, F64, I64
from emmy.compiler.ir.kernel.ir import WarpShuffle

SPEC = importlib.util.spec_from_file_location("wide_shuffle_candidate", Path(__file__).with_name("compile.py"))
compiler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compiler
SPEC.loader.exec_module(compiler)


def transport(kind, dtype):
    ptx = compiler.PTX.__new__(compiler.PTX)
    ptx.counts = {k: 0 for k in ("f32", "f64", "s32", "s64", "u32", "u64", "pred", "b16")}
    ptx.lines, ptx.env = [], {}
    ptx.env["state"] = ptx.fresh(kind)
    ptx.statement(WarpShuffle(state=("state",), state_b=("peer",),
                              combine_states=(), length=32, dtype=dtype))
    return "\n".join(ptx.lines)


class WideShuffleTests(unittest.TestCase):
    def check_wide(self, kind, dtype, prefix):
        text = transport(kind, dtype)
        shuffles = [line for line in text.splitlines() if "shfl.sync" in line]
        self.assertEqual(len(shuffles), 10)
        self.assertNotIn(".b32 %" + prefix, text)
        self.assertEqual(text.count("mov.b64"), 10)
        for offset in (16, 8, 4, 2, 1):
            self.assertEqual(sum(f", {offset}, 31, 0xffffffff;" in line for line in shuffles), 2)
        unpacked = re.findall(r"mov.b64 \{(%u\d+), (%u\d+)\}, (%\w+);", text)
        packed = re.findall(r"mov.b64 (%\w+), \{(%u\d+), (%u\d+)\};", text)
        self.assertEqual(len(unpacked), 5)
        self.assertEqual(len(packed), 5)
        for (low, high, _), (_, out_low, out_high), offset in zip(unpacked, packed, (16, 8, 4, 2, 1)):
            self.assertIn(f"shfl.sync.bfly.b32 {out_low}, {low}, {offset}, 31, 0xffffffff;", text)
            self.assertIn(f"shfl.sync.bfly.b32 {out_high}, {high}, {offset}, 31, 0xffffffff;", text)

    def test_f64_uses_both_words(self):
        self.check_wide("f64", F64, "d")

    def test_s64_uses_both_words(self):
        self.check_wide("s64", I64, "l")

    def test_f32_path_stays_single_word(self):
        text = transport("f32", F32)
        self.assertEqual(text.count("shfl.sync.bfly.b32"), 5)
        self.assertNotIn("mov.b64", text)
        self.assertNotIn("%u", text)

    def test_raw_low_high_word_reconstruction(self):
        rng = random.Random(64)
        patterns = [0, 1, (1 << 32) - 1, 1 << 32, (1 << 64) - 1,
                    0x8000000000000000, 0x3FF0000000000000, 0x7FF0000000000000,
                    0xFFF0000000000000, 0x7FF8000000000042]
        patterns.extend(rng.getrandbits(64) for _ in range(1024))
        for bits in patterns:
            with self.subTest(bits=bits):
                low, high = bits & 0xFFFFFFFF, bits >> 32
                self.assertEqual(struct.pack("<II", low, high), struct.pack("<Q", bits))
                self.assertEqual(low | (high << 32), bits)

    def test_canonical_shared_i64_alias(self):
        self.assertEqual(compiler.typ("int64_t"), "s64")
        self.assertEqual(compiler.typ(I64), "s64")


if __name__ == "__main__":
    unittest.main()
