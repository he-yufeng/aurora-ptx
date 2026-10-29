# Aurora PTX

An independent Python compiler from Emmy's typed Kernel IR to NVIDIA PTX, developed for the [Emmy Hackathon 2026](https://riftstack.ai/hackathon). It walks the scheduled IR directly; it does **not** call Emmy's CUDA renderer, `emmy compile`, `nvcc`, or another compiler.

## Current evidence

On 2026-10-22, the same compiler source passed all **15/15 official open problems** on the organizer's RTX 5090 runner, covering pointwise, reductions, matmul, fused normalization/softmax, and attention-like kernels. It additionally passed **21/21 local metamorphic cases**: all15 with renamed kernel/buffer/SSA/axis identities and a reversed argument ABI, plus six different static pointwise sizes.

These are development correctness results, **not a hidden-set score, ranking or prize**. See [development_evidence.json](development_evidence.json) for source binding, per-case errors, and the initial failed full attempt. That dated evidence predates the separate October24 project/video submission. No kernel-speed superiority is claimed.

A separate frozen-source R01 campaign passed **20/20 additional local shape cases** on the same runner: reductions with tail lanes, fused row widths, different asynchronous-matmul K lengths, split-K geometries, grouped-raster tail groups, and attention sequence/head/output sizes. A read-only audit compared saved GPU outputs with the published independent NumPy reference functions. These are not organizer-hidden problems; see [development_evidence_r01.json](development_evidence_r01.json).

On October26, R03 fixed a separately reproduced 64-bit shuffle-width defect: the old compiler's f64 reduction was rejected by ptxas. The repair transports the two 32-bit words without numeric conversion and recognizes the canonical `int64_t` shared-memory spelling. **Five fixed local cases passed native execution**: three f64 row reductions (maximum absolute error1.07e-14) and two exact i64 cases. The original56 open/metamorphic/shape PTX files remain byte-identical; their October22 native results were reused, not rerun. See [development_evidence_r03.json](development_evidence_r03.json), including the preserved observation-client failure and lossless-range audit. R02's unrelated unknown compile request remains closed and was not replayed or promoted. Hidden-grade impact is unknown.

On October29, R05 reproduced a distinct predicate-valued ternary defect: the prior compiler emitted `selp.pred`, which the organizer's ptxas rejected. The repair uses four predicate logic instructions only when both selected values are predicates; numeric selection is unchanged. **Four fixed native cases passed all512 exact f32 outputs**, covering eight-state Boolean selection, sixteen-state nested selection, conditional writes and selection of an integer value before an f32 store. Twenty local tests passed and all61 previously verified PTX files stayed byte-identical, without historical GPU reruns. See [development_evidence_r05.json](development_evidence_r05.json). R04's separate incomplete shift-count candidate and unknown run are not incorporated. These remain development results, not hidden-set scores, rank or awards.

On October29, R06 fixed the independently reproduced typed-materialization defect: a valid `Let(F32, comparison)` generated an unsupported `cvt.f32.pred`. Numeric/predicate conversion now uses `selp` and `setp`, and a Boolean immediate is materialized as a predicate register. Floating-point truth uses an unordered nonzero comparison: NaN is true and both signed zeros are false. **Four fixed native cases passed all512 exact f32 outputs**, combining both conversion directions across five supported numeric IR types, internal f32/f64 NaN/Inf/zero truth, and Boolean literals/aliases. All32 local tests passed; all65 prior PTX files stayed byte-identical without historical GPU reruns. See [development_evidence_r06.json](development_evidence_r06.json). This is finite development coverage, not a new hidden grade, rank, prize, or general exceptional-value proof.

On October29, R07 reproduced a separate nested-comparison defect: equality of two comparison results emitted invalid `setp.eq.pred`. Non-logical binary operations now promote a predicate left operand to numeric data; Boolean pairs and Boolean shift data become int32, while a numeric peer supplies the other arithmetic type. Predicate logical operations and numeric-left emission are unchanged. **Four fixed native cases passed all512 exact f32 outputs**, covering six pair comparisons, Boolean arithmetic/bitwise operations with safe nonzero divisors, five scalar numeric peer types, and valid predicate-data shift contexts. All44 local tests passed;160 numeric-only emission rows and69 prior PTX files stayed unchanged, without historical GPU reruns. See [development_evidence_r07.json](development_evidence_r07.json). The left-shift native context checks an integer round trip rather than a separate raw shifted-output buffer; no general shift/type-system or hidden-grade claim is made.

## Run

Python 3.12+ is required; the recorded environment used Python 3.13.11.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-deps -r requirements.txt
git clone --depth 1 https://github.com/cloudrift-ai/emmy-hackathon-2026 ../upstream
.venv/bin/python compile.py ../upstream/problems/r01_array_sum/k_sum_1_reduce.ir.json > out.ptx
.venv/bin/python test_lowering.py
.venv/bin/python test_wide_shuffle.py
.venv/bin/python -m unittest test_predicate_select
.venv/bin/python -m unittest test_materialization
.venv/bin/python -m unittest test_predicate_binary
```

The minimal dependency set intentionally supports IR deserialization, not every feature of the Emmy application. No Torch, CUDA toolkit, or local GPU is required to generate PTX. For optional real-GPU checks, install `requests` and use the official [submit.py](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/main/submit.py) with this compiler. Read the organizer's API limits first; do not flood the shared runner.

```sh
.venv/bin/python -m pip install requests
.venv/bin/python ../upstream/submit.py r01_array_sum --compiler ./compile.py
```

`compile.py` accepts exactly one IR path and prints PTX to stdout. `meta.json` in the IR's directory supplies the kernel's `arg_order` and dynamic shared-memory limit. Neither reference CUDA files nor expected-output files are read by the compiler.

To reproduce the extra shape fixtures locally, with the official task checkout beside this repository as above:

```sh
.venv/bin/python tools/make_shape_cases.py
.venv/bin/python compile.py tools/shape_cases/row_sum_3_33/r01_row_sum_3_33.ir.json > out.ptx
```

The generator does not make network requests or change the upstream problem set. It refuses to overwrite an existing fixture directory. Its NumPy oracles and finite shape design are test infrastructure, never part of the submitted compiler. The official checker also accepts a generated problem-folder path if a further native check is actually needed.

The fixed wide-value cases can likewise be generated offline:

```sh
.venv/bin/python tools/make_wide_cases.py
.venv/bin/python compile.py tools/wide_cases/f64_7_257/k_r03_f64_7_257.ir.json > out.ptx
```

Their oracles are in `expected.json`, using `math.fsum` or exact Python integer sums. The recorded runner serializes i64 values as JSON floating-point numbers; the recorded integer outputs are integral and strictly inside2^53, where the conversion is lossless. Larger i64 JSON results must not be treated as exact without an appropriate output interface.

The four predicate cases are independently reproducible offline, including their Boolean oracles:

```sh
.venv/bin/python tools/make_predicate_cases.py
.venv/bin/python compile.py tools/predicate_cases/predicate_truth8_f32/k_r05_predicate_truth8_f32.ir.json > out.ptx
```

The generator refuses an existing output directory and makes no network or GPU calls. Its generated inputs, oracles and compiler PTX were bound to the recorded fixed cases; changing the test generator does not change the submitted compiler.

The four materialization cases are also generated offline:

```sh
.venv/bin/python tools/make_materialization_cases.py
.venv/bin/python compile.py tools/materialization_cases/pred_to_numeric_5types/k_r06_pred_to_numeric_5types.ir.json > out.ptx
```

The generator refuses existing case directories and makes no network/GPU calls. Its inputs and independent Boolean oracles reproduce the fixed native fixtures. The public tests generate temporary fixtures and check36 frozen numeric-only conversions from the previous published compiler; they do not call that compiler as a backend.

The four predicate-operand cases can be generated without network or GPU calls:

```sh
.venv/bin/python tools/make_predicate_binary_cases.py
.venv/bin/python compile.py tools/predicate_binary_cases/predicate_numeric_peers/k_r07_predicate_numeric_peers.ir.json > out.ptx
```

The generator refuses existing output directories. Its exact scalar oracles and temporary-fixture tests are validation infrastructure, not compiler inputs or a translation backend. `test_predicate_binary.py` also preserves160 frozen numeric-only emission rows from the preceding published compiler; emission identity alone does not validate pre-existing numeric-only limitations.

## Design

1. Deserialize with the required `Graph.from_dict` API and bind the ABI from launch metadata.
2. Collect shared-memory declarations and allocate typed virtual registers.
3. Decode each `Tile` from the actual thread/block geometry; emit the Tile-size bounds guard and grouped raster mapping.
4. Recursively lower expressions and statements, including loop-scoped reduction seeding, coupled softmax states, warp shuffles, shared-memory trees/barriers, asynchronous copies, and atomic writes.
5. Emit PTX8.0 targeting `sm_89`; the organizer assembles and runs it on `sm_120`.

See [WALKTHROUGH.md](WALKTHROUGH.md) for the invariants and failure analysis.

## Boundaries

The broad verified campaigns are f32/static-shape; R03 adds only the five declared f64/i64 reduction checks, R05 the four predicate-selection contexts, R06 the four scalar-materialization contexts, and R07 the four predicate-operand contexts, not general dtype/operator support. R06/R07 cover f32/f64/s32/s64/u32 scalar IR and locally check internal u64 registers; they do not add a U64 graph dtype or Boolean memory operations. R06's nonfinite truth test is not blanket IEEE exceptional-value correctness for all arithmetic and comparison operators. R07 does not repair general mixed-numeric promotions or incorporate the sealed R04 numeric-64-bit-data shift candidate. The implementation rejects unsupported IR rather than falling back to an existing compiler. Non-NONE shared-memory swizzles, auxiliary producer bands, serial launch parameters, tensor-core/TMA primitives, and arbitrary dtype/operator combinations are not claimed. All additional checks are finite coverage, not an exhaustive sweep or proof of hidden-set generalization.

This project was developed with AI coding assistance, which the published event rules permit. The participant provided a personally narrated video with the October24 project entry; that is not a claim of judging, prize eligibility or an award. AI assistance does not replace the participant's required understanding.

## Attribution

Emmy and the open problem set are by CloudRift/RiftStack and their contributors; they are not original work of this repository. The original contribution here is the direct IR-to-PTX visitor and its validation. Upstream problem assets and reference kernels are not vendored. Consult the [task contract](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/main/TASK.md) and [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) for the underlying interfaces.
