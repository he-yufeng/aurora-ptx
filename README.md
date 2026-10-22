# Aurora PTX

An independent Python compiler from Emmy's typed Kernel IR to NVIDIA PTX, developed for the [Emmy Hackathon 2026](https://riftstack.ai/hackathon). It walks the scheduled IR directly; it does **not** call Emmy's CUDA renderer, `emmy compile`, `nvcc`, or another compiler.

## Current evidence

On 2026-10-22, the same compiler source passed all **15/15 official open problems** on the organizer's RTX 5090 runner, covering pointwise, reductions, matmul, fused normalization/softmax, and attention-like kernels. It additionally passed **21/21 local metamorphic cases**: all15 with renamed kernel/buffer/SSA/axis identities and a reversed argument ABI, plus six different static pointwise sizes.

These are development correctness results, **not a hidden-set score, ranking, prize, or completed contest submission**. See [development_evidence.json](development_evidence.json) for source binding, per-case errors, and the initial failed full attempt. No kernel-speed superiority is claimed.

## Run

Python 3.12+ is required; the recorded environment used Python 3.13.11.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-deps -r requirements.txt
git clone --depth 1 https://github.com/cloudrift-ai/emmy-hackathon-2026 ../upstream
.venv/bin/python compile.py ../upstream/problems/r01_array_sum/k_sum_1_reduce.ir.json > out.ptx
.venv/bin/python test_lowering.py
```

The minimal dependency set intentionally supports IR deserialization, not every feature of the Emmy application. No Torch, CUDA toolkit, or local GPU is required to generate PTX. For optional real-GPU checks, install `requests` and use the official [submit.py](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/main/submit.py) with this compiler. Read the organizer's API limits first; do not flood the shared runner.

```sh
.venv/bin/python -m pip install requests
.venv/bin/python ../upstream/submit.py r01_array_sum --compiler ./compile.py
```

`compile.py` accepts exactly one IR path and prints PTX to stdout. `meta.json` in the IR's directory supplies the kernel's `arg_order` and dynamic shared-memory limit. Neither reference CUDA files nor expected-output files are read by the compiler.

## Design

1. Deserialize with the required `Graph.from_dict` API and bind the ABI from launch metadata.
2. Collect shared-memory declarations and allocate typed virtual registers.
3. Decode each `Tile` from the actual thread/block geometry; emit the Tile-size bounds guard and grouped raster mapping.
4. Recursively lower expressions and statements, including loop-scoped reduction seeding, coupled softmax states, warp shuffles, shared-memory trees/barriers, asynchronous copies, and atomic writes.
5. Emit PTX8.0 targeting `sm_89`; the organizer assembles and runs it on `sm_120`.

See [WALKTHROUGH.md](WALKTHROUGH.md) for the invariants and failure analysis.

## Boundaries

The verified campaign is f32/static-shape. The implementation rejects unsupported IR rather than falling back to an existing compiler. Non-NONE shared-memory swizzles, auxiliary producer bands, serial launch parameters, tensor-core/TMA primitives, and arbitrary dtype/operator combinations are not claimed. Local shape checks currently cover pointwise geometry; the other categories have open-set plus naming/ABI checks, not an exhaustive shape sweep.

This project was developed with AI coding assistance, which the published event rules permit. The participant must independently understand the compiler and personally narrate the required video before submitting; this repository is not a claim that those steps have happened.

## Attribution

Emmy and the open problem set are by CloudRift/RiftStack and their contributors; they are not original work of this repository. The original contribution here is the direct IR-to-PTX visitor and its validation. Upstream problem assets and reference kernels are not vendored. Consult the [task contract](https://github.com/cloudrift-ai/emmy-hackathon-2026/blob/main/TASK.md) and [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) for the underlying interfaces.
