# Compiler walkthrough and evidence boundaries

## 1. Separate the ABI from the computation

`compile_file` deserializes the typed graph, locates its single `KernelOp`, and selects the matching `meta.json` entry. Arguments are declared in `arg_order`, not inferred from alphabetic names or graph input order. The body is never parsed as a string. Reversing all15 ABIs while renaming buffers and values gave the same numerical results.

## 2. A Tile is a coordinate map, not a problem identifier

A linear thread index is computed from block id, block width and thread id. Dividing by products of inner extents decodes each logical axis. A guard excludes excess physical threads: the small pointwise problem schedules more threads than Tile cells. Grouped raster mapping changes CTA visitation order without changing the logical matrix coordinates.

The initial six extra shape checks use 1,17,63,129,257 and500 Tile cells with two output elements per cell. They exercise excess threads and multi-block coverage. A separate frozen-source R01 campaign then tested20 new shapes in reductions, fused kernels, pipelined/split-K matmul and attention, including a grouped-raster tail group. All20 passed native checking and an independent numerical audit; these remain bounded local cases, not the organizer's hidden set.

## 3. Distinguish reduction scope from lexical nesting

A loop seeds only accumulators whose declared reduction axes contain that loop's axis. In attention, the inner dot-product accumulator resets for each key, while the outer running maximum, normalization sum, and weighted-value sum persist across keys. Seeding every accumulator at every enclosing loop would destroy those states.

`Accum.base` is meaningful: an online-softmax update can use a rescaled previous partial as its left input, rather than simply adding to the old register. All shuffle partners are read before the coupled combine program runs. Shared-memory reduction steps synchronize before the next stage and broadcast the final partial.

## 4. Preserve memory and launch semantics

Addresses are flattened using graph tensor or shared-array shapes. Global pointers are64-bit; shared addresses are32-bit. Grouped loads/stores are conservatively scalarized. Atomic addition and the launch's zero-output list preserve split-K accumulation.

For asynchronous copies, the visitor emits direct `cp.async`/commit/wait instructions and keeps the IR's CTA barriers. Shared arrays totaling more than48KiB use the dynamic shared allocation already specified by `meta.json`; smaller arrays are static. No launch shape or tolerance was loosened to obtain a pass.

## 5. What failed, and what the results do not prove

The first full15 attempt assembled no kernels: the generator used an integer type suffix for a bitwise instruction and declared an external shared symbol inside the entry. The fixes changed instruction spelling/scope, not input data or correctness criteria. The complete failed attempt and exact source were retained locally.

The repaired source passed the official open15 and local metamorphic21. A separate read-only audit recomputed every numerical comparison from saved HTTP results, checked launch/argument/oracle/source hashes, and verified the upstream checkout stayed unchanged. One positive-control kernel passed first. The campaign made89 compile/run POSTs plus one health GET, serially, with no unknown request outcomes.

Fixture generation itself had three local failures before native admission: statement serialization, incoherent primary buffer names, and an ABI-coverage assertion. Those versions were retained; they are not compiler-quality failures or extra native passes. The production compiler did not change during the successful open and metamorphic checks.

The separate R01 shape campaign required no production-code change:40 compile/run POSTs plus one health GET,20 distinct new shape cases, and zero unknown outcomes. Its oracles were independently checked against the published NumPy reference functions, rather than trusting only the fixture producer. The generator is included under tools/ for reproducibility.

No hidden-set result, official ranking, award, speed benchmark, formal project submission or participant-understanding/video completion is established. Next value comes from careful review and a truthful participant explanation, not repeating the same inputs to enlarge the pass count.
