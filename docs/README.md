# Xenora documentation

## Getting started

Start with the [project README](../README.md) for scope and build commands,
then read the [interface contracts](interfaces.md). Reproduce an
[original guest test program](../experiments/x1/README.md) or run the
[independent validation suite](../tests/validation/README.md) before changing
either path. The [changelog](../CHANGELOG.md) records notable changes.

## Architecture and technical design

- [Interface contracts](interfaces.md) — Translated shader stages, guest services, native command submission, and synthetic textures.
- [Fixture memory model](adr/0003-demo-memory-model.md) and [CPU boundary](adr/0004-cpu-boundary.md) — Buffer bindings, guest address translation, and dependency direction.

## Native Metal graphics

- [Translated offscreen stages](interfaces.md#translated-offscreen-stages) — Translated vertex and pixel stages for the bounded gfx1030 subset.
- [Blend and depth passes](interfaces.md) — Native blend and depth graphics passes with per-draw state.
- [Apple GPU evidence](research/apple-gpu.md) and [host lab record](research/host-lab.md) — Observed behavior and historical floating-point results.
- [Metal acceleration-structure boundaries](research/metal-acceleration-structures.md) — Public build/refit/copy lifetime, scratch ownership, synchronization and deterministic fixture requirements.

## RDNA 2 decoding and shader translation

- [Instruction coverage](shaders/instruction-coverage.md) — Desktop gfx1030 instruction forms, operand limits, and CPU and GPU results.
- [Semantic coverage](shaders/semantic-coverage.md) — Generated per-family counts of identified, declared, decoded, lowered and verified gfx1030 forms.
- [gfx1030 semantic campaign](coordination/rdna2-campaign.md) — P0 family backlog, ownership, and verification checkpoints.
- [MIMG explicit-LOD 2D sampling](shaders/mimg.md) — `IMAGE_SAMPLE_LZ` and `IMAGE_SAMPLE_L`, their T#/S# contract, refusals and Metal binding.
- [SOP1 EXEC-writing forms](shaders/sop1-saveexec.md) — The twelve Wave32 forms that read, combine and rewrite the execution mask.
- [SOP2 scalar ALU](shaders/scalar-alu.md) — Supported scalar add, subtract and bitwise forms, their SCC rules and sources.
- [Semantic registry](shaders/semantic-registry.md) — How to add an instruction form: registry fields, validation, generation and the independence rule.
- [Scalar buffer loads](shaders/scalar-memory.md) — SMEM loads, their bounds and wait rules, and where a loaded SGPR may be used.
- [Run-time uniform branches](shaders/runtime-branches.md) — `s_cbranch_scc0` on a loaded value, if/else regions and their limits.
- [VOP1 vector unary and swap](shaders/vector-unary.md) — Wave32 integer bit operations, register exchange, and operand limits.
- [VOP2 binary32 arithmetic](shaders/vector-binary-f32.md) — Wave32 binary32 add, subtract both ways, multiply, and the guide's minimum and maximum with their NaN and signed-zero rules.
- [VOP2 select and carry](shaders/vector-vcc.md) — Wave32 lane selection, carry and borrow through the vector condition code.
- [VOPC integer compares](shaders/vector-compare.md) — Wave32 compare forms, the VCC/EXEC contract, runtime `s_cbranch_vccz` and the Metal mapping.
- [Metal Shading Language generation](shaders/msl.md) and [independent CPU reference](shaders/reference-interpreter.md) — Translation rules and separate CPU execution models.

## Guest CPU and runtime experiments

- [Guest experiment guide](../experiments/x1/README.md) — Bounded x86-64 execution, executable loading, experimental services, native compute, and offscreen rendering.
- [x86-64 instruction registry](cpu/instruction-registry.md) — Shared decoder, generated tables, semantics, native lowering and how to add a verified instruction form.
- [Runtime service registry](runtime/registry.md), [typed service coverage](runtime/services.md) and the [live inventory](runtime/inventory.md) — Import resolution, typed Win64 adapters, manual registrations, aliases and status metadata.
- [Guest threads](runtime/threads.md) — Deterministic scheduler, thread services, limits and tests.
- [Guest graphics command path](runtime/guest-graphics.md) — The guest-callable graphics ABI, ordered resource destruction, and the path from a guest program to a rendered framebuffer.
- [Guest CPU research](research/cpu-boundary.md) — Public specifications and unresolved console requirements.
- [Synthetic import fixtures](../tests/compat_fixtures/README.md) — Reproducible PE32+ builds and expected service behavior.

## Testing, validation, and benchmarks

- [Testing and acceptance index](testing/README.md) — Entry point for test procedures, evidence and review records.
- [AMD RDNA2 desktop GPU (gfx1030) form acceptance](testing/shader-form-acceptance.md) — Independence, command, log and promotion rules for fixture-verified instruction forms.
- [Engineering progress](progress.md) — Verification commands, metric definitions, and open release gates.
- [CPU test runtime and shard selection](testing/cpu-coverage.md) — Measured CPU gate cost and deterministic test-target shards.
- [Test inventory comparison](testing/test-inventory.md) — Detect missing or newly ignored Cargo tests across commits.
- [Persistent resource and translated rendering acceptance](testing/x9-acceptance.md) — Pinned upstream tests, native ARM64 results and unsupported cases.
- [Baseline acceptance](testing/phase-02-baseline.md), [Phase 01 assurance](testing/phase01-assurance.md), and [independent reviews](testing/review-log.md) — Candidate-specific results and release findings.
- [Structured startup reports](testing/startup-reports.md) — Interpret the CLI worker’s live startup stages and termination details.
- [Gate A: a compiled C program](testing/gate-a.md) — The two milestones, what the A1 and A2 tests check and what they do not show.
- [Mutation testing](testing/mutation.md) and [benchmark protocol](testing/benchmarks.md) — Test sensitivity, benchmark inputs, and timing requirements.

## Research and engineering decisions

- [Public sources](research/sources.md), [RDNA 2 notes](research/rdna2.md), and [Xbox GPU evidence](research/xbox-gpu.md) — Specifications, evidence scope, and unknown console behavior.
- [RDNA 2 guide errata](research/rdna2-errata.md) — The defects, ambiguities and undefined helpers in AMD's RDNA 2 guide, and the reading Xenora adopts for each.
- [Shader inputs](research/shader-containers.md) and [risk register](risk-register.md) — Input provenance and unresolved engineering risks.
- [Texture format inventory](research/texture-formats.md) — Formats and layouts by stage (decode, layout, native upload, sampling, guest), their public specifications, unknowns and ranked follow-ups.
- [Xbox Series X|S executable startup](research/xbox-launch.md) — Experimental manifest preflight, PE32+ startup audit, and console research blockers.
- [Exception and unwind requirements](research/exception-handling.md) — The Windows x64 exception and unwind ABI mapped to missing guest runtime pieces, measured probe results, unknowns, a fixture plan and ranked follow-ups.
- [Synthetic integration and upstream evidence](research/synthetic-integration.md) — Audited source snapshots and remaining commercial startup prerequisites.

## Security, licensing, and contributions

- [Threat model](threat-model.md) and [security reporting](../SECURITY.md) — Execution limits and disclosure instructions.
- [Legal scope](legal.md), [dependency licensing](licenses.md), and [third-party notices](../THIRD_PARTY_NOTICES.md) — Input policy, attribution, and redistribution findings.
- [Contribution guidelines](../CONTRIBUTING.md) — Original authorship, Developer Certificate of Origin sign-off, and evidence requirements.
- [Agent task queue](coordination/agent-tasks.md) — Areas, claiming, leases, upstream sync, verification and the security model for parallel contributors and coding agents.
- [CPU/runtime expansion campaign ledger](coordination/cpu-campaign-ledger.md) — Progress ledger for the Xenora native CPU and runtime compatibility campaign: starting/current commits, completed batches, new forms, verification evidence and the next implementation target.
- [Infrastructure QA ownership](coordination/infrastructure-qa-ownership.md) — Non-overlapping paths for infrastructure and QA tooling.
- [Agent queue smoke evidence](coordination/agent-queue-smoke.md) — Live Actions results from the disposable smoke repository.
- [Infrastructure QA tooling](testing/infrastructure/README.md) — Advisory orchestration, PR validation, benchmark analysis and release checks.
