# Native Metal acceleration-structure compatibility boundaries

This note records a bounded reading of the public Metal SDK headers installed
with Xcode on the development host. It is research for a future independent
compatibility layer, not an implementation report. It does not use or infer any
proprietary console-specific ray-tracing ABI.

## Public Metal model

The classic Metal API exposes acceleration structures as device-created opaque
resources. `MTLDevice` reports `MTLAccelerationStructureSizes` for a descriptor
through `accelerationStructureSizesWithDescriptor:`. That query determines both
the acceleration-structure allocation size and the scratch storage needed by a
build. A build is encoded into an `MTLAccelerationStructureCommandEncoder` with
`buildAccelerationStructure:descriptor:scratchBuffer:scratchBufferOffset:`.

The public descriptor hierarchy separates:

- primitive acceleration structure descriptors, which contain geometry
  descriptors;
- instance acceleration structure descriptors, which reference an instance
  descriptor buffer, its offset/stride/count, the instance descriptor type and
  the primitive acceleration structures being instanced;
- geometry descriptors, including triangle and other public geometry forms,
  with opacity, intersection-function and primitive-data-related properties.

This maps naturally to the generic concepts of bottom-level and top-level
acceleration structures, but Xenora should use the Metal names in host-facing
documentation rather than pretending that the console contract is already
known.

## Lifetime, sizing and scratch ownership

A correct compatibility layer must treat sizing as transactional:

1. construct the complete descriptor;
2. query sizes on the exact device that will perform the build;
3. allocate the acceleration structure and scratch storage for at least the
   returned sizes;
4. encode exactly one build for the new output;
5. keep descriptor-referenced vertex, index, instance and primitive-data
   buffers alive until the build completes;
6. publish the built acceleration structure only after command-buffer
   completion or an explicitly ordered dependency.

Scratch memory is build machinery, not a long-term semantic input. It may be
reused only when no overlapping encoded build/refit can still read it. Since
sizes are device- and descriptor-dependent, hard-coded sizes are invalid.

## Build, refit, copy and compaction

The public command encoder exposes build, refit, copy, compacted-size query and
copy-and-compact operations. Refit is not merely a smaller build: usage must
permit it, the descriptor topology must remain compatible, and the documented
refit options control whether vertex or per-primitive data is refreshed. The
headers also state that refitting can reduce ray-tracing quality compared with
a full rebuild. A future implementation therefore needs separate positive
controls for full rebuild and refit rather than treating them as equivalent.

Copy and compaction introduce a second lifetime boundary: the compacted size is
produced asynchronously into a buffer, so allocation of the compacted
destination must be ordered after reading that size or must use a conservatively
validated allocation strategy. Silent truncation is not acceptable.

## Synchronization and capability detection

Metal acceleration-structure commands are encoder work. The implementation
must prove ordering between geometry production, acceleration-structure build,
consumption by compute/render shaders, refit and destruction. The presence of
API availability annotations is not proof that a particular Apple GPU supports
the desired feature combination; runtime device capability checks remain
required. The newer Metal 4 compute-encoder API also exposes build/refit/copy
operations and residency-set obligations, but Xenora should not mix classic and
Metal 4 ownership models in one abstraction without an explicit boundary.

## Required independent fixtures

Before implementation claims are made, Xenora needs original deterministic
fixtures covering:

- one-triangle and small-mesh primitive builds;
- multiple geometry entries with different opacity and culling options;
- instance transforms and instance descriptor stride/offset validation;
- full rebuild versus vertex-data refit with different expected intersections;
- copy and compacted-copy lifetime ordering;
- use-after-build, premature-release and overlapping-scratch negative controls;
- deterministic ray/triangle intersection results generated independently on
  CPU, with Metal execution treated as the implementation under test.

A passing fixture can establish build ordering, lifetime and deterministic
intersection behavior for the tested case. It cannot establish proprietary
console compatibility, general hardware support or performance.

## Unsupported or unresolved

No console-specific bottom-level/top-level ABI, memory layout, shader binding
or ray-tracing instruction contract is claimed here. No implementation exists
in this branch. Visual quality, performance, motion geometry behavior and
cross-generation Apple GPU support remain unverified. Active renderer, shader
and RDNA implementation files are intentionally untouched.

## Sources

- `MTLAccelerationStructure.h` in the public macOS Metal SDK installed with Xcode;
- `MTLAccelerationStructureCommandEncoder.h` in the same SDK;
- `MTLDevice.h`, `MTLHeap.h`, `MTLComputeCommandEncoder.h` and
  `MTL4ComputeCommandEncoder.h` in the same SDK.
