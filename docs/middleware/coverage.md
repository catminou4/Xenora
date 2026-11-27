# Campaign E middleware coverage inventory

Reviewed against `main` at `8756f038`. This is a coordination and coverage baseline, not a claim that any middleware API is already compatible.

Evidence labels use the project evidence model: local synthetic verification is lower than independent acceptance, and neither is Xbox hardware verification. Xenora has no Xbox hardware access. “Publicly documented” means the interface documentation is publicly readable; it does not imply that a console uses the same contract.

## Current runtime baseline

The regenerated live inventory reports 862 canonical services in 21 libraries at this revision. `ws2_32.dll` remains the only library classified as `windows-networking`: 41 named services are registered, six implemented and 35 explicitly partial. The six implemented entries are byte-order conversion and error accessors (`htonl`, `htons`, `ntohl`, `ntohs`, `WSAGetLastError` and `WSASetLastError`). Socket creation, datagrams, streams, name resolution, `select` and event integration exist only as a process-local loopback Winsock surface and remain partial.

`winhttp.dll` is now registered with eleven services, all explicitly partial: the handle model (`WinHttpOpen`, `WinHttpConnect`, `WinHttpOpenRequest`, `WinHttpCloseHandle`), synchronous send/receive with a blocking TCP transport, the four DWORD timeout options through `WinHttpSetTimeouts`/`WinHttpQueryOption`, and header and data accessors. There is still no `wininet.dll` and no libcurl-compatible service module. `xgameruntime.dll` covers the `XTaskQueue*` family and one `XNetworkingGetConnectivityHint` registration. `getservbyname` can resolve names such as `http` and `https` from the standard service table, but that is service-name resolution, not an HTTP client.

## Status definitions used here

| Status | Meaning |
| --- | --- |
| implemented | A service or host component exists and has a defined behavior in Xenora. This does not by itself mean verified middleware compatibility. |
| verified locally | A reproducible local test exercised the behavior on native ARM64 macOS and passed. Independent review is still separate. |
| partial | Some behavior exists but important documented semantics, error cases or integration points are absent. |
| unsupported | Xenora intentionally does not provide the behavior, or no authorized/public contract is available. |
| research | Public material exists, but no implementation or verified fixture exists yet. |

## HTTP and networking

| Capability | Public contract source | Current state | Verification | Next independently testable increment |
| --- | --- | --- | --- | --- |
| Winsock loopback sockets | Microsoft Winsock documentation | partial | Locally tested loopback behavior exists from the merged runtime work | Preserve, but do not duplicate while runtime PRs are active |
| Service-name resolution | standard Internet service assignments and Winsock documentation | partial | `getservbyname` local unit coverage exists | Keep separate from HTTP protocol handling |
| `XNetworkingGetConnectivityHint` | public GDK XNetworking documentation | implemented | registered on `xgameruntime.dll` with local test evidence | The wider XNetworking family beyond this one export remains unsupported |
| XNetworking security/certificate URL queries | public GDK XNetworking documentation | unsupported | none | Document as requiring platform security configuration; do not invent Xbox behavior |
| WinHTTP session/connect/request handles | Microsoft WinHTTP documentation | partial | handle lifetime and validation merged; local tests only | Independent acceptance and wider negative controls |
| WinHTTP synchronous request execution | Microsoft WinHTTP documentation | partial | blocking TCP transport merged with silent-server timeout coverage on Apple Silicon | Controlled loopback fixtures for response shapes beyond the timeout path |
| WinHTTP asynchronous callbacks | Microsoft WinHTTP documentation | unsupported | none | Requires audited guest callback dispatch and memory ownership rules |
| Request/response headers | Microsoft WinHTTP documentation | unsupported | none | Add positive, negative and malformed-header controls |
| Connection reuse, cancellation and timeouts | Microsoft WinHTTP documentation | partial | the four DWORD timeout options are validated, inherited and applied to the blocking transport | Connection reuse and cancellation remain open: thread-safe cache and cancellation state machine |
| TLS certificate validation | platform TLS documentation and WinHTTP flags | unsupported | none | Use a licensed TLS implementation or native facility; never disable validation silently |
| libcurl-compatible easy interface | public curl documentation and curl license | unsupported | none | Start with a narrow documented subset; do not copy curl implementation code |
| WinINet | Microsoft WinINet documentation | unsupported | none | Legacy surface; inventory only unless compatibility evidence justifies it |

Public GDK networking documentation directs titles toward xCurl or WinHTTP for HTTP, libHttpClient or WinHTTP for WebSocket use, Winsock for sockets and XNetworking for connectivity hints. That public documentation does not disclose a proprietary console HTTP ABI. The first implementation target should therefore be the documented WinHTTP handle model plus a narrow libcurl-compatible compatibility layer, with XNetworking connectivity state as an independent small increment.

The existing Winsock implementation is private to its service module. A higher-level HTTP implementation must not copy it. Either an explicit internal transport seam has to be added on current `main`, or HTTP transport has to use host networking independently with the ownership boundary documented. That choice affects active runtime pull requests and must be coordinated before implementation.

## AMD FidelityFX and Metal equivalence research

The public AMD FSR/FidelityFX SDK documentation describes open-source DirectX 12 and Vulkan samples and shader-based techniques. The repository-level statement says samples are available under MIT, but the exact `docs/license.md` is more restrictive at the top and then lists source files subject to MIT. Consequently, no file should be copied or adapted until that exact file appears in the MIT-licensed list and its upstream revision is recorded. The presence of Vulkan sample code does not authorize or justify a Vulkan runtime in Xenora.

| Capability | Current state | Verification | Boundary |
| --- | --- | --- | --- |
| Resource lifecycle equivalence | research | none | Must be designed against Xenora's native Metal resource ownership, not copied from the SDK framework |
| Compute dispatch equivalence | research | none | Existing native Metal compute infrastructure can be reused only after graphics checkpoint conflicts are resolved |
| Spatial upscaling algorithm research | research | none | Requires original workload and independent expected-output generation |
| Temporal upscaling/history research | research | none | Requires motion vectors, depth, exposure and history invalidation semantics |
| Shader translation integration | research | none | RDNA2 and shader-registry work is active elsewhere; middleware work must consume interfaces rather than edit those campaigns |
| FidelityFX proprietary or binary-only surfaces | unsupported | none | No reverse engineering, disassembly or undocumented console-specific interface |

MetalFX temporal upscaling and frame interpolation are related but are not AMD FidelityFX. Issue #658 is a research offer involving MetalFX lessons; it is not implementation evidence and does not establish AMD FidelityFX compatibility.

## Metal acceleration structures

Apple publicly documents `MTLAccelerationStructure`, primitive and instance acceleration-structure descriptors, triangle and bounding-box geometry descriptors, build commands and usage flags. Xenora now has a narrow host-side probe that performs runtime ray-tracing capability detection and builds one original single-triangle primitive acceleration structure through the public classic Metal APIs; there is still no guest ABI and no intersection, rendering or refit behavior.

| Capability | Current state | Verification | Boundary |
| --- | --- | --- | --- |
| Bottom-level triangle geometry | partial | single-triangle host probe passes repeated construction locally on Apple Silicon | Must use runtime capability detection across Apple GPU generations; no intersection testing yet |
| Bottom-level bounding-box geometry | research | none | Same as above |
| Top-level instance structures | research | none | Requires carefully defined lifetime and synchronization semantics |
| Build/refit operations | research | none | Refit usage flags and mutation rules must come from Apple documentation |
| Deterministic CPU intersection oracle | research | none | Independently authored triangles/rays should generate expected intersections without using Metal as its own oracle |
| Console-specific acceleration-structure ABI | unsupported | none | No public console ABI is available; none should be invented |

This work collides with active graphics and Metal pull requests. Until the large graphics checkpoint is split or its ownership is clarified, implementation should start as a narrow additive probe or design document rather than edits to the shared renderer.

## Video middleware foundation

Apple publicly documents AVFoundation and VideoToolbox. VideoToolbox provides low-level compression and decompression sessions; AVFoundation provides higher-level asset playback and timing. Xenora now has a narrow native playback foundation in the Swift application layer - AVFoundation-backed single-track decoding with explicit lifecycle states, serialized seek, stable end-of-stream behavior and independently generated H.264 fixture tests - plus a bounded copy-based upload layer that moves decoded BGRA frames into native shader-read Metal textures. There is no guest-visible video ABI, no audio synchronization, no real-time scheduling and no Bink implementation.

| Capability | Current state | Verification | Boundary |
| --- | --- | --- | --- |
| Playback lifecycle/state machine | partial | explicit lifecycle states tested in native Swift fixture tests | Host-layer only; no guest ABI exists |
| Frame timing | partial | timestamp and frame-duration reporting covered by fixture tests | Real-time scheduling remains open |
| Texture upload | partial | copy-based BGRA upload verified byte-exact on native ARM64 tests | Zero-copy import and presentation remain open |
| Audio synchronization | unsupported | none | Adjacent to the open native PCM audio research issue; do not duplicate it |
| Seeking | partial | serialized seek and reader replacement covered by fixture tests | Exact state transitions under concurrent load remain open |
| Resource cleanup | partial | terminal cleanup semantics covered by fixture tests | Repeated create/destroy cycling evidence still thin |
| Openly licensed fixture media | research | none | Fixture media must be generated independently or carry an explicit compatible license |
| Bink video | unsupported | none | No authorized Bink implementation or public console-specific contract is available |

Video playback must not be represented as game video compatibility. A successful local fixture demonstrates only the abstraction, codec path, timing and resource behavior for that fixture.

## Cross-cutting validation requirements

Every implementation slice should add:

- a coverage inventory update with implemented, locally verified, partial and unsupported rows separated;
- positive and negative controls;
- ABI conformance tests for guest-visible layouts and calling behavior;
- multithreaded execution tests where concurrency is part of the contract;
- memory-ownership and resource-cleanup tests;
- native ARM64 macOS integration evidence;
- reproducible commands and raw logs under ignored `artifacts/`.

A placeholder, honest refusal, import registration or status-table entry is not a verified implementation.

## Public sources consulted for this baseline

- Microsoft Learn WinHTTP documentation: `WinHttpOpen`, `WinHttpConnect`, `WinHttpOpenRequest` and the `winhttp.h` interface inventory.
- Microsoft Learn WinINet documentation: HTTP sessions, `InternetOpen`, `InternetConnect` and `HttpOpenRequest`.
- Microsoft GDK networking overview and XNetworking API contents, including the documented roles of WinHTTP, xCurl, libHttpClient, Winsock and XNetworking.
- curl project `COPYING` at public commit `d5ec024cf205997c6a3416b2998953ce7cf23fcb`: copyright-and-permission notice permitting use, copy, modification and distribution with notice retention.
- AMD FSR/FidelityFX SDK public repository, `readme.md`, `docs/license.md` and `3rdpartynotice.md` at public commit `7539ae8b6568ad29ed78a753404351daf3b5cca7`.
- Apple Metal documentation for `MTLAccelerationStructure` and acceleration-structure descriptors.
- Apple AVFoundation and VideoToolbox documentation for playback, decompression sessions, hardware decoder access and pixel-buffer output.
