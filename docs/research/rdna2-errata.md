# RDNA 2 guide errata and the resolutions Xenora adopts

This document records the defects, ambiguities and undefined helpers in AMD's
"RDNA 2" Instruction Set Architecture reference guide (document 70648, revision
30 November 2020), together with the reading Xenora adopts for each one.

The guide is the primary source for every gfx1030 instruction form. It is also
a 283-page document with a number of internal contradictions, typographical
errors and undefined helper functions, and AMD's own specification agreement
says the document "may contain preliminary information, errors, or inaccuracies"
and is provided without warranty. A form that cites a guide page without saying
how that page is defective is a form that a later maintainer can "correct" back
into a wrong state, so the defect is recorded here and referenced from the
registry row that depends on it.

## How to read this document

- Page numbers are the guide's own printed page numbers, not PDF page numbers.
  The PDF page is the printed page plus eight.
- "Defect" means the guide contradicts itself, contradicts another part of the
  guide, or states something that cannot be true.
- "Ambiguity" means the guide does not say enough to derive a behaviour.
- Every row names the resolution Xenora uses and, where one exists, a second
  source that supports it. A second source is a pointer to re-derive from the
  guide, never a substitute for it.
- The registry rows that depend on a row here carry a note in their `spec`
  comment, so the citation points at the decision as well as the page.

## Defects in the guide's own text

| Form or subject | Page | What the guide says | Resolution |
| --- | --- | --- | --- |
| `S_ADDC_U32`, `S_SUBB_U32` | 34 | "D = S0 + S1 + SCC = overflow." The carry-in term is written as if it were the output flag and the punctuation is missing. `S_ADDC_U32` also says "SCC = overflow" where the unsigned forms set carry-out. | Read as "D = S0 + S1 + SCC_in; SCC = carry out" and "D = S0 - S1 - SCC_in; SCC = borrow". The registry declares `carry_in = "scc"` with `scc = "carry_out"` and `"borrow"`. |
| `S_FLBIT_I32` | 36, 115 | The name appears twice in the table, once as "find last bit" and once as the sign-bit-run count that "behaves the same as `V_FFBH_I32`". Opcode 23 carries the second behaviour while the guide prints opcode 21's name against it. | Opcode 23 is the sign-bit-run count. The registry names it `S_FLBIT_I32` with `signedness = "signed"`; the sign-bit-run reading is the one implemented. |
| `S_QUADMASK_B32` | 36 | "D[0] = OR(S0[3:0]), D[1] = OR(S0[7:4])", with `S_BITREPLICATE_B64_B32` described as "the inverse". A quad mask sets all four bits of a quad when any is set, which is `S_WQM`; the printed form is not the inverse of bit replication. | Not implemented. Neither `S_QUADMASK` nor `S_WQM` is in the registry, so no behaviour depends on the printed text. |
| `VCC_LO` | 32, 41 | The scalar operand table says `VCC_LO` holds "the low Dword of the vector condition code (7 bits)". A Dword is 32 bits, and the vector table gives `vcc[31:0]`. | `VCC_LO` is `vcc[31:0]`. "(7 bits)" is a typo. |
| `MODE.EXCP_EN` | 16, 20-21 | The MODE table puts it at bits 20:12 with eight exception bits; section 3.10 says `MODE.EXCP_EN[8:0]` with a 0-based table of eight entries. | Neither reading is implemented. Exception enable is outside the current contract; the conflict is recorded so that a future batch does not pick one silently. |
| `TRAPSTS.mem_viol` | 21-22 | Section 3.11 names a `mem_viol` bit that the status table does not have; memory violation appears only as EXCP bit 8. | Not implemented. |
| `S_SUBVECTOR_LOOP_BEGIN` | 28 | "If EXECHI = 0: the body is executed only once: EXEC_LO is stored in S0 and restored at the end, but it was zero anyway." If EXEC_HI is zero then EXEC_LO is not implied to be zero, and the printed pseudocode branch `D0 = EXEC_LO` cannot be right. | The subvector loop is not implemented. The half-selection rule is not recoverable from the text. |
| `S_SETREG_IMM32_B32` | 106, 108 | Described as "a 64-bit instruction format", which conflicts with the rule that SOPK carries no literal constant. | Not implemented. |
| `S_ABS_I32` | 119 | The pseudocode uses `S.i` where it must be `S0.i`. | Read as `S0.i`. The registry declares `signedness = "signed"`; the implemented behaviour is the absolute value of the sign-extended source. |
| `V_CMP_U_F32` "unordered" | 151, 153 | The summary table defines U as `(!isNaN(S0) \|\| !isNaN(S1))`; the per-opcode table defines `V_CMP_U_F32` as `(isNan(S0) \|\| isNan(S1))`. The two are complements. | The per-opcode table is right: "unordered" is true when either input is a NaN. Not implemented yet; recorded so the VOPC float batch uses the per-opcode table. |
| VOPC summary tables | 151-152, 154-160 | Table 58's F16 and F64 hex ranges and Table 60's integer ranges disagree with the per-opcode tables. Table 60 also labels `V_CMP_{COMPI}_U16` as writing EXEC, and `V_CMPX_{COMPI}_I16` as a signed compare. | The per-opcode tables are authoritative. Only the `X` forms write EXEC. |
| `V_CMP_T_U16`, `V_CMPX_T_U16` | 158-159 | Absent from the VOPC table (175 and 191). | Treated as reserved. |
| VOP3 opcode bases | 137, 150, 162, 166 | Four different rules are given: "VOP2 opcode + 0x100", "VOP2 opcode + 0x180" for VOP1-as-VOP3, "VOP2 opcode + 0x000" for VOPC-as-VOP3A, and "VOP2 opcode + 0x270" for VINTERP-as-VOP3A. | None of these sentences is used. Opcode numbers come from the pinned XML snapshot and the encoding chapter. |
| `V_RSQ_F64` | 142 | The pseudocode reads `D.f16 = 1.0 / sqrt(S0.f16)` for a double-precision instruction. | Read as `D.d = 1.0 / sqrt(S0.d)`. Not implemented yet. |
| `V_LOG_F16` | 147 | The pseudocode reads `D.f16 = log2(S0.f)`; the source is an FP16 operand. | Read as `S0.f16`. Not implemented yet. |
| `V_PK_MIN_F16` | 164 | The high half uses `S1.f[31:16]` but the low half uses `S1.u[15:0]`. | Read as `S1.f[15:0]`. Not implemented yet. |
| `V_CUBETC_F32` | 169 | The first condition reads `abs(S2.f) >= bs(S1.f)`; `bs(` is a typo for `abs(`. | Read as `abs(`. Not implemented yet. |
| `V_CUBESC_F32`, `V_CUBETC_F32` | 168-169 | The face branches do not compute the standard cube-map coordinates: the "Y face" branch returns the x coordinate and the x-face branch returns `-S1.f` where the standard convention gives `-y` and `+y`. `V_CUBEID_F32` and `V_CUBEMA_F32` are self-consistent with the face numbering. | Not implemented. The branch logic, not the face numbering, is what is in doubt. |
| `V_CMP_CLASS_F32` in VOP3A | 156-157, 162 | The VOPC form writes VCC and the X form writes EXEC, but the VOP3A destination field is VDST, a VGPR. The guide never says where the class result goes in the VOP3A encoding. | Not implemented. |
| `V_MQSAD_PK_U16_U8`, `V_QSAD_PK_U16_U8` | 177 | Write a 64-bit destination `D[63:48]…D[15:0]` while the VOP3 destination field is a VGPR. | Not implemented. |
| `V_TRIG_PREOP_F64` | 177 | `shift = S1.u * 53` reaches 1643 and is masked with `0x1fffff_ffffffff`; no range limit is given. | Not implemented. |
| `V_PERMLANE16`, `V_PERMLANEX16` | 186-187 | The comment "interval is 0 … 1 for wave32 mode" is unexplained, and the loops run `for row in 0 … 3` in both wave sizes. | Not implemented. |
| MIMG `DMASK` | 69 | "DMASK[0] = red, DMASK[1] = green, DMASK[2] = blue, DMASK[2] = alpha". | Alpha is `DMASK[3]`. Not implemented yet. |
| image atomics `DMASK` | 74 | `DMASK` 0x3 is listed for two different widths. | Not implemented. |
| resource type field | 65, 75 | V# has a 2-bit type field; T# has 4 bits. | Both come from the pinned XML snapshot, not from the prose. |
| `const_stride` | 60 | A 14-bit field with the range "0 to 16,383 bytes, or 0 to 262,143 bytes". | A 14-bit field reaches 16,383. The second range cannot be reached. |
| `const_num_records` | 60, 65 | Tables 33 and 37 disagree on the unit. | The field description on page 60 is used: bytes when `const_stride == 0` or the swizzle is off, otherwise stride units. |
| `DS_PERMUTE_B32`, `DS_BPERMUTE_B32` lane index | 97, 205 | Section 10.4.4 says only index bits [6:2] are used, which selects one of 32 lanes. The section 12.13 pseudocode takes `floor((ADDR + OFFSET) / 4) mod 64` and loops over lanes 0..63. | Not implemented. A Wave32 batch takes the lane from bits [6:2] of the byte index after adding `OFFSET`, as section 10.4.4 says. It records the 12.13 reading as the Wave64 rule and does not apply it to Wave32. |
| `DS_PERMUTE_B32` colliding writers | 205 | The pseudocode comment says the highest-numbered source lane wins when several lanes select one destination; the prose under it says the result "is not predictable" and is one of the writers. | Not implemented. A batch refuses, or excludes from its oracle, any case where two active lanes select the same destination, until hardware evidence settles which is right. |
| FLAT `OFFSET` | 83, 279 | Section 9.1 describes a 12-bit field and says "Flat: must be positive"; the encoding chapter calls it an "11-bit unsigned offset (MSB is ignored)". One says bit 11 is forbidden, the other that it is ignored. | Not implemented. A FLAT batch decodes bits 10:0 as an unsigned offset and refuses an encoding with bit 11 set instead of ignoring it, because the two readings agree only when bit 11 is clear. Scratch and Global use the 12-bit signed offset both pages give. |
| cube `face_id` | 71-72 | Section 8.2.3 says `face_id = slice * 6 + face`; section 8.2.4 says `face_id = slice * 8 + face`. | A cube array slice holds six faces, so the with-sampler formula is wrong. Not implemented. |
| A16 offset | 70, 73 | The offset is signed in one place and unsigned in the other. | The pinned XML is authoritative. |
| T# base address | 75 | Described as 40 bits "representing bits 47:8". | The pinned XML is authoritative. |
| `_U64` compares | 123 | Written as signed comparisons. | The pinned XML is authoritative for the operand format. |
| compute user SGPRs | 25 | "First 0..16 of User data registers" against "0..32" for the other stages. | Not implemented; the fixture ABI is fixed independently. |
| pagination | 87 | The page footer and the section number disagree. | A transcription artefact of the text extraction, not of the guide. |

## Helpers the guide uses but never defines

These appear in the guide's pseudocode and are not defined anywhere in the 283
pages. Each one blocks the forms that depend on it, so each one has to be
decided explicitly rather than inferred.

| Helper | Used by | What is missing |
| --- | --- | --- |
| `flt32_to_uint8` | `V_CVT_PK_U8_F32`, p215 | Saturation and rounding behaviour. Mesa's errata states it saturates and uses the single-precision rounding mode rather than truncating. |
| `flt32_to_flt16`, `flt16_to_unorm16`, `flt16_to_uint16`, `flt16_to_snorm16`, `flt16_to_int16`, `flt16_to_flt32` | the `V_CVT_*` and `V_CVT_PK_*` forms | Saturation, rounding and NaN handling. |
| `Quiet(x)` | 18 places, including `V_MIN_F32` and `V_MAX_F32` | Whether the quieted payload is preserved. |
| `Round(x)` | 15 places | Which rounding mode; nearest-even is implied by the round-mode table but not stated. |
| `Mantissa(x)`, `Exponent(x)`, `TwosComplement(x)` | `V_FREXP_*`, `V_LDEXP_*` | The exact splitting and rounding of the mantissa. |
| `sat8(x)` | `V_SAT_PK_U8_I16` | Saturation behaviour. |
| `FindFirst1fromLSB(exec)` | `V_READFIRSTLANE_B32`, p137 | Only the definition. The name gives the search order and the text gives the empty-mask result ("Lane# = 0 if exec is zero"), so no reading is open. `V_FFBH_*` and `V_FFBL_B32` do not use it; they carry their own loops and return -1 when no bit is set (p143). |
| `isNaN(x)` | the VOPC float comparisons | Whether a signalling NaN and a quiet NaN are distinguished. |

`IEEE_MODE` is a fourth kind of gap: it is used in the pseudocode but never
defined as a mode. It is the MODE register's IEEE bit, position 9, defined in
the MODE table on page 15 next to `DX10_CLAMP` at position 8.

## Rules that are easy to get wrong

These are stated correctly in the guide and are the ones most likely to be
implemented wrongly.

- **The EXEC mask is 64 bits, but Wave32 uses only the low half.** Page 13:
  "the upper 32-bit of EXEC are ignored" for Wave32, and `EXECZ` reflects only
  the lower 32 bits. A `S_*_SAVEEXEC_B32` therefore updates `EXEC_LO` only and
  must not clear the upper lanes.
- **The mask does not gate scalar instructions.** Page 13: the mask "affects
  vector-ALU, vector-memory, LDS, GDS, and export instructions. It does not
  affect scalar (ALU or memory) execution or branches." An `S_*_SAVEEXEC_B32`
  executes even when the mask is empty.
- **An out-of-range destination still writes EXEC and SCC.** Page 33: "If the
  destination SGPR is out-of-range, no SGPR is written with the result. However,
  SCC and possibly EXEC (if saveexec) is still written." The decoder refuses
  such a destination rather than executing the form with a discarded result,
  and the refusal is recorded as an unsupported operand variant.
- **`V_MIN_F32` and `V_MAX_F32` are not Metal's `fmin` and `fmax`.** Page 132
  gives the rule: under IEEE mode a signalling NaN is quieted; otherwise a
  quiet NaN in either operand yields **the other operand**; and a signed zero
  pair resolves to **negative zero**. Metal's builtins do neither.
- **The scalar and vector operand tables differ.** Page 32 reserves 209-234 and
  249-250 for scalar sources; page 41 reserves 209-232 and repurposes 233, 234,
  249 and 250 as DPP8, DPP8FI, SDWA and DPP16. A shared source-class table
  would mis-decode.
- **MODE holds two round-mode fields and two denormal fields.** Page 47:
  `FP_ROUND[1:0]` is the single-precision mode and `[3:2]` the double and half
  mode; `FP_DENORM[5:4]` is single and `[7:6]` double and half.
- **A 64-bit float literal keeps its 32 bits in the high half.** Page 43:
  for a 64-bit float "the lower 32-bit are padded with zero". This reads as
  if it loses the value, but it is right: the literal supplies bits 63:32 (sign,
  exponent and the top of the mantissa) and bits 31:0 are zero. It is not an
  erratum; integer literals are zero- or sign-extended instead (same page).
- **`S_BFE_U32` and `S_BFE_I32` write SCC.** Pages 104 and 105 give `SCC = (D.u != 0)` and `SCC = (D.i != 0)`.
  Older guides omit the SCC line, which is why the errata list mentions them.

## Second sources

Two public sources list or check guide errors. They are used as pointers to
re-derive from the guide, and are never a substitute for it.

- Mesa's `src/amd/compiler/README-ISA.md`, an unofficial GCN and RDNA errata
  list. It covers several of the rows above and adds the hardware hazard notes
  below. Two of its items do not apply to this guide: it says `s_bfe_*` omit
  the SCC write, which document 70648 states on pages 104 and 105, and it reports a
  conflict between guide sections 9.3 and 13.8.1 over the FLAT `SADDR` field,
  where page 86 already accepts both "NULL or 0x7f".
- An independent audit of this guide by another RDNA2 project, whose findings
  are concentrated in the areas this campaign has not reached yet: scalar ALU
  SCC effects, vector ALU modifiers and f16, MUBUF and MIMG opcode bits,
  EXEC and VCC accounting, and the MIMG, EXP and VINTERP surfaces.

## Hardware hazards that apply to EXEC-writing forms

Mesa's errata list records these for gfx10 and gfx10.3, which is this target.
They are not part of the ISA contract and Xenora does not model them, but a
form that writes EXEC has to know about them.

- A SALU instruction that changes EXEC from zero to non-zero immediately
  before an NSA MIMG instruction with four or more dwords can cause stability
  problems. Any instruction, including `s_nop`, mitigates it.
- A non-VALU instruction that reads the EXEC mask followed by a VALU
  instruction that writes it needs a mitigating VALU instruction or
  `s_waitcnt_depctr`.
- A VMEM, FLAT, GLOBAL, SCRATCH or DS instruction that reads an SGPR, EXEC or
  M0 followed by a SALU or SMEM instruction that writes the same SGPR needs a
  VALU instruction or `s_waitcnt` between them.

## Source freshness

The sources this document depends on were re-fetched and compared against their
published versions.

| Source | Pin | Published version |
| --- | --- | --- |
| RDNA 2 guide PDF | SHA-256 `22aec03f2706dba2e148ffde1f3a25d5cdd42c71a96f95240aeb48d995d44795` | 4,430,368 bytes, byte-identical to `https://gpuopen.com/download/rdna2-shader-instruction-set-architecture.pdf` |
| RDNA 2 machine-readable ISA | SHA-256 `d671ecbc36543674ab59e9b2feffd56718fd2137e7ad913c314f2e2508b9f4f7` | `amdgpu_isa_rdna2.xml` from AMD's machine-readable archive, release date 2026-02-20, schema 1.1.1 |
| LLVM AMDGPU backend | commit `5b10d2d74929eae78912771222c9855af97496f4` | LLVM 23.1.6, the current release |

The guide PDF is not stored in the repository. It is public documentation and
is cited by page; the pinned XML snapshot and its licence are in
`crates/isa/data/rdna2.jsonl` and `crates/isa/LICENSE-AMD`.
