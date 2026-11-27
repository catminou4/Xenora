#!/usr/bin/env python3
"""Validate the declarative gfx1030 semantic registry and generate its Rust tables.

  validate   check crates/isa/semantics/rdna2.toml (and the family files it
             includes) against its closed vocabulary
             and the pinned AMD XML snapshot (crates/isa/data/rdna2.jsonl)
  generate   validate, then write crates/isa/src/semantics_gen.rs
  --check    validate and regenerate in memory; exit 1 if the committed file
             differs. Never writes.

The registry is data only. Behaviour is chosen from a closed set of primitives
(crates/isa/src/primitive.rs) by the declared operation, signedness and SCC
effect, never from a mnemonic. Only the Python standard library is used
(tomllib needs Python 3.11 or newer). See docs/shaders/semantic-registry.md.
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter
    sys.exit("generate_semantics.py needs Python 3.11 or newer (tomllib)")

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "crates/isa/semantics/rdna2.toml"
SNAPSHOT = ROOT / "crates/isa/data/rdna2.jsonl"
RUST = ROOT / "crates/isa/src/semantics_gen.rs"

SCHEMA = 1
ARCHITECTURE = "gfx1030"

FORM_KEYS = {
    "name", "family", "opcode", "variant", "operation", "signedness", "carry_in",
    "width", "scc", "vcc", "exec", "wave", "operands", "spec", "verify",
}
TOP_KEYS = {
    "schema", "architecture", "source_sha256", "documents", "operand_rules", "form",
    "encoding_variant", "append_after_variants",
}
# An encoding variant is another encoding of a form already in the registry (for
# example the VOP3 `_e64` encoding of a VOP2 form): pure data that names the base
# form, the encoding family, its opcode there and the operand rule of that
# encoding. The rule may redirect the base form's mask to an explicit destination.
VARIANT_KEYS = {"of", "family", "opcode", "operands", "spec", "verify"}
# The assembler suffix LLVM prints for a variant's encoding family.
ASSEMBLY_SUFFIX = {"VOP3": "_e64"}

# Control effects select an existing typed instruction and its operand contract.
# The XML records SCC as a separate implicit input. VCC and EXEC predicates
# have fieldless mask inputs instead.
CONTROL_OPERATIONS = {
    "end_program": ("EndProgram", "sopp_end", None),
    "branch": ("Branch", "sopp_branch", None),
    "branch_scc_zero": ("BranchSccZero", "sopp_branch", None),
    "branch_vcc_zero": ("BranchVccZero", "sopp_branch", "OPR_VCC"),
    "branch_exec_zero": ("BranchExecZero", "sopp_branch", "OPR_SDST_EXEC"),
    "wait_load": ("WaitLoad", "sopp_wait", None),
}

OPERATIONS = {
    **{name: variant for name, (variant, _, _) in CONTROL_OPERATIONS.items()},
    "add": "Add", "sub": "Sub", "and": "And", "or": "Or", "xor": "Xor",
    "andn2": "AndNot", "orn2": "OrNot", "nand": "Nand", "nor": "Nor", "xnor": "Xnor",
    "andn1": "AndNot0", "orn1": "OrNot0",
    "lshl": "ShiftLeft", "lshr": "LogicalShiftRight", "ashr": "ArithmeticShiftRight",
    "compare": "Compare",
    "not": "Not", "brev": "ReverseBits", "bcnt0": "CountZeros", "bcnt1": "CountOnes",
    "ff0": "FindFirstZero", "ff1": "FindFirstOne", "flbit": "FindLastBit",
    "sext": "SignExtend", "abs": "Absolute",
    "min": "Minimum", "max": "Maximum", "cselect": "Select", "bfm": "BitfieldMask",
    "mul": "Multiply", "mul_hi": "MultiplyHigh", "bfe": "BitfieldExtract",
    "absdiff": "AbsoluteDifference", "lshl_add": "ShiftLeftAdd",
    "mul24": "Multiply24", "mul_hi24": "MultiplyHigh24",
    "mov": "Move", "cmov": "ConditionalMove", "swap": "Swap",
    "trunc": "Truncate", "ceil": "Ceiling", "rndne": "RoundNearestEven", "floor": "Floor",
    # MIMG image sample forms. They have no primitive.rs function: the
    # operation names the texture sample the decoder, reference, interpreters
    # and MSL lowering implement.
    "image_sample_lz": "ImageSampleLz",
    "image_sample_l": "ImageSampleL",
}
# MIMG image operations. Their operand rule has kind "image"; behaviour comes
# from the MIMG contract (docs/shaders/mimg.md), not from a primitive function.
IMAGE_OPERATIONS = {"image_sample_lz", "image_sample_l"}
# Unary operations read one source; they need an operand rule of the unary kind.
UNARY = {
    "not", "brev", "bcnt0", "bcnt1", "ff0", "ff1", "flbit", "sext", "abs",
    "trunc", "ceil", "rndne", "floor",
}
# Binary32 rounding to an integral value (guide 12.8). They take float operands only.
FLOAT_ROUNDING = {"trunc", "ceil", "rndne", "floor"}
# ---- VOP2 binary32 arithmetic (#784) ----
# V_ADD/SUB/SUBREV/MUL/MIN/MAX_F32 (guide 12.7, pages 131-133) reuse these
# operation names with signedness float; only the vop2_f32 rule takes them.
VOP2_FLOAT_ARITHMETIC = {"add", "sub", "mul", "min", "max"}
FLOAT_OPERATIONS = FLOAT_ROUNDING | VOP2_FLOAT_ARITHMETIC
# ---- end VOP2 binary32 arithmetic (#784) ----
# Sign extension reads only the low source_bits of S0 (required for sext only).
SOURCE_BITS_KEY = "source_bits"
SOURCE_BITS = {8, 16}
# Compares take a relation (S0 on the left) and set SCC to whether it holds.
RELATION_KEY = "relation"
RELATIONS = {
    "eq": "Equal", "lg": "NotEqual", "gt": "Greater", "ge": "GreaterOrEqual",
    "lt": "Less", "le": "LessOrEqual", "f": "False", "t": "True",
}
# F and T exist only as VOPC compares (guide Table 59 offsets 0 and 7).
VECTOR_ONLY_RELATIONS = {"f", "t"}
# Guide 12.9 Tables 59 and 61: integer VOPC opcodes are the group base
# for the operand width and signedness plus the relation offset.
VOPC_INTEGER_BASES = {
    16: {"signed": 136, "unsigned": 168},
    32: {"signed": 128, "unsigned": 192},
    64: {"signed": 160, "unsigned": 224},
}
VOPC_RELATION_OFFSETS = {"f": 0, "lt": 1, "eq": 2, "le": 3, "gt": 4, "lg": 5, "ge": 6, "t": 7}
# (relation, signedness) -> primitive. Equality does not depend on signedness,
# so both declared signednesses of EQ and LG share one primitive.
COMPARE_PRIMITIVES = {
    ("eq", "signed"): "eq_b32", ("eq", "unsigned"): "eq_b32",
    ("lg", "signed"): "ne_b32", ("lg", "unsigned"): "ne_b32",
    ("gt", "signed"): "gt_i32", ("gt", "unsigned"): "gt_u32",
    ("ge", "signed"): "ge_i32", ("ge", "unsigned"): "ge_u32",
    ("lt", "signed"): "lt_i32", ("lt", "unsigned"): "lt_u32",
    ("le", "signed"): "le_i32", ("le", "unsigned"): "le_u32",
    ("f", "signed"): "false_b32", ("f", "unsigned"): "false_b32",
    ("t", "signed"): "true_b32", ("t", "unsigned"): "true_b32",
}
# Logical operations take untyped bits; arithmetic ones never do. Shifts take
# the signedness their primitive names, and their S1 is an unsigned count.
LOGICAL = {
    "and", "or", "xor", "andn2", "orn2", "nand", "nor", "xnor",
    "andn1", "orn1",
    "not", "brev", "bcnt0", "bcnt1", "ff0", "ff1", "cselect", "bfm", "swap",
}
ARITHMETIC = {
    "add", "sub", "min", "max", "mul", "mul_hi", "bfe", "absdiff", "lshl_add", "mul24", "mul_hi24",
}
SHIFTS = {"lshl", "lshr", "ashr"}
# Operations whose S1 is an unsigned control word (a shift count or a bit-field
# descriptor), so the XML gives it FMT_NUM_U32 whatever the form's signedness.
CONTROL_S1 = SHIFTS | {"bfe"}
# Operations that always read SCC: the select condition. They need carry_in scc.
SCC_READERS = {"cselect", "cmov"}
# XML number formats a source may have for each declared signedness.
SOURCE_FORMATS = {
    "signed": {"FMT_NUM_I32"},
    "unsigned": {"FMT_NUM_U32"},
    "bitwise": {"FMT_NUM_U32", "FMT_NUM_B32"},
    "float": {"FMT_NUM_F32"},
}
# Optional keys: required exactly when the condition holds.
SHIFT_KEY = "shift_count_bits"
# Shift-and-add declares its fixed S0 shift (required for lshl_add only).
SHIFT_AMOUNT_KEY = "shift_amount"
SHIFT_AMOUNTS = {1, 2, 3, 4}
# Vector forms whose guide pseudocode reads S1 before S0 (D = op(S1, S0))
# declare it; the XML records no operand order.
SOURCE_ORDER_KEY = "source_order"
SOURCE_ORDERS = {"reversed"}
SIGNEDNESS = {"unsigned": "Unsigned", "signed": "Signed", "bitwise": "Bitwise", "float": "Float"}
CARRY_IN = {"none": "None", "scc": "Scc", "vcc": "Vcc"}
SCC = {
    "unchanged": "Unchanged",
    "carry_out": "CarryOut",
    "borrow": "Borrow",
    "signed_overflow": "SignedOverflow",
    "nonzero": "Nonzero",
    "relation": "Relation",
    "selected_src0": "SelectedSrc0",
}
# lane_mask: the Wave32 compare or carry result written to VCC_LO.
# sdst_lane_mask: the Wave32 compare result written to explicit VOP3A SDST.
# EXEC writes such as v_cmpx have no contract, so exec must stay unchanged.
MASK = {"unchanged": "Unchanged", "lane_mask": "LaneMask", "sdst_lane_mask": "SdstLaneMask"}
# Effects on the EXEC mask. `exec_save` writes the combined result to EXEC_LO and
# leaves the previous EXEC_LO in the destination; `exec_write` writes the combined
# result to both EXEC_LO and the destination. Guide 12.3, printed pages 117-122.
EXEC = {"unchanged": "Unchanged", "exec_save": "ExecSave", "exec_write": "ExecWrite"}
# The operations that combine S0 with EXEC_LO. The EXEC operand is the second one,
# except for andn1 and orn1, which negate the first.
EXEC_COMBINE = {"and", "or", "xor", "andn2", "orn2", "nand", "nor", "xnor", "andn1", "orn1"}
WAVE = {"any": "Any", "wave32": "Wave32"}
# The synthetic compute fixture ABI fixes the wave size for these operand-rule
# kinds and for the VCC-reading VOP2 rule. Other forms declare "any".
WAVE_BY_KIND = {"control": "wave32", "vector_compare": "wave32", "vector_unary": "wave32", "vector_swap": "wave32", "image": "wave32"}
VCC_VECTOR_PRIMITIVES = {
    ("cselect", "bitwise"): "Select",
    ("add", "unsigned"): "AddCarry",
    ("sub", "unsigned"): "SubtractBorrow",
}
EVIDENCE = {
    "reference": "Reference",
    "independent": "Independent",
    "metal": "Metal",
    "llvm_encoding": "LlvmEncoding",
}
DOCUMENTS = {"guide": "Guide", "xml": "Xml"}
# The public documents spec references may cite, pinned so a citation cannot
# silently point at another file.
PINNED_DOCUMENTS = {
    "guide": {
        "title": "AMD RDNA 2 Instruction Set Architecture Reference Guide",
        "file": "rdna2-shader-instruction-set-architecture.pdf",
        "pages": 291,
    },
    "xml": {
        "title": "AMD RDNA 2 machine-readable ISA (pinned XML)",
        "file": "amdgpu_isa_rdna2.xml",
    },
}

# (operation, signedness) -> (required SCC effect, primitive, takes carry-in).
# A form whose declaration is not a row here has no primitive and is rejected.
PRIMITIVES = {
    ("add", "unsigned"): ("carry_out", "add_carry_u32", True),
    ("sub", "unsigned"): ("borrow", "sub_borrow_u32", True),
    ("add", "signed"): ("signed_overflow", "add_overflow_i32", False),
    ("sub", "signed"): ("signed_overflow", "sub_overflow_i32", False),
    ("and", "bitwise"): ("nonzero", "and_nonzero_b32", False),
    ("or", "bitwise"): ("nonzero", "or_nonzero_b32", False),
    ("xor", "bitwise"): ("nonzero", "xor_nonzero_b32", False),
    ("andn2", "bitwise"): ("nonzero", "andn2_nonzero_b32", False),
    ("orn2", "bitwise"): ("nonzero", "orn2_nonzero_b32", False),
    ("andn1", "bitwise"): ("nonzero", "andn1_nonzero_b32", False),
    ("orn1", "bitwise"): ("nonzero", "orn1_nonzero_b32", False),
    ("nand", "bitwise"): ("nonzero", "nand_nonzero_b32", False),
    ("nor", "bitwise"): ("nonzero", "nor_nonzero_b32", False),
    ("xnor", "bitwise"): ("nonzero", "xnor_nonzero_b32", False),
    ("lshl", "bitwise"): ("nonzero", "lshl_nonzero_b32", False),
    ("lshr", "unsigned"): ("nonzero", "lshr_nonzero_b32", False),
    ("ashr", "signed"): ("nonzero", "ashr_nonzero_i32", False),
    ("mov", "bitwise"): ("unchanged", "mov_b32", False),
    ("not", "bitwise"): ("nonzero", "not_nonzero_b32", False),
    ("brev", "bitwise"): ("unchanged", "brev_b32", False),
    ("bcnt0", "bitwise"): ("nonzero", "bcnt0_nonzero_b32", False),
    ("bcnt1", "bitwise"): ("nonzero", "bcnt1_nonzero_b32", False),
    ("ff0", "bitwise"): ("unchanged", "ff0_b32", False),
    ("ff1", "bitwise"): ("unchanged", "ff1_b32", False),
    ("flbit", "bitwise"): ("unchanged", "flbit_b32", False),
    ("flbit", "signed"): ("unchanged", "flbit_i32", False),
    ("sext", "signed"): ("unchanged", "sext_i32", False),
    ("abs", "signed"): ("nonzero", "abs_nonzero_i32", False),
    ("min", "signed"): ("selected_src0", "min_i32", False),
    ("min", "unsigned"): ("selected_src0", "min_u32", False),
    ("max", "signed"): ("selected_src0", "max_i32", False),
    ("max", "unsigned"): ("selected_src0", "max_u32", False),
    ("cselect", "bitwise"): ("unchanged", "cselect_b32", True),
    ("bfm", "bitwise"): ("unchanged", "bfm_b32", False),
    ("mul", "signed"): ("unchanged", "mul_lo_i32", False),
    ("mul_hi", "unsigned"): ("unchanged", "mul_hi_u32", False),
    ("mul_hi", "signed"): ("unchanged", "mul_hi_i32", False),
    ("bfe", "unsigned"): ("nonzero", "bfe_nonzero_u32", False),
    ("bfe", "signed"): ("nonzero", "bfe_nonzero_i32", False),
    ("absdiff", "signed"): ("nonzero", "absdiff_nonzero_i32", False),
    ("lshl_add", "unsigned"): ("carry_out", "lshl_add_carry_u32", False),
}

# SOPK forms (operation, signedness) -> (SCC effect, primitive, reads SCC). D is
# SDST and the immediate is SIMM16 sign-extended (guide 12.2); `sext_simm16` is
# the sign extension, so a move or conditional move writes it unchanged.
IMMEDIATE_PRIMITIVES = {
    ("mov", "signed"): ("unchanged", "sext_simm16", False),
    ("cmov", "signed"): ("unchanged", "sext_simm16", True),
    ("add", "signed"): ("signed_overflow", "add_overflow_i32", False),
    ("mul", "signed"): ("unchanged", "mul_lo_i32", False),
}
# SOPK compares: the opcode is the group base (3 signed, 9 unsigned) plus the
# relation's index in this order (guide 13.1.2).
SOPK_COMPARE_BASES = {"signed": 3, "unsigned": 9}
SOPK_RELATION_ORDER = ["eq", "lg", "gt", "ge", "lt", "le"]
# Operand-rule kinds whose forms have a 16-bit immediate field.
IMMEDIATE_KINDS = {"immediate", "immediate_compare"}

# Vector (VOP2) forms: (operation, signedness) -> the per-lane VectorBinary their
# MSL lowering uses. They write no SCC, VCC or EXEC, so scc must be unchanged.
VECTOR_PRIMITIVES = {
    ("add", "unsigned"): "Add",
    ("sub", "unsigned"): "Sub",
    ("xnor", "bitwise"): "Xnor",
    ("min", "signed"): "MinSigned",
    ("min", "unsigned"): "MinUnsigned",
    ("max", "signed"): "MaxSigned",
    ("max", "unsigned"): "MaxUnsigned",
    ("mul24", "signed"): "Mul24Signed",
    ("mul24", "unsigned"): "Mul24Unsigned",
    ("mul_hi24", "signed"): "MulHi24Signed",
    ("mul_hi24", "unsigned"): "MulHi24Unsigned",
    ("and", "bitwise"): "And",
    ("or", "bitwise"): "Or",
    ("xor", "bitwise"): "Xor",
    ("lshl", "bitwise"): "ShiftLeft",
    ("lshr", "bitwise"): "ShiftRight",
    ("ashr", "signed"): "ArithmeticShiftRight",
    # ---- VOP2 binary32 arithmetic (#784) ----
    # V_SUBREV_F32 is ("sub", "float") with source_order = "reversed".
    ("add", "float"): "AddF32",
    ("sub", "float"): "SubF32",
    ("mul", "float"): "MulF32",
    ("min", "float"): "MinF32",
    ("max", "float"): "MaxF32",
    # ---- end VOP2 binary32 arithmetic (#784) ----
}

# Operand rules the decoder implements, each exactly once. A registry rule must
# match one of these shapes; anything else is an unimplemented rule.
UNSUPPORTED_COMBOS = {
    "dst_vcc", "dst_exec", "dst_ttmp", "dst_m0", "dst_null", "src_vcc", "src_exec",
    "src_m0", "src_ttmp", "src_float_inline", "src_special", "two_literals",
}
# Every combo name a rule may list (VOP2 also names vcc_hi and NULL sources,
# which SOP2 groups under src_vcc and src_special).
# VOP3 rules also name source classes and encoding fields they refuse.
VOP3_COMBOS = {"src_sgpr", "src_inline", "src_literal", "modifiers", "src2"}
# src_nan: a NaN value in a float source, whose result depends on MODE.IEEE
# and MODE.DX10_CLAMP (guide 3.5); the translator does not model it.
ALL_COMBOS = UNSUPPORTED_COMBOS | {"src_vcc_hi", "src_null", "dst_vcc_hi", "src_nan"} | VOP3_COMBOS
VOP3_VGPR2_UNSUPPORTED = {
    "src_sgpr", "src_vcc", "src_exec", "src_m0", "src_ttmp", "src_null", "src_inline",
    "src_float_inline", "src_special", "src_literal", "modifiers", "src2",
}

VECTOR_UNARY_PRIMITIVES = {
    ("not", "bitwise"): "Not",
    ("brev", "bitwise"): "ReverseBits",
    ("flbit", "unsigned"): "FindHighUnsigned",
    ("ff1", "bitwise"): "FindLow",
    ("flbit", "signed"): "FindHighSigned",
    ("trunc", "float"): "TruncateF32",
    ("ceil", "float"): "CeilingF32",
    ("rndne", "float"): "RoundNearestEvenF32",
    ("floor", "float"): "FloorF32",
}

OPERAND_RULES = {
    "sopp_end": {
        "rust": "SoppEnd",
        "enum": "SoppControlOp",
        "shape": {"family": "SOPP", "src": [], "max_literals": 0, "immediate": 0},
        "unsupported": set(),
        "kind": "control",
    },
    "sopp_branch": {
        "rust": "SoppBranch",
        "enum": "SoppControlOp",
        "shape": {"family": "SOPP", "src": [{"field": "SIMM16", "class": "label"}], "max_literals": 0},
        "unsupported": set(),
        "kind": "control",
    },
    "sopp_wait": {
        "rust": "SoppWait",
        "enum": "SoppControlOp",
        "shape": {"family": "SOPP", "src": [{"field": "SIMM16", "class": "waitcnt"}],
                  "max_literals": 0, "immediate": 0x3f70},
        "unsupported": set(),
        "kind": "control",
    },
    "vop1_vgpr": {
        "rust": "Vop1Vgpr",
        "enum": "VectorUnaryOp",
        "shape": {
            "family": "VOP1",
            "dst": {"field": "VDST", "class": "vgpr", "min": 0, "max": 255},
            "src": [{"field": "SRC0", "class": "vector_source"}],
            "max_literals": 1,
        },
        "unsupported": {
            "src_vcc_hi", "src_exec", "src_m0", "src_ttmp", "src_null", "src_float_inline",
            "src_special",
        },
        "kind": "vector_unary",
    },
    # Binary32 rounding: the guide does not say how a float operation reads an
    # integer inline constant, so only VGPR, SGPR, vcc_lo and literal SRC0 are kept.
    "vop1_f32": {
        "rust": "Vop1F32",
        "enum": "VectorUnaryOp",
        "shape": {
            "family": "VOP1",
            "dst": {"field": "VDST", "class": "vgpr", "min": 0, "max": 255},
            "src": [{"field": "SRC0", "class": "vector_source"}],
            "max_literals": 1,
        },
        "unsupported": {
            "src_vcc_hi", "src_exec", "src_m0", "src_ttmp", "src_null", "src_inline",
            "src_float_inline", "src_special", "src_nan",
        },
        "kind": "vector_unary",
    },
    "vop1_swap": {
        "rust": "Vop1Swap",
        "enum": "VectorSwapOp",
        "shape": {
            "family": "VOP1",
            "dst": {"field": "VDST", "class": "vgpr", "min": 0, "max": 255},
            "src": [{"field": "SRC0", "class": "vgpr"}],
            "max_literals": 0,
        },
        "unsupported": {
            "src_sgpr", "src_vcc", "src_exec", "src_m0", "src_ttmp", "src_null", "src_inline",
            "src_float_inline", "src_special", "src_literal",
        },
        "kind": "vector_swap",
    },
    "sop2_sgpr": {
        "rust": "Sop2Sgpr",
        "enum": "ScalarAluOp",
        "shape": {
            "family": "SOP2",
            "dst": {"field": "SDST", "class": "sgpr", "min": 0, "max": 105},
            "src": [
                {"field": "SSRC0", "class": "scalar_source"},
                {"field": "SSRC1", "class": "scalar_source"},
            ],
            "max_literals": 1,
        },
        # Every one of these is rejected by the decoder's SOP2 operand checks.
        "unsupported": UNSUPPORTED_COMBOS,
        "kind": "alu",
    },
    "sopc_scalar": {
        "rust": "SopcScalar",
        "enum": "ScalarCompareOp",
        "shape": {
            "family": "SOPC",
            "src": [
                {"field": "SSRC0", "class": "scalar_source"},
                {"field": "SSRC1", "class": "scalar_source"},
            ],
            "max_literals": 1,
        },
        # SOPC has no destination field; the decoder rejects these sources.
        "unsupported": {combo for combo in UNSUPPORTED_COMBOS if not combo.startswith("dst_")},
        "kind": "compare",
    },
    "sop1_sgpr": {
        "rust": "Sop1Sgpr",
        "enum": "ScalarUnaryOp",
        "shape": {
            "family": "SOP1",
            "dst": {"field": "SDST", "class": "sgpr", "min": 0, "max": 105},
            "src": [{"field": "SSRC0", "class": "scalar_source"}],
            "max_literals": 1,
        },
        # One source, so two literals cannot occur; the rest are rejected.
        "unsupported": {combo for combo in UNSUPPORTED_COMBOS if combo != "two_literals"},
        "kind": "unary",
    },
    "sop1_move": {
        "rust": "Sop1Move",
        "enum": "ScalarUnaryOp",
        "shape": {
            "family": "SOP1",
            "dst": {"field": "SDST", "class": "sgpr_or_vcc_exec"},
            "src": [{"field": "SSRC0", "class": "scalar_source"}],
            "max_literals": 1,
        },
        "unsupported": UNSUPPORTED_COMBOS - {"two_literals", "dst_vcc", "dst_exec"},
        "kind": "unary",
    },
    "sop1_saveexec": {
        "rust": "Sop1SaveExec",
        "enum": "ScalarUnaryOp",
        "shape": {
            "family": "SOP1",
            "dst": {"field": "SDST", "class": "sgpr", "min": 0, "max": 105},
            "src": [{"field": "SSRC0", "class": "scalar_source"}],
            "max_literals": 1,
        },
        # One source, so two literals cannot occur; the rest are rejected.
        "unsupported": {combo for combo in UNSUPPORTED_COMBOS if combo != "two_literals"},
        "kind": "unary",
    },
    "sopk_sgpr": {
        "rust": "SopkSgpr",
        "enum": "ScalarImmediateOp",
        "shape": {
            "family": "SOPK",
            "dst": {"field": "SDST", "class": "sgpr", "min": 0, "max": 105},
            "src": [{"field": "SIMM16", "class": "simm16"}],
            "max_literals": 0,
        },
        # SOPK never takes a literal dword; the decoder rejects SDST 106-127.
        "unsupported": {combo for combo in UNSUPPORTED_COMBOS if combo.startswith("dst_")},
        "kind": "immediate",
    },
    "sopk_compare": {
        "rust": "SopkCompare",
        "enum": "ScalarImmediateCompareOp",
        "shape": {
            "family": "SOPK",
            "src": [
                {"field": "SDST", "class": "sgpr", "min": 0, "max": 105},
                {"field": "SIMM16", "class": "simm16"},
            ],
            "max_literals": 0,
        },
        # The compared register is SDST (S0); the decoder rejects SDST 106-127.
        "unsupported": {combo for combo in UNSUPPORTED_COMBOS if combo.startswith("dst_")},
        "kind": "immediate_compare",
    },
    "vopc_vcc": {
        "rust": "VopcVcc",
        "enum": "VectorCompareOp",
        "shape": {
            "family": "VOPC",
            "src": [
                {"field": "SRC0", "class": "vector_source"},
                {"field": "VSRC1", "class": "vgpr"},
            ],
            "max_literals": 1,
        },
        # The destination is implicitly vcc_lo; the decoder rejects these SRC0
        # selectors (the same vector_source as VOP2).
        "unsupported": {
            "src_vcc_hi", "src_exec", "src_m0", "src_ttmp", "src_null", "src_float_inline",
            "src_special",
        },
        "kind": "vector_compare",
    },
    "vopc_vcc_half": {
        "rust": "VopcVccHalf",
        "enum": "VectorCompareOp",
        "shape": {
            "family": "VOPC",
            "src": [
                {"field": "SRC0", "class": "vector_source"},
                {"field": "VSRC1", "class": "vgpr"},
            ],
            "max_literals": 0,
        },
        # The destination is implicitly vcc_lo; the decoder rejects these SRC0
        # selectors (the same vector_source as VOP2).
        "unsupported": {
            "src_literal", "src_vcc_hi", "src_exec", "src_m0", "src_ttmp", "src_null", "src_float_inline",
            "src_special",
        },
        "kind": "vector_compare",
        "width": 16,
    },
    "vopc_vcc_pair": {
        "rust": "VopcVccPair",
        "enum": "VectorCompareOp",
        "shape": {
            "family": "VOPC",
            "src": [
                {"field": "SRC0", "class": "vector_pair_or_inline_integer"},
                {"field": "VSRC1", "class": "vgpr_pair"},
            ],
            "max_literals": 0,
        },
        "unsupported": {
            "src_vcc", "src_literal", "src_exec", "src_m0", "src_ttmp", "src_null",
            "src_float_inline", "src_special",
        },
        "kind": "vector_compare",
        "width": 64,
    },
    "vop3_vgpr2": {
        "rust": "Vop3Vgpr2",
        "enum": "VectorAluOp",
        "shape": {
            "family": "VOP3",
            "dst": {"field": "VDST", "class": "vgpr", "min": 0, "max": 255},
            "src": [
                {"field": "SRC0", "class": "vgpr"},
                {"field": "SRC1", "class": "vgpr"},
            ],
            "max_literals": 0,
            # Declared in the data, so every reader (the planner too) knows this
            # rule only re-encodes declared forms.
            "variant_only": True,
        },
        # Two VGPR sources only, with ABS, OP_SEL, CLAMP, OMOD and NEG clear and
        # SRC2 the assembler's unused inline zero; the decoder rejects the rest.
        "unsupported": VOP3_VGPR2_UNSUPPORTED,
        "kind": "vector_vop3",
        # Only encoding variants of forms whose rule has this kind may use it.
        "variant_of": "vector",
        # The generated enum of this rule's variants; base() maps to "enum".
        "variant_enum": "VectorAluVop3Op",
    },
    "vop3_cmp_vgpr2": {
        "rust": "Vop3CmpVgpr2",
        "enum": "VectorCompareOp",
        "shape": {
            "family": "VOP3",
            "dst": {"field": "VDST", "class": "sgpr", "min": 0, "max": 106},
            "src": [
                {"field": "SRC0", "class": "vgpr"},
                {"field": "SRC1", "class": "vgpr"},
            ],
            "max_literals": 0,
            "variant_only": True,
        },
        "unsupported": VOP3_VGPR2_UNSUPPORTED | {
            "dst_vcc_hi", "dst_exec", "dst_ttmp", "dst_m0", "dst_null",
        },
        "kind": "vector_compare_vop3",
        "variant_of": "vector_compare",
        "variant_enum": "VectorCompareVop3Op",
        "vcc_transition": ("lane_mask", "sdst_lane_mask"),
    },
    "vop2_vgpr": {
        "rust": "Vop2Vgpr",
        "enum": "VectorAluOp",
        "shape": {
            "family": "VOP2",
            "dst": {"field": "VDST", "class": "vgpr", "min": 0, "max": 255},
            "src": [
                {"field": "SRC0", "class": "vector_source"},
                {"field": "VSRC1", "class": "vgpr"},
            ],
            "max_literals": 1,
        },
        # vector_source = VGPR, s0-s105, vcc_lo, inline integer -16..64 or a
        # literal; the decoder rejects these SRC0 selectors.
        "unsupported": {
            "src_vcc_hi", "src_exec", "src_m0", "src_ttmp", "src_null", "src_float_inline",
            "src_special",
        },
        "kind": "vector",
    },
    "vop2_vcc": {
        "rust": "Vop2Vcc",
        "enum": "VectorAluOp",
        "shape": {
            "family": "VOP2",
            "dst": {"field": "VDST", "class": "vgpr", "min": 0, "max": 255},
            "src": [
                {"field": "SRC0", "class": "vector_source"},
                {"field": "VSRC1", "class": "vgpr"},
            ],
            "max_literals": 1,
        },
        "unsupported": {
            "src_vcc_hi", "src_exec", "src_m0", "src_ttmp", "src_null", "src_float_inline",
            "src_special",
        },
        "kind": "vector",
    },
    "vop2_vgpr2": {
        "rust": "Vop2Vgpr2",
        "enum": "VectorAluOp",
        "shape": {
            "family": "VOP2",
            "dst": {"field": "VDST", "class": "vgpr", "min": 0, "max": 255},
            "src": [
                {"field": "SRC0", "class": "vgpr"},
                {"field": "VSRC1", "class": "vgpr"},
            ],
            "max_literals": 0,
        },
        # The subset the hand-written decoder accepted before the registry (#215
        # M27): SRC0 must select a VGPR (256-511); the decoder rejects the rest.
        "unsupported": {
            "src_sgpr", "src_vcc", "src_exec", "src_m0", "src_ttmp", "src_null", "src_inline",
            "src_float_inline", "src_special", "src_literal",
        },
        "kind": "vector",
    },
    # ---- VOP2 binary32 arithmetic (#784) ----
    # As vop1_f32: the guide does not say how a float operation reads an
    # integer inline constant, so SRC0 is a VGPR, normal SGPR, vcc_lo or a
    # literal; NaN source values are an unsupported variant.
    "vop2_f32": {
        "rust": "Vop2F32",
        "enum": "VectorAluOp",
        "shape": {
            "family": "VOP2",
            "dst": {"field": "VDST", "class": "vgpr", "min": 0, "max": 255},
            "src": [
                {"field": "SRC0", "class": "vector_source"},
                {"field": "VSRC1", "class": "vgpr"},
            ],
            "max_literals": 1,
        },
        "unsupported": {
            "src_vcc_hi", "src_exec", "src_m0", "src_ttmp", "src_null", "src_inline",
            "src_float_inline", "src_special", "src_nan",
        },
        "kind": "vector",
    },
    # ---- end VOP2 binary32 arithmetic (#784) ----
    # MIMG image sample (IMAGE_SAMPLE_LZ and IMAGE_SAMPLE_L). The image operand
    # structure (VDATA, VADDR, SRSRC, SSAMP, plus the DMASK, DIM, R128 and
    # modifier fields) does not fit the src/dst shape of the ALU rules, so the
    # shape declares no fields and the decoder validates the image encoding. The
    # generator emits a dedicated `from_word` for the split OP field.
    "mimg_2d": {
        "rust": "Mimg2D",
        "enum": "ImageSampleOp",
        "shape": {"family": "MIMG", "src": [], "max_literals": 0},
        "unsupported": set(),
        "kind": "image",
    },
}

VARIANT = re.compile(r"^[A-Z][A-Za-z0-9]*$")
RESERVED_VARIANTS = {"Self", "ALL"}
U16_MAX = 0xFFFF
NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
SECTION = re.compile(r"^[0-9]+(\.[0-9]+)*$")


class RegistryError(ValueError):
    """A registry file set that cannot be merged into one registry."""


# An include names a family file as `<root stem>/<lower-case family>.toml`.
INCLUDE = re.compile(r"^[a-z0-9_]+/[a-z0-9]+\.toml$")
# Keys a family file may hold; everything else lives in the root file.
FAMILY_FILE_KEYS = {"operand_rules", "form", "encoding_variant"}


def load_registry(path=REGISTRY):
    """Parse the root registry file and merge its family files into one dict.

    The root lists its family files under `include`, in order. Each family file
    holds only operand rules and forms of the one encoding family it is named
    after (`rdna2/vop2.toml` holds VOP2), so parallel work on different
    families never edits the same file. Forms keep the include order, so the
    merged registry is the same as one file with the parts concatenated.
    """
    path = Path(path)
    with open(path, "rb") as handle:
        registry = tomllib.load(handle)
    includes = registry.pop("include", [])
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise RegistryError(f"{path.name}: include must be a list of file names")
    # Family files live directly in the directory named after the root
    # (rdna2.toml -> rdna2/), as `<stem>/<family>.toml`; nothing else resolves.
    for include in includes:
        if not INCLUDE.match(include) or include.split("/", 1)[0] != path.stem:
            raise RegistryError(
                f"{path.name}: include {include!r} is not {path.stem}/<family>.toml"
            )
    if len(set(includes)) != len(includes):
        raise RegistryError(f"{path.name}: a family file is included twice")
    for key, kind, name in (("operand_rules", dict, "a table"), ("form", list, "an array"),
                            ("encoding_variant", list, "an array")):
        if not isinstance(registry.get(key, kind()), kind):
            raise RegistryError(f"{path.name}: {key} must be {name}")
    rules = registry.setdefault("operand_rules", {})
    forms = registry.setdefault("form", [])
    # A standalone registry (no includes) needs no family directory.
    if includes:
        if (path.parent / path.stem).is_symlink():
            raise RegistryError(f"{path.name}: the family directory {path.stem}/ is a link")
        try:
            directory = (path.parent / path.stem).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise RegistryError(
                f"{path.name}: the family directory {path.stem}/ cannot be read: {error}"
            )
    identities = set()
    for include in includes:
        part_path = path.parent / include
        family = Path(include).stem.upper()
        if not part_path.is_file():
            raise RegistryError(f"{path.name}: included family file {include} does not exist")
        # A family file is a regular file directly in the registry directory,
        # not a link to data elsewhere (nor a second name for another include).
        if part_path.is_symlink() or part_path.resolve().parent != directory:
            raise RegistryError(f"{path.name}: included family file {include} is a link or outside {path.stem}/")
        status = part_path.stat()
        if (status.st_dev, status.st_ino) in identities:
            raise RegistryError(f"{path.name}: included family file {include} is another name for an included file")
        identities.add((status.st_dev, status.st_ino))
        with open(part_path, "rb") as handle:
            part = tomllib.load(handle)
        unknown = sorted(set(part) - FAMILY_FILE_KEYS)
        if unknown:
            raise RegistryError(f"{include}: keys {unknown} belong in the root file")
        for key, kind, name in (("operand_rules", dict, "a table"), ("form", list, "an array"),
                                ("encoding_variant", list, "an array")):
            if not isinstance(part.get(key, kind()), kind):
                raise RegistryError(f"{include}: {key} must be {name}")
        for name, rule in part.get("operand_rules", {}).items():
            if name in rules:
                raise RegistryError(f"{include}: operand rule {name} is declared twice")
            if not isinstance(rule, dict) or rule.get("family") != family:
                raise RegistryError(f"{include}: operand rule {name} is not a {family} rule")
            rules[name] = rule
        for index, form in enumerate(part.get("form", [])):
            if not isinstance(form, dict) or form.get("family") != family:
                label = form.get("name", f"form #{index}") if isinstance(form, dict) else index
                raise RegistryError(f"{include}: form {label} is not a {family} form")
            forms.append(form)
        for index, variant in enumerate(part.get("encoding_variant", [])):
            if not isinstance(variant, dict) or variant.get("family") != family:
                label = variant.get("of", f"#{index}") if isinstance(variant, dict) else index
                raise RegistryError(f"{include}: encoding variant {label} is not a {family} encoding")
            registry.setdefault("encoding_variant", []).append(variant)
    return registry


def load_snapshot(path=SNAPSHOT):
    """Index the normalised XML snapshot: source line, encodings, default variants."""
    source = None
    encodings = {}
    variants = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "source":
                source = record
            elif kind == "encoding":
                encodings[record["name"]] = record
            elif kind == "variant" and record.get("condition") == "default":
                variants[(record["instruction"], record["encoding"])] = record
    return {"source": source or {}, "encodings": encodings, "variants": variants}


def _family_encoding(family):
    return "ENC_" + family


def _rust_family(family):
    return family[0] + family[1:].lower()


def _check_rule(name, rule, errors):
    implemented = OPERAND_RULES.get(name)
    if implemented is None:
        errors.append(f"operand rule {name}: not implemented by the decoder")
        return
    unsupported = rule.get("unsupported", [])
    unknown = sorted(set(unsupported) - ALL_COMBOS)
    if unknown:
        errors.append(f"operand rule {name}: unknown unsupported combos {unknown}")
    if len(set(unsupported)) != len(unsupported):
        errors.append(f"operand rule {name}: duplicate unsupported combos")
    if set(unsupported) != implemented["unsupported"]:
        errors.append(
            f"operand rule {name}: unsupported combos differ from those the decoder rejects"
        )
    declared = {key: value for key, value in rule.items() if key != "unsupported"}
    if declared != implemented["shape"]:
        errors.append(
            f"operand rule {name}: declaration differs from the decoder's implemented rule"
        )


def _check_xml(form, rule, snapshot, errors):
    label = form["name"]
    encoding_name = _family_encoding(form["family"])
    encoding = snapshot["encodings"].get(encoding_name)
    if encoding is None:
        errors.append(f"{label}: family {form['family']} has no XML encoding {encoding_name}")
        return
    variant = snapshot["variants"].get((form["name"], encoding_name))
    if variant is None:
        errors.append(f"{label}: no default {encoding_name} variant in the XML snapshot")
        return
    if variant["opcode"] != form["opcode"]:
        errors.append(
            f"{label}: opcode {form['opcode']} disagrees with the XML opcode {variant['opcode']}"
        )
    if _identifier(encoding, form["opcode"]) not in encoding["identifiers"]:
        errors.append(f"{label}: opcode {form['opcode']} is not an identifier of {encoding_name}")
    operands = {}
    scc_in = scc_out = False
    implicit = set()
    vcc_operands = []
    exec_operands = []
    control_mask = CONTROL_OPERATIONS.get(form["operation"], (None, None, None))[2]
    mask_inputs = 0
    for field, data_format, kind, size, is_input, is_output, is_implicit, encoded in variant["operands"]:
        if kind == "OPR_SSRC_SPECIAL_SCC":
            scc_in |= is_input
            scc_out |= is_output
        elif control_mask and (field, kind, data_format, size, is_input, is_output, is_implicit, encoded) == (
            "", control_mask, "FMT_NUM_M64", 64, True, False, True, True
        ):
            mask_inputs += 1
        elif field:
            operands[field] = (data_format, kind, size, is_input, is_output)
        elif kind == "OPR_VCC":
            vcc_operands.append((data_format, size, is_input, is_output))
        elif kind == "OPR_SDST_EXEC":
            # The EXEC mask is an implicit operand: the XML lists it as a read
            # and a write for the forms that update it.
            exec_operands.append((is_input, is_output))
        else:
            implicit.add(kind)
    if form["carry_in"] == "vcc":
        if vcc_operands.count(("FMT_NUM_M64", 64, True, False)) != 1:
            errors.append(f"{label}: carry_in vcc requires one implicit XML VCC input")
        if ("FMT_NUM_M64", 64, True, False) in vcc_operands:
            vcc_operands.remove(("FMT_NUM_M64", 64, True, False))
    if form["vcc"] == "lane_mask":
        if vcc_operands != [("FMT_NUM_M64", 64, False, True)]:
            errors.append(f"{label}: vcc lane_mask but the XML has no implicit VCC output")
    elif vcc_operands:
        implicit.add("OPR_VCC")
    if control_mask and mask_inputs != 1:
        errors.append(f"{label}: expected one implicit {control_mask} input in the XML")
    if implicit:
        errors.append(f"{label}: XML implicit operands {sorted(implicit)} are not declared")
    if scc_in != (form["carry_in"] == "scc"):
        errors.append(f"{label}: carry_in {form['carry_in']} contradicts the XML SCC input")
    if scc_out != (form["scc"] != "unchanged"):
        errors.append(f"{label}: scc {form['scc']} contradicts the XML SCC output")
    if form["exec"] == "unchanged":
        if exec_operands:
            errors.append(f"{label}: the XML has an implicit EXEC operand but no EXEC effect is declared")
    elif exec_operands.count((True, False)) != 1 or exec_operands.count((False, True)) != 1:
        errors.append(
            f"{label}: exec {form['exec']} needs one implicit EXEC read and one implicit EXEC write"
        )
    if _rule_kind(form) == "image":
        # The MIMG operand set is fixed by the image contract: a written VDATA,
        # a read address tuple and the read resource and sampler descriptors.
        # Their variable widths (VADDR is 2 or 3 f32 components, SRSRC 4 or 8
        # SGPRs) are validated by the decoder, not through the ALU field rules.
        actual = set(operands)
        if actual != {"VDATA", "VADDR", "SRSRC", "SSAMP"}:
            errors.append(f"{label}: XML image operands {sorted(actual)} are not VDATA/VADDR/SRSRC/SSAMP")
        else:
            if operands["VDATA"][:3] != ("FMT_ANY", "OPR_VGPR", 128) or operands["VDATA"][3:] != (False, True):
                errors.append(f"{label}: VDATA must be a written 128-bit VGPR")
            if operands["VADDR"][3:] != (True, False):
                errors.append(f"{label}: VADDR must be a read address input")
            if operands["SRSRC"][1:] != ("OPR_SREG_NONULL", 256, True, False):
                errors.append(f"{label}: SRSRC must be a read 256-bit aligned SGPR")
            if operands["SSAMP"][1:] != ("OPR_SREG_NONULL", 128, True, False):
                errors.append(f"{label}: SSAMP must be a read 128-bit aligned SGPR")
        return
    if rule is None:
        return
    if not _well_formed(rule):
        return  # the rule check already reported the malformed shape
    fields = [(rule["dst"]["field"], False)] if "dst" in rule else []
    fields += [(src["field"], True) for src in rule["src"]]
    shift = form["operation"] in CONTROL_S1 and len(rule["src"]) > 1
    # The shift count is S1, or S0 for a form that reads its sources reversed.
    count_field = rule["src"][0 if SOURCE_ORDER_KEY in form else 1]["field"] if shift else None
    for field, is_source in fields:
        if field not in operands:
            errors.append(f"{label}: XML has no operand {field}")
            continue
        data_format, kind, size, is_input, is_output = operands[field]
        if field == "SIMM16" and _rule_kind(form) == "control":
            expected_kind, expected_format = (
                ("OPR_LABEL", "FMT_NUM_I16") if form["operands"] == "sopp_branch"
                else ("OPR_WAITCNT", "FMT_NUM_B16")
            )
            if [field, expected_format, expected_kind, 16, True, False, False, True] not in variant["operands"]:
                errors.append(f"{label}: SIMM16 is not an encoded {expected_kind} input")
            continue
        if field == "SIMM16":
            # The 16-bit immediate: signed forms read it as FMT_NUM_I16 (sign
            # extended), unsigned compares as FMT_NUM_U16 (zero extended).
            wanted = "FMT_NUM_U16" if form["signedness"] == "unsigned" else "FMT_NUM_I16"
            if size != 16 or data_format != wanted:
                errors.append(f"{label}: SIMM16 is {size} bits {data_format} in the XML, expected 16 {wanted}")
            if (is_input, is_output) != (True, False):
                errors.append(f"{label}: {field} direction disagrees with the XML")
            continue
        bits = form.get(SOURCE_BITS_KEY) if is_source else None
        expected = bits if type(bits) is int else form["width"]
        if not is_source and form["vcc"] == "sdst_lane_mask":
            # The XML describes the Wave64 pair. The Wave32 ABI writes one SGPR.
            expected = 64
            if (kind, data_format) != ("OPR_SREG", "FMT_NUM_M64"):
                errors.append(f"{label}: {field} is not a scalar mask destination in the XML")
        if size != expected:
            errors.append(f"{label}: {field} is {size} bits in the XML, declared {expected}")
        # The XML marks both swap operands as outputs. Guide 12.8 also
        # reads both old values before assigning either result.
        direction = (False, True) if _rule_kind(form) == "vector_swap" else (is_source, not is_source)
        if (is_input, is_output) != direction:
            errors.append(f"{label}: {field} direction disagrees with the XML")
        if field == count_field:
            if data_format != "FMT_NUM_U32":
                errors.append(
                    f"{label}: control operand {field} has XML format {data_format}, not FMT_NUM_U32"
                )
        elif type(bits) is int:
            if data_format != f"FMT_NUM_I{bits}":
                errors.append(f"{label}: {field} has XML format {data_format}, not FMT_NUM_I{bits}")
        elif is_source and data_format not in (
            {f"FMT_NUM_{'I' if form['signedness'] == 'signed' else 'U'}{form['width']}"}
            if form["operands"] in ("vopc_vcc_half", "vopc_vcc_pair")
            else SOURCE_FORMATS.get(form["signedness"], ())
        ):
            errors.append(
                f"{label}: signedness {form['signedness']} contradicts XML format {data_format}"
            )
    extra = sorted(set(operands) - {field for field, _ in fields})
    if extra:
        errors.append(f"{label}: XML operands {extra} are not covered by rule {form['operands']}")


def _identifier(encoding, opcode):
    fields = dict((name, ranges) for name, ranges in encoding["fields"])
    ranges = fields["OP"]
    base = encoding["identifiers"][0]
    if len(ranges) == 1:
        (offset, size), = ranges
        base &= ~(((1 << size) - 1) << offset)
        return base | (opcode << offset)
    # Split OP field (MIMG): the XML lists the ranges high-bits-first, so the
    # first range holds the low opcode bits and the second the high bit(s).
    (offset, size), (high_offset, high_size) = ranges
    base &= ~(((1 << size) - 1) << offset)
    base &= ~(((1 << high_size) - 1) << high_offset)
    low = opcode & ((1 << size) - 1)
    high = opcode >> size
    return base | (low << offset) | (high << high_offset)


def validate(registry, snapshot):
    """Return every error in the registry; an empty list means it is valid."""
    errors = []
    unknown = sorted(set(registry) - TOP_KEYS)
    if unknown:
        errors.append(f"unknown top-level keys {unknown}")
    if type(registry.get("schema")) is not int or registry["schema"] != SCHEMA:
        errors.append(f"schema {registry.get('schema')!r} is not {SCHEMA}")
    if registry.get("architecture") != ARCHITECTURE:
        errors.append(f"architecture {registry.get('architecture')!r} is not {ARCHITECTURE}")
    if registry.get("source_sha256") != snapshot["source"].get("sha256"):
        errors.append("source_sha256 does not match the XML snapshot")
    documents = registry.get("documents", {})
    for name, document in documents.items():
        if name not in DOCUMENTS:
            errors.append(f"unknown document {name}")
        elif document != PINNED_DOCUMENTS[name]:
            errors.append(f"document {name}: differs from the pinned public document")
    deferred = registry.get("append_after_variants", [])
    if not isinstance(deferred, list) or any(not isinstance(family, str) for family in deferred):
        errors.append("append_after_variants must be an array of family names")
    elif len(set(deferred)) != len(deferred):
        errors.append("append_after_variants names a family twice")
    else:
        declared_families = {form.get("family") for form in registry.get("form", [])}
        for family in deferred:
            if family not in declared_families:
                errors.append(f"append_after_variants: unknown family {family!r}")
    if "guide" not in documents:
        errors.append("the guide document is not declared")
    rules = registry.get("operand_rules", {})
    for name, rule in rules.items():
        _check_rule(name, rule, errors)
    forms = registry.get("form", [])
    if not forms:
        errors.append("the registry declares no forms")
    seen = {"name": set(), "opcode": set(), "variant": set()}
    for index, form in enumerate(forms):
        label = form.get("name", f"form #{index}")
        missing = sorted(FORM_KEYS - set(form))
        extra = sorted(
            set(form) - FORM_KEYS
            - {SHIFT_KEY, SHIFT_AMOUNT_KEY, RELATION_KEY, SOURCE_BITS_KEY, SOURCE_ORDER_KEY}
        )
        if missing:
            errors.append(f"{label}: incomplete, missing {missing}")
        if extra:
            errors.append(f"{label}: unknown keys {extra}")
        if missing:
            continue
        for key, vocabulary in (
            ("operation", OPERATIONS), ("signedness", SIGNEDNESS), ("carry_in", CARRY_IN),
            ("scc", SCC), ("vcc", MASK), ("exec", EXEC), ("wave", WAVE),
        ):
            if form[key] not in vocabulary:
                errors.append(f"{label}: unknown {key} {form[key]!r}")
        if not isinstance(form["name"], str) or not NAME.match(form["name"]):
            errors.append(f"{label}: invalid name")
        if not isinstance(form["variant"], str) or not VARIANT.match(form["variant"]):
            errors.append(f"{label}: invalid variant {form['variant']!r}")
        if form["variant"] in RESERVED_VARIANTS:
            errors.append(f"{label}: variant {form['variant']!r} is reserved")
        if type(form["opcode"]) is not int or not 0 <= form["opcode"] <= U16_MAX:
            errors.append(f"{label}: invalid opcode")
        reads_vcc = form["operands"] == "vop2_vcc"
        wave = "wave32" if reads_vcc else WAVE_BY_KIND.get(_rule_kind(form), "any")
        if (form["carry_in"] == "vcc") != reads_vcc:
            errors.append(f"{label}: implicit VCC input requires the vop2_vcc rule")
        if form["wave"] in WAVE and form["wave"] != wave:
            errors.append(f"{label}: wave {form['wave']} is not enforced for this rule (requires {wave})")
        lane_mask = _rule_kind(form) == "vector_compare" or (reads_vcc and form["operation"] != "cselect")
        vcc = "lane_mask" if lane_mask else "unchanged"
        if form["vcc"] in MASK and form["vcc"] != vcc:
            errors.append(
                f"{label}: vcc {form['vcc']} contradicts rule {form['operands']} "
                f"(requires {vcc})"
            )
        if form["exec"] != "unchanged":
            # The EXEC-writing forms are the Wave32 `_B32` SAVEEXEC and WREXEC
            # families of guide 12.3, printed pages 117-122. Each reads EXEC_LO,
            # writes the combined result back to EXEC_LO and sets SCC.
            if form["family"] != "SOP1" or form["width"] != 32:
                errors.append(f"{label}: exec {form['exec']} is only defined for 32-bit SOP1 forms")
            elif form["operands"] != "sop1_saveexec":
                errors.append(f"{label}: exec {form['exec']} requires the sop1_saveexec rule")
            elif form["operation"] not in EXEC_COMBINE:
                errors.append(
                    f"{label}: exec {form['exec']} requires one of the EXEC-combining operations"
                )
            elif form["scc"] != "nonzero":
                errors.append(f"{label}: exec {form['exec']} requires the nonzero SCC effect")
        expected_width = (_implemented_rule(form) or {}).get("width", 32)
        if type(form["width"]) is not int or form["width"] != expected_width:
            errors.append(f"{label}: width {form['width']} is not supported")
        for key, value in (
            ("name", form["name"]),
            ("opcode", (form["family"], form["opcode"])),
            # Variants are names in their operand rule's enum.
            ("variant", (_rule_enum(form), form["variant"])),
        ):
            if value in seen[key]:
                shown = form["variant"] if key == "variant" else value
                errors.append(f"{label}: duplicate {key} {shown}")
            seen[key].add(value)
        if (
            # #818: an image sample operation carries no ALU signedness.
            form["operation"] not in IMAGE_OPERATIONS
            and (
                (form["operation"] in LOGICAL and form["signedness"] != "bitwise")
                or (form["operation"] in ARITHMETIC and form["signedness"] == "bitwise")
                or (form["operation"] in FLOAT_ROUNDING and form["signedness"] != "float")
                # #784: float signedness also admits the VOP2 binary32 arithmetic.
                or (form["signedness"] == "float" and form["operation"] not in FLOAT_OPERATIONS)
            )
        ):
            errors.append(
                f"{label}: signedness {form['signedness']} contradicts operation {form['operation']}"
            )
        if form["operation"] in SHIFTS:
            bits = form.get(SHIFT_KEY)
            if type(bits) is not int or not 0 < bits < 32 or (1 << bits) != form["width"]:
                errors.append(
                    f"{label}: {SHIFT_KEY} must be log2 of the width ({form['width']}), got {bits!r}"
                )
        elif SHIFT_KEY in form:
            errors.append(f"{label}: {SHIFT_KEY} on a non-shift operation")
        if form["operation"] == "lshl_add":
            amount = form.get(SHIFT_AMOUNT_KEY)
            if type(amount) is not int or amount not in SHIFT_AMOUNTS:
                errors.append(
                    f"{label}: {SHIFT_AMOUNT_KEY} must be one of {sorted(SHIFT_AMOUNTS)}, got {amount!r}"
                )
        elif SHIFT_AMOUNT_KEY in form:
            errors.append(f"{label}: {SHIFT_AMOUNT_KEY} on a non-lshl_add operation")
        if SOURCE_ORDER_KEY in form:
            order = form[SOURCE_ORDER_KEY]
            if not isinstance(order, str) or order not in SOURCE_ORDERS:
                errors.append(f"{label}: unknown {SOURCE_ORDER_KEY} {order!r}")
            elif _rule_kind(form) != "vector":
                errors.append(f"{label}: {SOURCE_ORDER_KEY} on a non-vector form")
        if not reads_vcc and form["operation"] in SCC_READERS and form["carry_in"] != "scc":
            errors.append(f"{label}: operation {form['operation']} reads SCC (requires carry_in scc)")
        if form["operation"] == "sext":
            bits = form.get(SOURCE_BITS_KEY)
            if type(bits) is not int or bits not in SOURCE_BITS:
                errors.append(f"{label}: {SOURCE_BITS_KEY} must be one of {sorted(SOURCE_BITS)}, got {bits!r}")
        elif SOURCE_BITS_KEY in form:
            errors.append(f"{label}: {SOURCE_BITS_KEY} on a non-sext operation")
        if form["operation"] == "compare":
            relation = form.get(RELATION_KEY)
            if not isinstance(relation, str) or relation not in RELATIONS:
                errors.append(f"{label}: unknown or missing {RELATION_KEY} {relation!r}")
            elif (relation, form["signedness"]) not in COMPARE_PRIMITIVES:
                errors.append(f"{label}: no primitive implements {relation}/{form['signedness']}")
            elif relation in VECTOR_ONLY_RELATIONS and _rule_kind(form) != "vector_compare":
                errors.append(f"{label}: relation {relation} exists only for VOPC compares")
            elif _rule_kind(form) == "immediate_compare":
                base = SOPK_COMPARE_BASES.get(form["signedness"])
                expected = None if base is None else base + SOPK_RELATION_ORDER.index(relation) \
                    if relation in SOPK_RELATION_ORDER else None
                if form["opcode"] != expected:
                    errors.append(
                        f"{label}: relation {relation}/{form['signedness']} has SOPK opcode "
                        f"{expected} (guide 13.1.2), not {form['opcode']}"
                    )
            elif _rule_kind(form) == "vector_compare":
                base = VOPC_INTEGER_BASES.get(expected_width, {}).get(form["signedness"])
                expected = None if base is None else base + VOPC_RELATION_OFFSETS[relation]
                if form["opcode"] != expected:
                    errors.append(
                        f"{label}: relation {relation}/{form['signedness']} has VOPC opcode "
                        f"{expected} (guide Table 59), not {form['opcode']}"
                    )
            # SOPC compares write SCC; VOPC compares write VCC and leave SCC.
            scc = "unchanged" if _rule_kind(form) == "vector_compare" else "relation"
            if form["scc"] != scc:
                errors.append(f"{label}: scc {form['scc']} contradicts compare (requires {scc})")
            if form["carry_in"] != "none":
                errors.append(f"{label}: carry_in {form['carry_in']} contradicts compare")
            primitive = (scc, None, False)
        elif form["operation"] in CONTROL_OPERATIONS:
            variant, operand_rule, _ = CONTROL_OPERATIONS[form["operation"]]
            signedness = "signed" if operand_rule == "sopp_branch" else "bitwise"
            if (form["variant"], form["operands"], form["signedness"]) != (variant, operand_rule, signedness):
                errors.append(f"{label}: control operation requires {variant}, {operand_rule}, {signedness}")
            if RELATION_KEY in form or form["width"] != 32:
                errors.append(f"{label}: control operation requires width 32 and no relation")
            primitive = ("unchanged", None, form["operation"] == "branch_scc_zero")
        elif _rule_kind(form) in ("vector", "vector_unary", "vector_swap"):
            if RELATION_KEY in form:
                errors.append(f"{label}: {RELATION_KEY} on a non-compare operation")
            key = (form["operation"], form["signedness"])
            table = {
                "vector": VCC_VECTOR_PRIMITIVES if reads_vcc else VECTOR_PRIMITIVES,
                "vector_unary": VECTOR_UNARY_PRIMITIVES,
                "vector_swap": {("swap", "bitwise"): "Swap"},
            }[_rule_kind(form)]
            lowering = table.get(key)
            primitive = None if lowering is None else ("unchanged", lowering, False)
        elif _rule_kind(form) == "image":
            if RELATION_KEY in form:
                errors.append(f"{label}: {RELATION_KEY} on an image sample operation")
            if form["operation"] not in IMAGE_OPERATIONS:
                errors.append(f"{label}: image rule requires an image sample operation")
            if form["signedness"] != "float":
                errors.append(f"{label}: image sample requires float signedness")
            # Image samples write no SCC, VCC or EXEC and have no primitive.rs
            # function; behaviour is the MIMG contract, not a shared primitive.
            primitive = ("unchanged", None, False)
        else:
            if RELATION_KEY in form:
                errors.append(f"{label}: {RELATION_KEY} on a non-compare operation")
            table = IMMEDIATE_PRIMITIVES if _rule_kind(form) == "immediate" else PRIMITIVES
            primitive = table.get((form["operation"], form["signedness"]))
        if primitive is None:
            errors.append(f"{label}: no primitive implements {form['operation']}/{form['signedness']}")
        else:
            scc, _, takes_carry = primitive
            if form["scc"] != scc:
                errors.append(
                    f"{label}: scc {form['scc']} contradicts {form['operation']}/"
                    f"{form['signedness']} (requires {scc})"
                )
            if form["carry_in"] == "scc" and not takes_carry:
                errors.append(f"{label}: carry_in scc contradicts {form['operation']}/{form['signedness']}")
        rule = rules.get(form["operands"])
        if _implemented_rule(form) and "variant_of" in _implemented_rule(form):
            errors.append(f"{label}: operand rule {form['operands']} is only for encoding variants")
        if rule is None:
            errors.append(f"{label}: unknown operand rule {form['operands']!r}")
        elif rule.get("family") != form["family"]:
            errors.append(f"{label}: operand rule {form['operands']} is for {rule.get('family')}")
        elif form["operands"] in OPERAND_RULES and OPERAND_RULES[form["operands"]]["kind"] != (
            (_rule_kind(form) if _rule_kind(form) in ("vector_compare", "immediate_compare") else "compare")
            if form["operation"] == "compare"
            else "vector_swap" if form["operation"] == "swap"
            else "vector_unary" if _rule_kind(form) == "vector_unary" and form["operation"] in UNARY
            else "unary" if form["operation"] in UNARY or (
                form["operation"] == "mov" and form["family"] == "SOP1"
            ) or form["exec"] != "unchanged"
            else "vector" if _rule_kind(form) == "vector" and form["operation"] not in UNARY
            else "control" if form["operation"] in CONTROL_OPERATIONS
            else "immediate" if _rule_kind(form) == "immediate"
            else "image" if form["operation"] in IMAGE_OPERATIONS
            else "alu"
        ):
            errors.append(f"{label}: operation {form['operation']} does not fit rule {form['operands']}")
        _check_spec(label, form["spec"], documents, errors)
        verify = form["verify"]
        if not isinstance(verify, list) or not all(isinstance(item, str) for item in verify):
            errors.append(f"{label}: verify must be a list of evidence names")
            verify = []
        if not verify:
            errors.append(f"{label}: no verification requirement")
        if any(item not in EVIDENCE for item in verify):
            errors.append(f"{label}: unknown verification requirement in {verify}")
        if len(set(verify)) != len(verify):
            errors.append(f"{label}: duplicate verification requirement")
        _check_xml(form, rule, snapshot, errors)
    _check_variants(registry, rules, documents, snapshot, seen, errors)
    return errors


def _well_formed(rule):
    """Whether a rule has the dst/src shape the operand checks read."""
    return (
        isinstance(rule, dict)
        and isinstance(rule.get("src"), list)
        and all(isinstance(src, dict) and isinstance(src.get("field"), str) for src in rule["src"])
        and ("dst" not in rule or (isinstance(rule["dst"], dict) and isinstance(rule["dst"].get("field"), str)))
    )


def variant_row(registry, variant, base=None):
    """The base form re-encoded: its declaration with the variant's identity."""
    if base is None:
        base = next(form for form in registry["form"] if form.get("name") == variant["of"])
    _, vcc = OPERAND_RULES[variant["operands"]].get("vcc_transition", ("unchanged", "unchanged"))
    return dict(
        base, family=variant["family"], opcode=variant["opcode"], operands=variant["operands"],
        vcc=vcc, spec=variant["spec"], verify=variant["verify"],
    )


def _check_variants(registry, rules, documents, snapshot, seen, errors):
    forms = {form.get("name"): form for form in registry.get("form", [])}
    identities = {(form.get("name"), form.get("family")) for form in registry.get("form", [])}
    variants = registry.get("encoding_variant", [])
    if not isinstance(variants, list):
        errors.append("encoding_variant must be an array of tables")
        return
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            errors.append(f"encoding variant #{index}: must be a table, got {variant!r}")
            continue
        label = f"{variant.get('of', f'variant #{index}')!s} ({variant.get('family')!s})"
        missing = sorted(VARIANT_KEYS - set(variant))
        extra = sorted(set(variant) - VARIANT_KEYS)
        if missing:
            errors.append(f"{label}: incomplete encoding variant, missing {missing}")
        if extra:
            errors.append(f"{label}: unknown keys {extra}")
        if missing:
            continue
        malformed = [key for key in ("of", "family", "operands") if not isinstance(variant[key], str)]
        if malformed:
            errors.append(f"{label}: {malformed} must be strings")
            continue
        base = forms.get(variant["of"])
        if base is None:
            errors.append(f"{label}: encoding variant of an undeclared form")
            continue
        if type(variant["opcode"]) is not int or not 0 <= variant["opcode"] <= U16_MAX:
            errors.append(f"{label}: invalid opcode")
            continue
        if variant["family"] == base.get("family"):
            errors.append(f"{label}: an encoding variant must use another family than its form")
        identity = (variant["of"], variant["family"])
        if identity in identities:
            errors.append(f"{label}: duplicate encoding of {variant['of']}")
        identities.add(identity)
        if (variant["family"], variant["opcode"]) in seen["opcode"]:
            errors.append(f"{label}: duplicate opcode {(variant['family'], variant['opcode'])}")
        seen["opcode"].add((variant["family"], variant["opcode"]))
        implemented = OPERAND_RULES.get(variant["operands"]) if isinstance(variant["operands"], str) else None
        rule = rules.get(variant["operands"])
        if rule is None or implemented is None:
            errors.append(f"{label}: unknown operand rule {variant['operands']!r}")
            continue
        if rule.get("family") != variant["family"]:
            errors.append(f"{label}: operand rule {variant['operands']} is for {rule.get('family')}")
        if implemented.get("variant_of") != _rule_kind(base) or implemented["enum"] != _rule_enum(base):
            errors.append(
                f"{label}: operand rule {variant['operands']} cannot encode a {_rule_kind(base)} form"
            )
        _check_spec(label, variant["spec"], documents, errors)
        verify = variant["verify"]
        if (
            not isinstance(verify, list) or not all(isinstance(item, str) for item in verify)
            or not verify or any(item not in EVIDENCE for item in verify)
            or len(set(verify)) != len(verify)
        ):
            errors.append(f"{label}: invalid verification requirements {verify}")
        # The XML check runs on the re-encoded declaration, so the variant's
        # operands must agree with the base form's widths, signedness and effects.
        # An incomplete base form has its own diagnostic; checking its re-encoding
        # would only fail on the missing fields.
        if FORM_KEYS - set(base):
            errors.append(f"{label}: not checked, its form {variant['of']} is incomplete")
            continue
        base_vcc, _ = implemented.get("vcc_transition", ("unchanged", "unchanged"))
        if base["vcc"] != base_vcc:
            errors.append(
                f"{label}: operand rule {variant['operands']} requires a base "
                f"destination contract of {base_vcc}, got {base['vcc']}"
            )
        # The same base row as the completeness check above (with duplicate
        # declarations, the dict keeps the last one).
        _check_xml(variant_row(registry, variant, base), rule, snapshot, errors)
        # Explicit destinations and sources keep the base encoding's data format
        # in the same role. The implicit-to-explicit mask transition is checked
        # above and in _check_xml for each encoding.
        base_xml = snapshot["variants"].get((base["name"], _family_encoding(base["family"])))
        own_xml = snapshot["variants"].get((variant["of"], _family_encoding(variant["family"])))
        base_rule = rules.get(base.get("operands"))
        if base_xml and own_xml and _well_formed(base_rule) and _well_formed(rule):
            # Operands correspond by role, in rule order: the destination, then
            # each source (VSRC1 of VOP2 is SRC1 of VOP3).
            def roles(rule, record, explicit_dst):
                by_field = {row[0]: row[1] for row in record["operands"] if row[0]}
                fields = [rule["dst"]["field"]] if explicit_dst else []
                fields += [src["field"] for src in rule.get("src", [])]
                return [(field, by_field.get(field)) for field in fields]

            explicit_dst = "dst" in base_rule and "dst" in rule
            pairs = zip(roles(base_rule, base_xml, explicit_dst), roles(rule, own_xml, explicit_dst))
            for (base_field, base_format), (field, data_format) in pairs:
                if base_format != data_format:
                    errors.append(
                        f"{label}: XML {field} format {data_format} differs from the form's "
                        f"{base_field} format {base_format}"
                    )


def _implemented_rule(form):
    operands = form.get("operands")
    return OPERAND_RULES.get(operands) if isinstance(operands, str) else None


def _rule_kind(form):
    """The implemented kind of the form's operand rule, or None if unknown."""
    rule = _implemented_rule(form)
    return rule["kind"] if rule else None


def _rule_enum(form):
    """The generated enum of the form's operand rule, or None if unknown."""
    rule = _implemented_rule(form)
    return rule["enum"] if rule else None


def _check_spec(label, spec, documents, errors):
    has_guide = False
    if not isinstance(spec, list):
        errors.append(f"{label}: spec must be a list of references")
        return
    for ref in spec:
        if not isinstance(ref, dict) or set(ref) != {"document", "section", "page"}:
            errors.append(f"{label}: spec reference needs exactly document, section, page")
            continue
        if not isinstance(ref["document"], str):
            errors.append(f"{label}: spec document must be a name")
            continue
        if ref["document"] not in documents:
            errors.append(f"{label}: spec cites undeclared document {ref['document']!r}")
        if not SECTION.match(str(ref["section"])):
            errors.append(f"{label}: invalid spec section {ref['section']!r}")
        if type(ref["page"]) is not int or not 0 <= ref["page"] <= U16_MAX:
            errors.append(f"{label}: invalid spec page")
        if ref["document"] == "guide" and type(ref["page"]) is int and ref["page"] > 0:
            has_guide = True
    if not has_guide:
        errors.append(f"{label}: missing a guide section and page reference")


def render(registry, snapshot):
    """Return the complete generated Rust source. Requires a valid registry."""
    forms = registry["form"]
    # A family listed in append_after_variants is placed after the encoding
    # variants, so adding it does not move any pre-existing form or variant
    # index. Everything else keeps registry order.
    deferred = set(registry.get("append_after_variants", []))
    base_forms = [form for form in forms if form["family"] not in deferred]
    deferred_forms = [form for form in forms if form["family"] in deferred]
    enum_forms = base_forms + deferred_forms
    variants = registry.get("encoding_variant", [])
    variant_rows = [variant_row(registry, variant) for variant in variants]
    positions = {form["name"]: position for position, form in enumerate(base_forms)}
    for offset, form in enumerate(deferred_forms):
        positions[form["name"]] = len(base_forms) + len(variant_rows) + offset
    rule_names = sorted({form["operands"] for form in enum_forms})
    out = [
        "// @generated by crates/isa/generate_semantics.py from crates/isa/semantics/rdna2.toml.",
        "// Do not edit. Regenerate: python3 crates/isa/generate_semantics.py generate",
        "",
        "use crate::meta::EncodingFamily;",
        "use crate::primitive;",
        "use crate::semantics::{",
        "    CarryIn, Document, Evidence, FormSemantics, MaskEffect, Operation, OperandRule, SccEffect,",
        "    Signedness, SpecRef, VectorBinary, VectorLowering, VectorUnary, VectorVcc, Wave,",
        "};",
        "use crate::CompareRelation;",
        "",
        "/// Registry schema of the generated tables.",
        f"pub const GENERATED_SCHEMA: u32 = {registry['schema']};",
    ]
    # Rules that share a generated enum (VOP2 `vop2_vgpr` and `vop2_vgpr2`) render
    # it once, with the forms of all of them in registry order.
    rendered = set()
    for rule_name in rule_names:
        enum = OPERAND_RULES[rule_name]["enum"]
        if enum in rendered:
            continue
        rendered.add(enum)
        selected = [
            form for form in enum_forms if OPERAND_RULES[form["operands"]]["enum"] == enum
        ]
        out += _render_enum(rule_name, selected, snapshot, positions)
    variant_positions = {
        (variant["of"], variant["family"]): len(base_forms) + index
        for index, variant in enumerate(variants)
    }
    for rule_name in sorted({variant["operands"] for variant in variants}):
        selected = [variant for variant in variants if variant["operands"] == rule_name]
        out += _render_variants(rule_name, selected, registry, snapshot, variant_positions)
    # Encoding variants follow the forms, as rows of their own. Deferred
    # families are appended after them so pre-existing indices stay stable.
    forms = base_forms + variant_rows + deferred_forms
    out += [
        "",
        "/// Identity of every registry form and encoding variant; a family named in",
        "/// append_after_variants follows the encoding variants to keep existing",
        "/// positions stable, and the rest keep registry order.",
        f"pub const FORM_IDS: [(&str, EncodingFamily); {len(forms)}] = [",
    ]
    for form in forms:
        out.append(f"    (\"{form['name']}\", EncodingFamily::{_rust_family(form['family'])}),")
    out += [
        "];",
        "",
        "/// One validated row per registry form, in registry order.",
        f"pub static FORMS: [FormSemantics; {len(forms)}] = [",
    ]
    for form in forms:
        spec = ", ".join(
            f"SpecRef {{ document: Document::{DOCUMENTS[ref['document']]}, "
            f"section: \"{ref['section']}\", page: {ref['page']} }}"
            for ref in form["spec"]
        )
        verify = ", ".join(f"Evidence::{EVIDENCE[item]}" for item in form["verify"])
        out += [
            "    FormSemantics {",
            f"        name: \"{form['name']}\",",
            f"        family: EncodingFamily::{_rust_family(form['family'])},",
            f"        opcode: {form['opcode']},",
            f"        operation: Operation::{OPERATIONS[form['operation']]},",
            f"        signedness: Signedness::{SIGNEDNESS[form['signedness']]},",
            f"        carry_in: CarryIn::{CARRY_IN[form['carry_in']]},",
            f"        width: {form['width']},",
            f"        shift_count_bits: {form.get(SHIFT_KEY, 0)},",
            f"        shift_amount: {form.get(SHIFT_AMOUNT_KEY, 0)},",
            f"        swap_sources: {'true' if SOURCE_ORDER_KEY in form else 'false'},",
            f"        source_bits: {form.get(SOURCE_BITS_KEY, form['width'])},",
            "        relation: "
            + (f"Some(CompareRelation::{RELATIONS[form[RELATION_KEY]]})," if RELATION_KEY in form else "None,"),
            f"        scc: SccEffect::{SCC[form['scc']]},",
            f"        vcc: MaskEffect::{MASK[form['vcc']]},",
            f"        exec: MaskEffect::{EXEC[form['exec']]},",
            f"        wave: Wave::{WAVE[form['wave']]},",
            f"        operands: OperandRule::{OPERAND_RULES[form['operands']]['rust']},",
            f"        spec: &[{spec}],",
            f"        verify: &[{verify}],",
            "    },",
        ]
    out += ["];", ""]
    return "\n".join(out)


def _render_variants(rule_name, variants, registry, snapshot, positions):
    """One enum per variant rule: exactly its declared variants, with dispatch."""
    base_enum = OPERAND_RULES[rule_name]["enum"]
    enum = OPERAND_RULES[rule_name]["variant_enum"]
    family = variants[0]["family"]
    encoding = snapshot["encodings"][_family_encoding(family)]
    (offset, size), = dict((name, ranges) for name, ranges in encoding["fields"])["OP"]
    op_mask = ((1 << size) - 1) << offset
    fixed_mask = encoding["identifier_mask"] & ~op_mask
    fixed_bits = encoding["identifiers"][0] & fixed_mask
    if any(identifier & fixed_mask != fixed_bits for identifier in encoding["identifiers"]):
        raise ValueError(f"{family}: identifiers do not split into fixed bits and OP")
    variant_of = {form["name"]: form["variant"] for form in registry["form"]}
    rows = sorted(variants, key=lambda variant: variant["opcode"])
    suffix = ASSEMBLY_SUFFIX[family]
    out = [
        "",
        f"/// {family} encodings of registry forms (operand rule `{rule_name}`).",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        f"pub enum {enum} {{",
    ]
    for variant in variants:
        out += [
            f"    /// `{variant['of'].lower()}{suffix}` ({family} opcode {variant['opcode']}).",
            f"    {variant_of[variant['of']]},",
        ]
    out += [
        "}",
        "",
        f"impl {enum} {{",
        "    /// Every variant, in registry order.",
        f"    pub const ALL: [Self; {len(variants)}] = [",
    ]
    out += [f"        Self::{variant_of[variant['of']]}," for variant in variants]
    out += [
        "    ];",
        "",
        f"    /// Selects the variant from the first word of its {family} encoding: the",
        f"    /// encoding bits (mask 0x{fixed_mask:09_x}) and then the OP field (bits",
        f"    /// {offset}-{offset + size - 1}). `None` for undeclared opcodes.",
        "    pub const fn from_word(word: u32) -> Option<Self> {",
        f"        if word & 0x{fixed_mask:09_x} != 0x{fixed_bits:09_x} {{",
        "            return None;",
        "        }",
        f"        match (word >> {offset}) & 0x{(1 << size) - 1:x} {{",
    ]
    out += [f"            {row['opcode']} => Some(Self::{variant_of[row['of']]})," for row in rows]
    out += [
        "            _ => None,",
        "        }",
        "    }",
        "",
        "    /// The form this encodes, whose declaration it inherits.",
        f"    pub const fn base(self) -> {base_enum} {{",
        "        match self {",
    ]
    out += [
        f"            Self::{variant_of[variant['of']]} => {base_enum}::{variant_of[variant['of']]},"
        for variant in variants
    ]
    out += [
        "        }",
        "    }",
        "",
        f"    /// Assembler name of the {family} encoding, as LLVM prints it.",
        "    pub const fn assembly_name(self) -> &'static str {",
        "        match self {",
    ]
    out += [
        f"            Self::{variant_of[variant['of']]} => \"{variant['of'].lower()}{suffix}\","
        for variant in variants
    ]
    out += [
        "        }",
        "    }",
        "",
        "    /// The validated registry row of this encoding.",
        "    pub fn semantics(self) -> &'static FormSemantics {",
        "        &FORMS[match self {",
    ]
    out += [
        f"            Self::{variant_of[variant['of']]} => {positions[(variant['of'], variant['family'])]},"
        for variant in variants
    ]
    out += ["        }]", "    }", "}"]
    return out


def _render_image_enum(rule_name, forms, snapshot, positions):
    """One MIMG image-sample enum with the split OP field and the fixed 2D bits.

    MIMG spreads its 8-bit opcode across bit 0 (the high bit) and bits 24:18
    (the low seven bits), so the generic `from_word` that reads a contiguous OP
    field does not apply. The bounded scope fixes NSA, DIM, R128 and every
    cache/modifier flag; DMASK stays a variable field the decoder validates.
    """
    enum = OPERAND_RULES[rule_name]["enum"]
    family = forms[0]["family"]
    encoding = snapshot["encodings"][_family_encoding(family)]
    (offset, size), (low_offset, low_size) = dict(
        (name, ranges) for name, ranges in encoding["fields"]
    )["OP"]
    if (offset, size, low_offset, low_size) != (18, 7, 0, 1):
        raise ValueError(f"{family}: unexpected split OP field {encoding['fields']['OP']!r}")
    # The generic `from_word` reads a contiguous OP field; MIMG instead spreads
    # its opcode across bit 0 (the high bit) and bits 24:18 (the low seven
    # bits). `from_word` matches only the ENCODING bits and dispatches on the
    # opcode; the DIM, R128, NSA, DMASK and modifier flags are validated by the
    # decoder's image operand rule, so a wrong dimension stays `UnsupportedForm`.
    fixed_mask = 0xFC00_0000
    fixed_bits = 0xF000_0000
    out = [
        "",
        f"/// {family} image samples declared in the semantic registry (operand rule `{rule_name}`).",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        f"pub enum {enum} {{",
    ]
    for form in forms:
        out += [
            f"    /// `{form['name'].lower()}` ({family} opcode {form['opcode']}).",
            f"    {form['variant']},",
        ]
    by_opcode = {form["opcode"]: form["variant"] for form in forms}
    out += [
        "}",
        "",
        f"impl {enum} {{",
        f"    /// Every operation, in registry order.",
        f"    pub const ALL: [Self; {len(forms)}] = [",
    ]
    out += [f"        Self::{form['variant']}," for form in forms]
    out += [
        "    ];",
        "",
        f"    /// Selects the operation from the first word of its {family} encoding:",
        f"    /// the encoding bits (mask 0x{fixed_mask:09_x}) and then the split OP field",
        f"    /// (bits {offset}-{offset + size - 1} for the low bits and bit",
        f"    /// {low_offset} for the high bit). `None` for undeclared opcodes. The DIM,",
        "    /// R128, NSA, DMASK and modifier flags are checked separately by the decoder.",
        "    pub const fn from_word(word: u32) -> Option<Self> {",
        f"        if word & 0x{fixed_mask:09_x} != 0x{fixed_bits:09_x} {{",
        "            return None;",
        "        }",
        f"        let opcode = ((word >> {offset}) & 0x{(1 << size) - 1:x})",
        f"            | ((word & 0x{(1 << low_size) - 1:x}) << {size});",
        "        match opcode {",
    ]
    for form in sorted(forms, key=lambda form: form["opcode"]):
        out.append(f"            {form['opcode']} => Some(Self::{form['variant']}),")
    out += [
        "            _ => None,",
        "        }",
        "    }",
        "",
        "    /// Lower-case mnemonic.",
        "    pub const fn mnemonic(self) -> &'static str {",
        "        match self {",
    ]
    out += [f"            Self::{form['variant']} => \"{form['name'].lower()}\"," for form in forms]
    out += [
        "        }",
        "    }",
        "",
        "    /// The validated registry row.",
        "    pub fn semantics(self) -> &'static FormSemantics {",
        "        &FORMS[match self {",
    ]
    out += [f"            Self::{form['variant']} => {positions[form['name']]}," for form in forms]
    out += [
        "        }]",
        "    }",
        "}",
    ]
    return out


def _render_enum(rule_name, forms, snapshot, positions):
    enum = OPERAND_RULES[rule_name]["enum"]
    kind = OPERAND_RULES[rule_name]["kind"]
    if kind == "image":
        return _render_image_enum(rule_name, forms, snapshot, positions)
    rules = sorted({form["operands"] for form in forms})
    if any(OPERAND_RULES[name]["kind"] != kind for name in rules):
        raise ValueError(f"{enum}: operand rules {rules} of one enum differ in kind")
    table = re.sub(r"(?<!^)(?=[A-Z])", "_", enum).upper() + "_BY_OPCODE"
    family = forms[0]["family"]
    encoding = snapshot["encodings"][_family_encoding(family)]
    mask = encoding["identifier_mask"]
    (offset, size), = dict((name, ranges) for name, ranges in encoding["fields"])["OP"]
    op_mask = ((1 << size) - 1) << offset
    fixed_mask = mask & ~op_mask
    fixed_bits = encoding["identifiers"][0] & fixed_mask
    # Every identifier of the family must share the fixed bits, or the split
    # into an encoding test plus an OP match would change which words match.
    if any(identifier & fixed_mask != fixed_bits for identifier in encoding["identifiers"]) or (
        mask & op_mask != op_mask
    ):
        raise ValueError(f"{family}: identifiers do not split into fixed bits and OP")
    out = [
        "",
        f"/// {family} operations declared in the semantic registry (operand rule"
        f"{'s' if len(rules) > 1 else ''} {', '.join(f'`{name}`' for name in rules)}).",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        f"pub enum {enum} {{",
    ]
    for form in forms:
        carry = {"scc": " plus SCC", "vcc": " with VCC input"}.get(form["carry_in"], "")
        what = f"compare {form[RELATION_KEY]}" if RELATION_KEY in form else form["operation"]
        effect = f"VCC = {form['vcc']}" if form["vcc"] != "unchanged" else f"SCC = {form['scc']}"
        out += [
            f"    /// `{form['name'].lower()}` ({family} opcode {form['opcode']}): "
            f"{form['signedness']} {what}{carry}, {effect}.",
            f"    {form['variant']},",
        ]
    by_opcode = {form["opcode"]: form["variant"] for form in forms}
    out += [
        "}",
        "",
        f"/// {enum} by {family} OP field value; `None` for undeclared opcodes.",
        f"const {table}: [Option<{enum}>; {1 << size}] = [",
    ]
    out += [
        f"    Some({enum}::{by_opcode[opcode]})," if opcode in by_opcode else "    None,"
        for opcode in range(1 << size)
    ]
    out += [
        "];",
        "",
        f"impl {enum} {{",
        "    /// Every operation, in registry order.",
        f"    pub const ALL: [Self; {len(forms)}] = [",
    ]
    out += [f"        Self::{form['variant']}," for form in forms]
    out += [
        "    ];",
        "",
        f"    /// Selects the operation from a first instruction word: the {family} encoding",
        f"    /// bits (mask 0x{fixed_mask:09_x}) and then the OP field (bits {offset}-{offset + size - 1}).",
        "    pub const fn from_word(word: u32) -> Option<Self> {",
        f"        if word & 0x{fixed_mask:09_x} != 0x{fixed_bits:09_x} {{",
        "            return None;",
        "        }",
        f"        {table}[((word >> {offset}) & 0x{(1 << size) - 1:x}) as usize]",
        "    }",
        "",
        "    /// Lower-case mnemonic.",
        "    pub const fn mnemonic(self) -> &'static str {",
        "        match self {",
    ]
    out += [f"            Self::{form['variant']} => \"{form['name'].lower()}\"," for form in forms]
    out += [
        "        }",
        "    }",
        "",
        "    /// The validated registry row.",
        "    pub fn semantics(self) -> &'static FormSemantics {",
        "        &FORMS[match self {",
    ]
    out += [f"            Self::{form['variant']} => {positions[form['name']]}," for form in forms]
    out += [
        "        }]",
        "    }",
    ]
    if kind == "control":
        out += [
            "",
            "    /// Exact SIMM16 required by the fixture, or None for a displacement.",
            "    pub const fn required_immediate(self) -> Option<u16> {",
            "        match self {",
        ]
        for form in forms:
            exact = OPERAND_RULES[form["operands"]]["shape"].get("immediate")
            value = "None" if exact is None else f"Some({exact:#x})"
            out.append(f"            Self::{form['variant']} => {value},")
        out += [
            "        }",
            "    }",
            "",
            "    /// Constructs the existing decoded instruction, resolving a target only for a branch.",
            "    pub fn decode<E>(",
            "        self,",
            "        target: impl FnOnce() -> Result<usize, E>,",
            "    ) -> Result<crate::Instruction, E> {",
            "        Ok(match self {",
        ]
        for form in forms:
            variant = form["variant"]
            fields = " { target: target()? }" if form["operands"] == "sopp_branch" else ""
            out.append(f"            Self::{variant} => crate::Instruction::{variant}{fields},")
        out += ["        })", "    }", "}", "", "impl crate::Instruction {",
                "    /// The registry operation corresponding to a decoded SOPP control.",
                f"    pub const fn sopp_control(self) -> Option<{enum}> {{",
                "        match self {"]
        for form in forms:
            variant = form["variant"]
            fields = " { .. }" if form["operands"] == "sopp_branch" else ""
            out.append(f"            Self::{variant}{fields} => Some({enum}::{variant}),")
        out += ["            _ => None,", "        }", "    }", "}"]
        return out
    if kind in ("vector_unary", "vector_swap"):
        out += [
            "",
            "    /// Assembler name of the 32-bit VOP1 encoding.",
            "    pub const fn assembly_name(self) -> &'static str {",
            "        match self {",
        ]
        suffix = "_e32" if kind == "vector_unary" else ""
        out += [f"            Self::{form['variant']} => \"{form['name'].lower()}{suffix}\"," for form in forms]
        out += ["        }", "    }"]
    if kind == "vector_swap":
        return out + ["}"]
    if kind == "vector_unary":
        out += [
            "",
            "    /// Per-lane integer operation selected by the registry.",
            "    pub const fn lowering(self) -> VectorUnary {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => "
            f"VectorUnary::{VECTOR_UNARY_PRIMITIVES[(form['operation'], form['signedness'])]},"
            for form in forms
        ]
        return out + ["        }", "    }", "}"]
    if kind == "immediate":
        out += [
            "",
            "    /// Whether the operation reads SCC (the conditional move's condition).",
            "    pub const fn reads_scc(self) -> bool {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => {'true' if form['carry_in'] == 'scc' else 'false'},"
            for form in forms
        ]
        out += [
            "        }",
            "    }",
            "",
            "    /// Whether the operation writes SCC.",
            "    pub const fn writes_scc(self) -> bool {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => {'false' if form['scc'] == 'unchanged' else 'true'},"
            for form in forms
        ]
        out += [
            "        }",
            "    }",
            "",
            "    /// Whether the operation always reads the old SDST (`D = D op imm`).",
            "    pub const fn reads_dst(self) -> bool {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => {'true' if form['operation'] in ('add', 'mul') else 'false'},"
            for form in forms
        ]
        out += [
            "        }",
            "    }",
            "",
            "    /// The new SDST (`None` leaves it unchanged) and, for forms that write it, the",
            "    /// new SCC (`None` leaves SCC unchanged). `dst` is the old SDST (ignored by",
            "    /// forms that do not read it) and `scc_in` the SCC read by the conditional move.",
            "    pub const fn evaluate(self, dst: u32, simm16: u16, scc_in: bool) -> (Option<u32>, Option<bool>) {",
            "        match self {",
        ]
        for form in forms:
            _, function, _ = IMMEDIATE_PRIMITIVES[(form["operation"], form["signedness"])]
            immediate = "primitive::sext_simm16(simm16)"
            variant = form["variant"]
            if form["operation"] == "mov":
                out.append(f"            Self::{variant} => (Some({immediate}), None),")
            elif form["operation"] == "cmov":
                out.append(
                    f"            Self::{variant} => (if scc_in {{ Some({immediate}) }} else {{ None }}, None),"
                )
            elif form["scc"] == "unchanged":
                out.append(f"            Self::{variant} => (Some(primitive::{function}(dst, {immediate})), None),")
            else:
                out += [
                    f"            Self::{variant} => {{",
                    f"                let (value, scc) = primitive::{function}(dst, {immediate});",
                    "                (Some(value), Some(scc))",
                    "            }",
                ]
        out += ["        }", "    }", "}"]
        return out
    if kind == "immediate_compare":
        out += [
            "",
            "    /// The relation tested, with SDST (S0) on the left.",
            "    pub const fn relation(self) -> CompareRelation {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => CompareRelation::{RELATIONS[form[RELATION_KEY]]},"
            for form in forms
        ]
        out += [
            "        }",
            "    }",
            "",
            "    /// The SCC: whether the relation holds for SDST and SIMM16, which signed forms",
            "    /// sign-extend and unsigned forms zero-extend.",
            "    pub const fn test(self, dst: u32, simm16: u16) -> bool {",
            "        match self {",
        ]
        for form in forms:
            function = COMPARE_PRIMITIVES[(form[RELATION_KEY], form["signedness"])]
            immediate = "primitive::sext_simm16(simm16)" if form["signedness"] == "signed" else "simm16 as u32"
            out.append(f"            Self::{form['variant']} => primitive::{function}(dst, {immediate}),")
        out += ["        }", "    }", "}"]
        return out
    if kind == "unary":
        out += [
            "",
            "    /// Whether the operation writes SCC.",
            "    pub const fn writes_scc(self) -> bool {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => {'false' if form['scc'] == 'unchanged' else 'true'},"
            for form in forms
        ]
        out += [
            "        }",
            "    }",
            "",
            "    /// Whether the operation writes the EXEC mask.",
            "    pub const fn writes_exec(self) -> bool {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => {'false' if form['exec'] == 'unchanged' else 'true'},"
            for form in forms
        ]
        out += [
            "        }",
            "    }",
            "",
            "    /// The destination value, the new EXEC mask and, for forms that write it,",
            "    /// the new SCC (`None` leaves SCC unchanged).",
            "    ///",
            "    /// EXEC is an implicit operand of the forms that write it: they combine",
            "    /// the source with the current mask. For every other form the mask is",
            "    /// returned unchanged.",
            "    pub const fn evaluate(self, src: u32, exec: u32) -> (u32, u32, Option<bool>) {",
            "        match self {",
        ]
        for form in forms:
            _, function, _ = PRIMITIVES[(form["operation"], form["signedness"])]
            if form["exec"] != "unchanged":
                # The combine primitives take the source and the mask, and return
                # the combined mask with the flag for it.
                combine = f"primitive::{function}(src, exec)"
                if form["exec"] == "exec_save":
                    # The destination receives the previous mask.
                    destination = "exec"
                else:
                    # The destination receives the new mask.
                    destination = "value"
                out += [
                    f"            Self::{form['variant']} => {{",
                    f"                let (value, scc) = {combine};",
                    f"                ({destination}, value, Some(scc))",
                    "            }",
                ]
                continue
            if form["operation"] == "sext":
                value = f"primitive::{function}(src, {form[SOURCE_BITS_KEY]})"
            else:
                value = f"primitive::{function}(src)"
            if form["scc"] == "unchanged":
                out.append(f"            Self::{form['variant']} => ({value}, exec, None),")
            else:
                out += [
                    f"            Self::{form['variant']} => {{",
                    f"                let (value, scc) = {value};",
                    "                (value, exec, Some(scc))",
                    "            }",
                ]
        out += ["        }", "    }", "}"]
        return out
    if kind == "vector":
        out += [
            "",
            "    /// Assembler name of the 32-bit (VOP2) encoding, as LLVM prints it.",
            "    pub const fn assembly_name(self) -> &'static str {",
            "        match self {",
        ]
        out += [f"            Self::{form['variant']} => \"{form['name'].lower()}_e32\"," for form in forms]
        out += [
            "        }",
            "    }",
            "",
            "    /// The per-lane operation its MSL lowering uses.",
            "    pub const fn lowering(self) -> VectorLowering {",
            "        match self {",
        ]
        for form in forms:
            key = (form["operation"], form["signedness"])
            if form["operands"] == "vop2_vcc":
                lowering = f"Vcc(VectorVcc::{VCC_VECTOR_PRIMITIVES[key]})"
            else:
                lowering = f"Binary(VectorBinary::{VECTOR_PRIMITIVES[key]})"
            out.append(f"            Self::{form['variant']} => VectorLowering::{lowering},")
        out += [
            "        }",
            "    }",
            "",
            "    /// Whether the operation reads its sources reversed (`D = op(S1, S0)`).",
            "    pub const fn swaps_sources(self) -> bool {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => {'true' if SOURCE_ORDER_KEY in form else 'false'},"
            for form in forms
        ]
        out += [
            "        }",
            "    }",
            "",
            "    /// The operand rule the decoder applies to the 32-bit encoding.",
            "    pub const fn operand_rule(self) -> OperandRule {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => OperandRule::{OPERAND_RULES[form['operands']]['rust']},"
            for form in forms
        ]
        out += ["        }", "    }", "}"]
        return out
    if kind == "vector_compare":
        out += [
            "",
            "    /// Assembler name of the 32-bit (VOPC) encoding, as LLVM prints it.",
            "    pub const fn assembly_name(self) -> &'static str {",
            "        match self {",
        ]
        out += [f"            Self::{form['variant']} => \"{form['name'].lower()}_e32\"," for form in forms]
        out += [
            "        }",
            "    }",
            "",
            "    /// Whether the operands compare as two's-complement values.",
            "    pub const fn signed(self) -> bool {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => {'true' if form['signedness'] == 'signed' else 'false'},"
            for form in forms
        ]
        out += ["        }", "    }"]
    if kind in ("compare", "vector_compare"):
        result = "SCC" if kind == "compare" else "per-lane result"
        has_pairs = any(form["width"] == 64 for form in forms)
        out += [
            "",
            "    /// The relation tested, with S0 on the left.",
            "    pub const fn relation(self) -> CompareRelation {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => CompareRelation::{RELATIONS[form[RELATION_KEY]]},"
            for form in forms
        ]
        out += [
            "        }",
            "    }",
            "",
            f"    /// The {result}: whether the relation holds for S0 and S1.",
            "    pub const fn test(self, src0: u32, src1: u32) -> bool {",
        ]
        if has_pairs:
            out += [
                "        self.test_bits(src0 as u64, src1 as u64)",
                "    }",
                "",
                "    /// Test raw operand bits at the width declared by this VOPC form.",
                "    pub const fn test_bits(self, src0: u64, src1: u64) -> bool {",
            ]
        out += ["        match self {"]
        for form in forms:
            relation = form[RELATION_KEY]
            if form["width"] in (16, 64) and relation not in VECTOR_ONLY_RELATIONS:
                symbol = {"lt": "<", "eq": "==", "le": "<=", "gt": ">", "lg": "!=", "ge": ">="}[relation]
                width = form["width"]
                if width == 64 and form["signedness"] == "unsigned":
                    left, right = "src0", "src1"
                else:
                    cast = f"{'i' if form['signedness'] == 'signed' else 'u'}{width}"
                    left, right = f"(src0 as {cast})", f"(src1 as {cast})"
                expression = f"{left} {symbol} {right}"
            else:
                function = COMPARE_PRIMITIVES[(relation, form["signedness"])]
                args = "src0 as u32, src1 as u32" if has_pairs else "src0, src1"
                expression = f"primitive::{function}({args})"
            out.append(f"            Self::{form['variant']} => {expression},")
        out += ["        }", "    }", "}"]
        return out
    out += [
        "",
        "    /// Whether the operation reads SCC (carry-in, borrow-in or select condition).",
        "    pub const fn reads_scc(self) -> bool {",
        "        match self {",
    ]
    out += [
        f"            Self::{form['variant']} => {'true' if form['carry_in'] == 'scc' else 'false'},"
        for form in forms
    ]
    out += [
        "        }",
        "    }",
        "",
        "    /// Whether the operation writes SCC.",
        "    pub const fn writes_scc(self) -> bool {",
        "        match self {",
    ]
    out += [
        f"            Self::{form['variant']} => {'false' if form['scc'] == 'unchanged' else 'true'},"
        for form in forms
    ]
    out += [
        "        }",
        "    }",
        "",
        "    /// The result and, for forms that write it, the new SCC (`None` leaves SCC",
        "    /// unchanged). `scc_in` is the SCC read by forms that consume it (ignored by",
        "    /// the others).",
        "    pub const fn evaluate(self, src0: u32, src1: u32, scc_in: bool) -> (u32, Option<bool>) {",
        "        match self {",
    ]
    for form in forms:
        _, function, takes_carry = PRIMITIVES[(form["operation"], form["signedness"])]
        if takes_carry:
            argument = "scc_in" if form["carry_in"] == "scc" else "false"
            call = f"primitive::{function}(src0, src1, {argument})"
        elif form["operation"] in SHIFTS:
            mask = (1 << form[SHIFT_KEY]) - 1
            call = f"primitive::{function}(src0, src1 & 0x{mask:x})"
        elif form["operation"] == "lshl_add":
            call = f"primitive::{function}(src0, src1, {form[SHIFT_AMOUNT_KEY]})"
        else:
            call = f"primitive::{function}(src0, src1)"
        if form["scc"] == "unchanged":
            out.append(f"            Self::{form['variant']} => ({call}, None),")
        else:
            out += [
                f"            Self::{form['variant']} => {{",
                f"                let (value, scc) = {call};",
                "                (value, Some(scc))",
                "            }",
            ]
    out += ["        }", "    }", "}"]
    return out


def _display(path):
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", choices=("generate", "validate"))
    parser.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=RUST)
    arguments = parser.parse_args()
    if bool(arguments.mode) == arguments.check:
        parser.error("choose exactly one of generate, validate or --check")
    try:
        registry = load_registry(arguments.registry)
    except RegistryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    snapshot = load_snapshot(arguments.snapshot)
    errors = validate(registry, snapshot)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    count = len(registry["form"])
    if arguments.mode == "validate":
        print(f"registry valid: {count} forms")
        return 0
    text = render(registry, snapshot)
    if arguments.check:
        current = arguments.output.read_text(encoding="utf-8") if arguments.output.exists() else ""
        if current != text:
            print(f"{_display(arguments.output)} is stale; run generate", file=sys.stderr)
            return 1
        print(f"registry valid and generated code current: {count} forms")
        return 0
    arguments.output.write_text(text, encoding="utf-8")
    print(f"wrote {_display(arguments.output)}: {count} forms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
