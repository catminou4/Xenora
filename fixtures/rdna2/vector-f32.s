// SPDX-License-Identifier: GPL-3.0-or-later
// Wave32 VOP2 binary32 arithmetic forms (AMD RDNA 2 ISA guide 12.7).
// Add, subtract both ways and multiply the input by 1.25 (a literal), take
// the minimum against 0.75 from an SGPR and the maximum against -0 in odd lanes
// only, and combine the results.
.text
buffer_load_dword v1, v0, s[0:3], 0 idxen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v3, 0x3fa00000
v_add_f32_e32 v2, v1, v3
v_sub_f32_e32 v4, v1, v3
v_subrev_f32_e32 v5, v1, v3
v_mul_f32_e32 v6, 0x3fa00000, v1
s_mov_b32 s8, 0x3f400000
v_min_f32_e32 v7, s8, v1
v_mov_b32_e32 v8, v1
s_mov_b32 exec_lo, 0xaaaaaaaa
v_max_f32_e32 v8, 0x80000000, v1
s_mov_b32 exec_lo, -1
v_xor_b32_e32 v9, v2, v4
v_xor_b32_e32 v9, v5, v9
v_xor_b32_e32 v9, v6, v9
v_xor_b32_e32 v9, v7, v9
v_xor_b32_e32 v9, v8, v9
buffer_store_dword v9, v0, s[4:7], 0 idxen
s_waitcnt_vscnt null, 0
s_endpgm
