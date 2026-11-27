// SPDX-License-Identifier: GPL-3.0-or-later
// Wave32 MIMG 2D RGBA8 sampling grid: IMAGE_SAMPLE_LZ, DMASK 0x8 (alpha). The T#
// (s[8:15]) and S# (s[16:19]) come from the synthetic input's first twelve
// words. Each lane's packed coordinate word is input[lane]; lanes 0-11 carry the
// descriptor words and sample from them as well, so the whole wave stays active
// and the twenty cases start at lane 12.
.text
s_mov_b32 s8, 0
s_mov_b32 s9, 0
s_mov_b32 s10, 0
s_mov_b32 s11, 0
s_mov_b32 s12, 0
s_mov_b32 s13, 0
s_mov_b32 s14, 0
s_mov_b32 s15, 0
s_mov_b32 s16, 0
s_mov_b32 s17, 0
s_mov_b32 s18, 0
s_mov_b32 s19, 0
s_buffer_load_dwordx4 s[8:11], s[0:3], 0x0
s_buffer_load_dwordx4 s[12:15], s[0:3], 0x10
s_buffer_load_dwordx4 s[16:19], s[0:3], 0x20
s_waitcnt lgkmcnt(0)
buffer_load_dword v1, v0, s[0:3], 0 idxen
s_waitcnt vmcnt(0)
v_mov_b32_e32 v11, 16
v_lshlrev_b32_e32 v2, v11, v1
v_mov_b32_e32 v10, 0xffff0000
v_and_b32_e32 v3, v10, v1
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
image_sample_lz v5, v[2:3], s[8:15], s[16:19] dmask:0x8 dim:SQ_RSRC_IMG_2D
buffer_store_dword v5, v0, s[4:7], 0 idxen
s_waitcnt_vscnt null, 0
s_endpgm
