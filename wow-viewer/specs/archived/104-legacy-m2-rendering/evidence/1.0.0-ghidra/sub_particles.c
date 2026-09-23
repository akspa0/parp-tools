
undefined4 __thiscall FUN_0070ef60(int param_1,uint param_2)

{
  if (*(int *)(param_1 + 0x10) == 0) {
    FUN_0070b920(s_GetEmitter_0083f648);
  }
  if (*(uint *)(*(int *)(*(int *)(param_1 + 0x2c) + 0x130) + 0x13c) <= param_2) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083ed44,0xd38,
                 s_particleIndex_<_m_shared_>m_data_0083f614,0,1);
    return *(undefined4 *)(*(int *)(param_1 + 0x3a8) + param_2 * 4);
  }
  return *(undefined4 *)(*(int *)(param_1 + 0x3a8) + param_2 * 4);
}


