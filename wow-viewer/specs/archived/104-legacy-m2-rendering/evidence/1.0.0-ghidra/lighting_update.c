
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

void __thiscall FUN_0071c9d0(int param_1,float param_2,float param_3)

{
  float fVar1;
  float fVar2;
  float fVar3;
  
  if (param_2 < param_3 == (NAN(param_2) || NAN(param_3))) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f88c,0x92,s_start_<_end_0083f8c0,
                 0,1);
  }
  *(undefined4 *)(param_1 + 0x54) = 0;
  fVar1 = _DAT_007cba60 / (param_2 * param_2);
  fVar3 = (param_2 + param_3) * _DAT_007cbad0;
  fVar2 = fVar3 * fVar3 * fVar1;
  fVar2 = (_DAT_007cd8fc - fVar2) / (fVar3 - fVar2 * param_2);
  *(float *)(param_1 + 0x58) = fVar2;
  *(float *)(param_1 + 0x5c) = (_DAT_007cba60 - fVar2 * param_2) * fVar1;
  return;
}


