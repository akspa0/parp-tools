
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

undefined4 __fastcall FUN_00686d40(uint param_1)

{
  uint uVar1;
  uint uVar2;
  undefined4 *puVar3;
  int iVar4;
  int *piVar5;
  int iVar6;
  undefined1 local_124 [256];
  undefined **local_24;
  undefined4 local_20;
  undefined4 **local_1c;
  uint local_18;
  float local_14;
  uint local_10;
  int iStack_c;
  undefined1 local_5;
  
  if (0xb < param_1) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00834f64,0x1bf,
                 s_liquid_<_LIQUID_COUNT_00834fc8,0,1);
  }
  uVar1 = ftol();
  uVar2 = FUN_0042a920();
  iVar6 = 0;
  local_14 = ((float)(uVar2 % uVar1) / (float)(int)uVar1) * _DAT_007d1ac8;
  iStack_c = (int)ROUND(local_14 - _DAT_00834d48);
  if ((&DAT_00a871e4)[param_1] == '\0') {
    local_5 = 1;
    piVar5 = &DAT_00a7e608 + param_1 * 0x1e;
    local_10 = uVar1;
    do {
      if (*piVar5 == 0) {
        local_1c = &local_1c;
        local_18 = (uint)&local_1c | 1;
        local_20 = 8;
        local_24 = &PTR_FUN_007cbaa4;
        if ((&PTR_s_XTextures_river_lake_a__d_blp_00834d4c)[param_1] == (undefined *)0x0) {
          FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00834f64,0x1cf,
                       s_liquidTexBaseName_liquid__00834fac,0,1);
        }
        FUN_0063fc00(local_124,0x100,(&PTR_s_XTextures_river_lake_a__d_blp_00834d4c)[param_1],
                     iVar6 + 1);
        puVar3 = (undefined4 *)FUN_0058df10(3,1,1,0,0,0,DAT_00a601dc);
        iVar4 = FUN_0044dad0(*puVar3,1);
        *piVar5 = iVar4;
        FUN_004466c0();
        FUN_0040faf0();
      }
      iVar4 = FUN_0044ecb0(0);
      if (iVar4 == 0) {
        local_5 = 0;
      }
      iVar6 = iVar6 + 1;
      piVar5 = piVar5 + 1;
    } while (iVar6 < 0x1e);
    (&DAT_00a871e4)[param_1] = local_5;
  }
  *(undefined4 *)(&DAT_00a7e5d4 + param_1 * 4) = DAT_00a78b18;
  return (&DAT_00a7e608)[param_1 * 0x1e + iStack_c];
}


