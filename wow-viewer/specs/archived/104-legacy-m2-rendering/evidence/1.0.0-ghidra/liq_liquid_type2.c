
void __fastcall FUN_00686ee0(uint param_1)

{
  int *piVar1;
  int iVar2;
  
  if (0xb < param_1) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00834f64,0x1e6,
                 s_liquid_<_LIQUID_COUNT_00834fc8,0,1);
  }
  piVar1 = &DAT_00a7e608 + param_1 * 0x1e;
  iVar2 = 0x1e;
  do {
    if (*piVar1 != 0) {
      FUN_004114f0();
      *piVar1 = 0;
    }
    piVar1 = piVar1 + 1;
    iVar2 = iVar2 + -1;
  } while (iVar2 != 0);
  (&DAT_00a871e4)[param_1] = 0;
  return;
}


