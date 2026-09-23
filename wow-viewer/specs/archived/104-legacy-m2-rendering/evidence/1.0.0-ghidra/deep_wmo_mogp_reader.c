
void __fastcall FUN_006c5380(int param_1)

{
  int *piVar1;
  int iVar2;
  int *piVar3;
  int *piVar4;
  int iVar5;
  uint uVar6;
  
  piVar1 = *(int **)(param_1 + 0x14c);
  if (*piVar1 != 0x4d564552) {
    FUN_0063d520(0x85100000,s__C__build_buildWoW_WoW_Source_Wo_00838daf + 1,0x19d,
                 s_iffChunk_>token_____MVER__00838f04,0,1);
  }
  if (piVar1[2] != 0x11) {
    FUN_0063d520(0x85100000,s__C__build_buildWoW_WoW_Source_Wo_00838daf + 1,0x1a2,
                 s__version____0x0011_00838e48,0,1);
  }
  if (piVar1[3] != 0x4d4f4750) {
    FUN_0063d520(0x85100000,s__C__build_buildWoW_WoW_Source_Wo_00838daf + 1,0x1a9,
                 s_iffChunk_>token___MOGP__00838eec,0,1);
  }
  if (*(int *)(param_1 + 0x168) != 0) {
    FUN_0063d520(0x85100000,s__C__build_buildWoW_WoW_Source_Wo_00838daf + 1,0x1ad,
                 s_lameAssLink_IsLinked______0_00838ed0,0,1);
  }
  if (*(int *)(param_1 + 0x154) == 0) {
    FUN_0063d520(0x85100000,s__C__build_buildWoW_WoW_Source_Wo_00838daf + 1,0x1ae,s_parent_00838ec8,
                 0,1);
  }
  iVar5 = *(int *)(param_1 + 0x154);
  iVar2 = *(int *)(*(int *)(iVar5 + 0x1e4) + param_1);
  piVar3 = (int *)(*(int *)(iVar5 + 0x1e4) + param_1);
  if (iVar2 != 0) {
    uVar6 = piVar3[1];
    if (((uVar6 & 1) == 0) && (uVar6 != 0)) {
      piVar4 = (int *)((int)piVar3 + (uVar6 - *(int *)(iVar2 + 4)));
    }
    else {
      piVar4 = (int *)(uVar6 & 0xfffffffe);
    }
    *piVar4 = iVar2;
    *(int *)(*piVar3 + 4) = piVar3[1];
    *piVar3 = 0;
    piVar3[1] = 0;
  }
  iVar2 = *(int *)(iVar5 + 0x1e8);
  *piVar3 = iVar2;
  piVar3[1] = *(int *)(iVar2 + 4);
  *(int *)(iVar2 + 4) = param_1;
  *(int **)(iVar5 + 0x1e8) = piVar3;
  *(int *)(param_1 + 0xbc) = *(int *)(*(int *)(param_1 + 0x154) + 0x128) + piVar1[5];
  *(int *)(param_1 + 0x10) = piVar1[7];
  piVar3 = piVar1 + 8;
  piVar4 = (int *)(param_1 + 0x14);
  for (iVar5 = 6; iVar5 != 0; iVar5 = iVar5 + -1) {
    *piVar4 = *piVar3;
    piVar3 = piVar3 + 1;
    piVar4 = piVar4 + 1;
  }
  *(uint *)(param_1 + 0x2c) = (uint)*(ushort *)(piVar1 + 0xe);
  *(uint *)(param_1 + 0x30) = (uint)*(ushort *)((int)piVar1 + 0x3a);
  *(short *)(param_1 + 0x3c) = (short)piVar1[0xf];
  *(undefined2 *)(param_1 + 0x3e) = *(undefined2 *)((int)piVar1 + 0x3e);
  *(short *)(param_1 + 0x40) = (short)piVar1[0x10];
  *(int *)(param_1 + 0x34) = piVar1[0x11];
  *(int *)(param_1 + 0x38) = piVar1[0x12];
  *(int *)(param_1 + 0x148) = piVar1[0x13];
  FUN_006c55a0(piVar1 + 0x16);
  if (*(int *)(param_1 + 0x154) == 0) {
    FUN_0063d520(0x85100000,s__C__build_buildWoW_WoW_Source_Wo_00838daf + 1,0x1ca,s_mapObj_00832a10,
                 0,1);
  }
  uVar6 = 0;
  if (*(int *)(param_1 + 0x138) != 0) {
    iVar5 = 0;
    do {
      FUN_006c5080(*(undefined1 *)(iVar5 + 0x17 + *(int *)(param_1 + 0xd8)));
      uVar6 = uVar6 + 1;
      iVar5 = iVar5 + 0x18;
    } while (uVar6 < *(uint *)(param_1 + 0x138));
  }
  *(undefined1 *)(param_1 + 0x160) = 1;
  return;
}


