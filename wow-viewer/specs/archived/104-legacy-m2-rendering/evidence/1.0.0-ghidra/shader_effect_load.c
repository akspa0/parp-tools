
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

undefined4 __thiscall FUN_007213b0(int *param_1,int param_2,uint param_3)

{
  uint uVar1;
  int iVar2;
  undefined4 *puVar3;
  uint uVar4;
  uint uVar5;
  undefined4 uVar6;
  undefined4 uVar7;
  uint uVar8;
  uint local_14;
  uint local_10;
  
  if (*param_1 != 0) {
    FUN_004291a0(s_Model2__M2Initialize_called_mode_0083facc);
    return 1;
  }
  uVar1 = FUN_0042b8c0();
  if (uVar1 < 2) {
    param_1[0x400] = 0;
  }
  else {
    param_1[0x400] = param_2;
  }
  if (param_1[0x400] != 0) {
    param_1[0x402] = 0;
    param_1[0x403] = 0;
    FUN_00651680();
    FUN_00651680();
    iVar2 = FUN_00415300();
    param_1[0x406] = iVar2;
    FUN_00651690(param_1 + 0x401,s_Model2_0083fac4);
  }
  iVar2 = FUN_0058d450();
  if ((*(int *)(iVar2 + 0x5c) == 1) || (iVar2 = FUN_0058d450(), *(int *)(iVar2 + 0x5c) == 2)) {
    param_1[0x409] = param_3;
  }
  else {
    param_1[0x409] = 0;
  }
  if (param_1[0x409] != 0) {
    puVar3 = (undefined4 *)FUN_0063cb90(0xe10,s__PAVCGxVertexShader___0083faac,0xfffffffe,0);
    param_1[0x40a] = (int)puVar3;
    if (puVar3 == (undefined4 *)0x0) {
      return 0;
    }
    for (iVar2 = 900; iVar2 != 0; iVar2 = iVar2 + -1) {
      *puVar3 = 0;
      puVar3 = puVar3 + 1;
    }
    FUN_0058ef60(900);
  }
  iVar2 = FUN_0058d450();
  param_1[0x40b] = *(int *)(iVar2 + 0x7c);
  iVar2 = FUN_0058d450();
  if (*(int *)(iVar2 + 0x68) == 0) {
    iVar2 = FUN_0058d450();
    uVar6 = 1;
    uVar7 = 4;
    if (*(int *)(iVar2 + 100) == 0) {
      uVar7 = 3;
    }
  }
  else {
    uVar7 = 5;
    iVar2 = FUN_0058d450();
    uVar6 = *(undefined4 *)(iVar2 + 0x6c);
  }
  iVar2 = FUN_0063cb90(4,s___AUCGxTexFlags___0083fa74,0xfffffffe,0);
  if (iVar2 == 0) {
    iVar2 = 0;
  }
  else {
    iVar2 = FUN_0058df10(uVar7,1,1,0,1,0,uVar6);
  }
  param_1[0x40c] = iVar2;
  iVar2 = rand();
  uVar1 = rand();
  FUN_00452ff0(iVar2 << 0x10 | uVar1 & 0xffff);
  param_3 = 0;
  do {
    uVar5 = local_10 >> 8 & 0xff;
    uVar1 = local_10 >> 0x10 & 0xff;
    iVar2 = (local_10 >> 0x18) - 4;
    uVar4 = uVar1 - 0xc;
    if (iVar2 < 0) {
      iVar2 = (local_10 >> 0x18) + 0xb8;
    }
    uVar8 = uVar5 - 0x18;
    if ((int)uVar4 < 0) {
      uVar4 = uVar1 + 200;
    }
    uVar1 = (local_10 & 0xff) - 0x1c;
    if ((int)uVar8 < 0) {
      uVar8 = uVar5 + 0xd4;
    }
    if ((int)uVar1 < 0) {
      uVar1 = (local_10 & 0xff) + 0xd8;
    }
    local_14 = ((*(uint *)(&DAT_007cd6c0 + uVar8) >> 0x1d | *(uint *)(&DAT_007cd6c0 + uVar8) << 3) ^
                (*(uint *)(&DAT_007cd6c0 + uVar4) >> 0x1e | *(uint *)(&DAT_007cd6c0 + uVar4) << 2) ^
                *(uint *)(&DAT_007cd6c0 + uVar1) ^
               (*(uint *)(&DAT_007cd6c0 + iVar2) >> 0x1f | *(uint *)(&DAT_007cd6c0 + iVar2) * 2)) +
               local_14;
    *(float *)(&DAT_00aef770 + param_3) = (float)(local_14 & 0x7fffff | 0x3f800000) - _DAT_007cba60;
    param_3 = param_3 + 4;
    local_10 = ((iVar2 << 8 | uVar4) << 8 | uVar8) << 8 | uVar1;
  } while (param_3 < 0x200);
  FUN_00780ff0();
  *param_1 = 1;
  return 1;
}


