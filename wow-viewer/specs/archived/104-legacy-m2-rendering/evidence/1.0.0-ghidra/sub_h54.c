
undefined4 __fastcall FUN_0071f7a0(int param_1,uint param_2,uint *param_3)

{
  uint uVar1;
  uint uVar2;
  int iVar3;
  int iVar4;
  undefined4 uVar5;
  uint local_c;
  int local_8;
  
  uVar2 = param_3[1];
  if (param_2 < uVar2) {
LAB_0071f7b7:
    uVar5 = 0x10a;
  }
  else {
    uVar1 = *param_3;
    if (uVar1 * 0x38 + uVar2 <= param_2) {
      if (uVar1 == 0) {
        uVar2 = 0;
      }
      else {
        uVar2 = uVar2 + param_1;
      }
      param_3[1] = uVar2;
      local_c = 0;
      if (uVar1 != 0) {
        local_8 = 0;
        do {
          uVar2 = *(uint *)(param_3[1] + 8 + local_8);
          iVar4 = param_3[1] + local_8;
          if (param_2 < uVar2) goto LAB_0071f7b7;
          if (param_2 < uVar2 + *(int *)(iVar4 + 4) * 8) goto LAB_0071f7d4;
          if (*(int *)(iVar4 + 4) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 8) = iVar3;
          uVar2 = *(uint *)(iVar4 + 0x10);
          if (param_2 < uVar2) goto LAB_0071f7b7;
          if (param_2 < uVar2 + *(int *)(iVar4 + 0xc) * 4) goto LAB_0071f7d4;
          if (*(int *)(iVar4 + 0xc) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x10) = iVar3;
          iVar3 = FUN_0071f3c0(iVar4 + 0x14);
          if (iVar3 == 0) {
            return 0;
          }
          uVar2 = *(uint *)(iVar4 + 0x24);
          if (param_2 < uVar2) goto LAB_0071f7b7;
          if (param_2 < uVar2 + *(int *)(iVar4 + 0x20) * 8) goto LAB_0071f7d4;
          if (*(int *)(iVar4 + 0x20) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x24) = iVar3;
          uVar2 = *(uint *)(iVar4 + 0x2c);
          if (param_2 < uVar2) goto LAB_0071f7b7;
          if (param_2 < uVar2 + *(int *)(iVar4 + 0x28) * 4) goto LAB_0071f7d4;
          if (*(int *)(iVar4 + 0x28) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x2c) = iVar3;
          uVar2 = *(uint *)(iVar4 + 0x34);
          if (param_2 < uVar2) goto LAB_0071f7b7;
          if (param_2 < uVar2 + *(int *)(iVar4 + 0x30) * 2) goto LAB_0071f7d4;
          if (*(int *)(iVar4 + 0x30) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x34) = iVar3;
          local_c = local_c + 1;
          local_8 = local_8 + 0x38;
        } while (local_c < *param_3);
      }
      return 1;
    }
LAB_0071f7d4:
    uVar5 = 0x110;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f904,uVar5,&DAT_007f5634,0,1);
  return 0;
}


