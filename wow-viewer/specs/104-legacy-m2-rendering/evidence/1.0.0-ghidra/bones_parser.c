
undefined4 __fastcall FUN_0071f440(int param_1,uint param_2,uint *param_3)

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
LAB_0071f457:
    uVar5 = 0x10a;
  }
  else {
    uVar1 = *param_3;
    if (uVar1 * 0x6c + uVar2 <= param_2) {
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
          uVar2 = *(uint *)(param_3[1] + 0x14 + local_8);
          iVar4 = param_3[1] + local_8;
          if (param_2 < uVar2) goto LAB_0071f457;
          if (param_2 < uVar2 + *(int *)(iVar4 + 0x10) * 8) goto LAB_0071f474;
          if (*(int *)(iVar4 + 0x10) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x14) = iVar3;
          iVar3 = FUN_0071f5f0(iVar4 + 0x18);
          if (iVar3 == 0) {
            return 0;
          }
          iVar3 = FUN_0071f3c0(iVar4 + 0x20);
          if (iVar3 == 0) {
            return 0;
          }
          uVar2 = *(uint *)(iVar4 + 0x30);
          if (param_2 < uVar2) goto LAB_0071f457;
          if (param_2 < uVar2 + *(int *)(iVar4 + 0x2c) * 8) goto LAB_0071f474;
          if (*(int *)(iVar4 + 0x2c) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x30) = iVar3;
          iVar3 = FUN_0071f5f0(iVar4 + 0x34);
          if (iVar3 == 0) {
            return 0;
          }
          iVar3 = FUN_00720d30(iVar4 + 0x3c);
          if (iVar3 == 0) {
            return 0;
          }
          uVar2 = *(uint *)(iVar4 + 0x4c);
          if (param_2 < uVar2) goto LAB_0071f457;
          if (param_2 < uVar2 + *(int *)(iVar4 + 0x48) * 8) goto LAB_0071f474;
          if (*(int *)(iVar4 + 0x48) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x4c) = iVar3;
          uVar2 = *(uint *)(iVar4 + 0x54);
          if (param_2 < uVar2) goto LAB_0071f457;
          if (param_2 < uVar2 + *(int *)(iVar4 + 0x50) * 4) goto LAB_0071f474;
          if (*(int *)(iVar4 + 0x50) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x54) = iVar3;
          iVar4 = FUN_0071f3c0(iVar4 + 0x58);
          if (iVar4 == 0) {
            return 0;
          }
          local_c = local_c + 1;
          local_8 = local_8 + 0x6c;
        } while (local_c < *param_3);
      }
      return 1;
    }
LAB_0071f474:
    uVar5 = 0x110;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f904,uVar5,&DAT_007f5634,0,1);
  return 0;
}


