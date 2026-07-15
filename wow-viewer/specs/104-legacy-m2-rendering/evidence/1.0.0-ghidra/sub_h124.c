
undefined4 __fastcall FUN_00720450(int param_1,uint param_2,uint *param_3)

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
LAB_00720467:
    uVar5 = 0x10a;
  }
  else {
    uVar1 = *param_3;
    if (uVar1 * 0x7c + uVar2 <= param_2) {
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
          iVar4 = param_3[1] + local_8;
          iVar3 = FUN_007203d0(iVar4 + 0x14);
          if (iVar3 == 0) {
            return 0;
          }
          iVar3 = FUN_0071f5f0(iVar4 + 0x1c);
          if (iVar3 == 0) {
            return 0;
          }
          iVar3 = FUN_00720f30(iVar4 + 0x24);
          if (iVar3 == 0) {
            return 0;
          }
          uVar2 = *(uint *)(iVar4 + 0x40);
          if (param_2 < uVar2) goto LAB_00720467;
          if (param_2 < uVar2 + *(int *)(iVar4 + 0x3c) * 8) goto LAB_00720484;
          if (*(int *)(iVar4 + 0x3c) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x40) = iVar3;
          iVar3 = FUN_0071f5f0(iVar4 + 0x44);
          if (iVar3 == 0) {
            return 0;
          }
          iVar3 = FUN_00720f30(iVar4 + 0x4c);
          if (iVar3 == 0) {
            return 0;
          }
          uVar2 = *(uint *)(iVar4 + 0x68);
          if (param_2 < uVar2) goto LAB_00720467;
          if (param_2 < uVar2 + *(int *)(iVar4 + 100) * 8) goto LAB_00720484;
          if (*(int *)(iVar4 + 100) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x68) = iVar3;
          uVar2 = *(uint *)(iVar4 + 0x70);
          if (param_2 < uVar2) goto LAB_00720467;
          if (param_2 < uVar2 + *(int *)(iVar4 + 0x6c) * 4) goto LAB_00720484;
          if (*(int *)(iVar4 + 0x6c) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x70) = iVar3;
          uVar2 = *(uint *)(iVar4 + 0x78);
          if (param_2 < uVar2) goto LAB_00720467;
          if (param_2 < uVar2 + *(int *)(iVar4 + 0x74) * 0xc) goto LAB_00720484;
          if (*(int *)(iVar4 + 0x74) == 0) {
            iVar3 = 0;
          }
          else {
            iVar3 = uVar2 + param_1;
          }
          *(int *)(iVar4 + 0x78) = iVar3;
          local_c = local_c + 1;
          local_8 = local_8 + 0x7c;
        } while (local_c < *param_3);
      }
      return 1;
    }
LAB_00720484:
    uVar5 = 0x110;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f904,uVar5,&DAT_007f5634,0,1);
  return 0;
}


