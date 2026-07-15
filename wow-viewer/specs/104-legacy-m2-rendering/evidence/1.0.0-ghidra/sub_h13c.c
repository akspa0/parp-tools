
undefined4 __fastcall FUN_007208e0(int param_1,uint param_2,uint *param_3)

{
  uint uVar1;
  int iVar2;
  uint uVar3;
  int iVar4;
  undefined4 uVar5;
  uint local_c;
  int local_8;
  
  uVar3 = param_3[1];
  if (param_2 < uVar3) {
LAB_007208f7:
    uVar5 = 0x10a;
  }
  else {
    uVar1 = *param_3;
    if (uVar1 * 0x1f8 + uVar3 <= param_2) {
      if (uVar1 == 0) {
        uVar3 = 0;
      }
      else {
        uVar3 = uVar3 + param_1;
      }
      param_3[1] = uVar3;
      local_c = 0;
      if (uVar1 != 0) {
        local_8 = 0;
        do {
          uVar3 = *(uint *)(param_3[1] + 0x1c + local_8);
          iVar4 = param_3[1] + local_8;
          if (param_2 < uVar3) goto LAB_007208f7;
          if (param_2 < *(int *)(iVar4 + 0x18) + uVar3) goto LAB_00720917;
          if (*(int *)(iVar4 + 0x18) == 0) {
            iVar2 = 0;
          }
          else {
            iVar2 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 0x1c) = iVar2;
          uVar3 = *(uint *)(iVar4 + 0x24);
          if (param_2 < uVar3) goto LAB_007208f7;
          if (param_2 < *(int *)(iVar4 + 0x20) + uVar3) goto LAB_00720917;
          if (*(int *)(iVar4 + 0x20) == 0) {
            iVar2 = 0;
          }
          else {
            iVar2 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 0x24) = iVar2;
          iVar2 = FUN_007203d0(iVar4 + 0x38);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0x40);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 0x48);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0x54);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0x5c);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 100);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0x70);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0x78);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 0x80);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0x8c);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0x94);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 0x9c);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0xa8);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0xb0);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 0xb8);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0xc4);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0xcc);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 0xd4);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0xe0);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0xe8);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 0xf0);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0xfc);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0x104);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 0x10c);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0x118);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0x120);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 0x128);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0x134);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0x13c);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 0x144);
          if (iVar2 == 0) {
            return 0;
          }
          uVar3 = *(uint *)(iVar4 + 0x1d8);
          if (param_2 < uVar3) goto LAB_007208f7;
          if (param_2 < uVar3 + *(int *)(iVar4 + 0x1d4) * 0xc) goto LAB_00720917;
          if (*(int *)(iVar4 + 0x1d4) == 0) {
            iVar2 = 0;
          }
          else {
            iVar2 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 0x1d8) = iVar2;
          uVar3 = *(uint *)(iVar4 + 0x1e4);
          if (param_2 < uVar3) goto LAB_007208f7;
          if (param_2 < uVar3 + *(int *)(iVar4 + 0x1e0) * 8) goto LAB_00720917;
          if (*(int *)(iVar4 + 0x1e0) == 0) {
            iVar2 = 0;
          }
          else {
            iVar2 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 0x1e4) = iVar2;
          iVar2 = FUN_0071f5f0(iVar4 + 0x1e8);
          if (iVar2 == 0) {
            return 0;
          }
          iVar4 = FUN_00720e30(iVar4 + 0x1f0);
          if (iVar4 == 0) {
            return 0;
          }
          local_c = local_c + 1;
          local_8 = local_8 + 0x1f8;
        } while (local_c < *param_3);
      }
      return 1;
    }
LAB_00720917:
    uVar5 = 0x110;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f904,uVar5,&DAT_007f5634,0,1);
  return 0;
}


