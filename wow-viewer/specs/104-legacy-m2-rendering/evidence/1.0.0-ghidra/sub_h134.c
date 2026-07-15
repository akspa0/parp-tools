
undefined4 __fastcall FUN_00720600(int param_1,uint param_2,uint *param_3)

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
LAB_00720617:
    uVar5 = 0x10a;
  }
  else {
    uVar1 = *param_3;
    if (uVar1 * 0xdc + uVar3 <= param_2) {
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
          uVar3 = *(uint *)(param_3[1] + 0x18 + local_8);
          iVar4 = param_3[1] + local_8;
          if (param_2 < uVar3) goto LAB_00720617;
          if (param_2 < uVar3 + *(int *)(iVar4 + 0x14) * 2) goto LAB_00720637;
          if (*(int *)(iVar4 + 0x14) == 0) {
            iVar2 = 0;
          }
          else {
            iVar2 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 0x18) = iVar2;
          uVar3 = *(uint *)(iVar4 + 0x20);
          if (param_2 < uVar3) goto LAB_00720617;
          if (param_2 < uVar3 + *(int *)(iVar4 + 0x1c) * 2) goto LAB_00720637;
          if (*(int *)(iVar4 + 0x1c) == 0) {
            iVar2 = 0;
          }
          else {
            iVar2 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 0x20) = iVar2;
          iVar2 = FUN_007203d0(iVar4 + 0x28);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0x30);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f3c0(iVar4 + 0x38);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0x44);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0x4c);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720db0(iVar4 + 0x54);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0x60);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0x68);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 0x70);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_007203d0(iVar4 + 0x7c);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f5f0(iVar4 + 0x84);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_00720eb0(iVar4 + 0x8c);
          if (iVar2 == 0) {
            return 0;
          }
          uVar3 = *(uint *)(iVar4 + 0xac);
          if (param_2 < uVar3) goto LAB_00720617;
          if (param_2 < uVar3 + *(int *)(iVar4 + 0xa8) * 8) goto LAB_00720637;
          if (*(int *)(iVar4 + 0xa8) == 0) {
            iVar2 = 0;
          }
          else {
            iVar2 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 0xac) = iVar2;
          iVar2 = FUN_0071f5f0(iVar4 + 0xb0);
          if (iVar2 == 0) {
            return 0;
          }
          iVar2 = FUN_0071f2c0(iVar4 + 0xb8);
          if (iVar2 == 0) {
            return 0;
          }
          uVar3 = *(uint *)(iVar4 + 200);
          if (param_2 < uVar3) goto LAB_00720617;
          if (param_2 < uVar3 + *(int *)(iVar4 + 0xc4) * 8) goto LAB_00720637;
          if (*(int *)(iVar4 + 0xc4) == 0) {
            iVar2 = 0;
          }
          else {
            iVar2 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 200) = iVar2;
          uVar3 = *(uint *)(iVar4 + 0xd0);
          if (param_2 < uVar3) goto LAB_00720617;
          if (param_2 < uVar3 + *(int *)(iVar4 + 0xcc) * 4) goto LAB_00720637;
          if (*(int *)(iVar4 + 0xcc) == 0) {
            iVar2 = 0;
          }
          else {
            iVar2 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 0xd0) = iVar2;
          iVar4 = FUN_00720e30(iVar4 + 0xd4);
          if (iVar4 == 0) {
            return 0;
          }
          local_c = local_c + 1;
          local_8 = local_8 + 0xdc;
        } while (local_c < *param_3);
      }
      return 1;
    }
LAB_00720637:
    uVar5 = 0x110;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f904,uVar5,&DAT_007f5634,0,1);
  return 0;
}


