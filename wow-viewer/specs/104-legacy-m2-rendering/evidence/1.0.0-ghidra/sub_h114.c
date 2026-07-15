
undefined4 __fastcall FUN_0071ffc0(int param_1,uint param_2,uint *param_3)

{
  int iVar1;
  uint uVar2;
  uint *puVar3;
  uint uVar4;
  int iVar5;
  undefined4 uVar6;
  int local_c;
  
  puVar3 = param_3;
  uVar4 = param_3[1];
  if (param_2 < uVar4) {
LAB_0071ffd6:
    uVar6 = 0x10a;
  }
  else {
    uVar2 = *param_3;
    if (uVar2 * 0x2c + uVar4 <= param_2) {
      if (uVar2 == 0) {
        uVar4 = 0;
      }
      else {
        uVar4 = uVar4 + param_1;
      }
      param_3[1] = uVar4;
      param_3 = (uint *)0x0;
      if (uVar2 != 0) {
        local_c = 0;
        do {
          uVar4 = puVar3[1];
          uVar2 = *(uint *)(local_c + 0x20 + uVar4);
          iVar1 = local_c + 0x18 + uVar4;
          if (param_2 < uVar2) goto LAB_0071ffd6;
          if (param_2 < uVar2 + *(int *)(iVar1 + 4) * 8) goto LAB_0071fff3;
          if (*(int *)(iVar1 + 4) == 0) {
            iVar5 = 0;
          }
          else {
            iVar5 = uVar2 + param_1;
          }
          *(int *)(iVar1 + 8) = iVar5;
          uVar4 = *(uint *)(iVar1 + 0x10);
          if (param_2 < uVar4) goto LAB_0071ffd6;
          if (param_2 < uVar4 + *(int *)(iVar1 + 0xc) * 4) goto LAB_0071fff3;
          if (*(int *)(iVar1 + 0xc) == 0) {
            iVar5 = 0;
          }
          else {
            iVar5 = uVar4 + param_1;
          }
          *(int *)(iVar1 + 0x10) = iVar5;
          param_3 = (uint *)((int)param_3 + 1);
          local_c = local_c + 0x2c;
        } while (param_3 < *puVar3);
      }
      return 1;
    }
LAB_0071fff3:
    uVar6 = 0x110;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f904,uVar6,&DAT_007f5634,0,1);
  return 0;
}


