
undefined4 __fastcall FUN_0071f930(int param_1,uint param_2,uint *param_3)

{
  int *piVar1;
  uint uVar2;
  uint *puVar3;
  uint uVar4;
  int iVar5;
  int iVar6;
  undefined4 uVar7;
  
  puVar3 = param_3;
  uVar4 = param_3[1];
  if (param_2 < uVar4) {
LAB_0071f949:
    uVar7 = 0x10a;
  }
  else {
    uVar2 = *param_3;
    if (uVar2 * 0x10 + uVar4 <= param_2) {
      if (uVar2 == 0) {
        uVar4 = 0;
      }
      else {
        uVar4 = uVar4 + param_1;
      }
      param_3[1] = uVar4;
      param_3 = (uint *)0x0;
      if (uVar2 != 0) {
        iVar5 = 0;
        do {
          uVar4 = puVar3[1];
          uVar2 = *(uint *)(iVar5 + 0xc + uVar4);
          piVar1 = (int *)(iVar5 + 8 + uVar4);
          if (param_2 < uVar2) goto LAB_0071f949;
          iVar6 = *piVar1;
          if (param_2 < iVar6 + uVar2) goto LAB_0071f966;
          if (iVar6 == 0) {
            iVar6 = 0;
          }
          else {
            iVar6 = uVar2 + param_1;
          }
          piVar1[1] = iVar6;
          param_3 = (uint *)((int)param_3 + 1);
          iVar5 = iVar5 + 0x10;
        } while (param_3 < *puVar3);
      }
      return 1;
    }
LAB_0071f966:
    uVar7 = 0x110;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f904,uVar7,&DAT_007f5634,0,1);
  return 0;
}


