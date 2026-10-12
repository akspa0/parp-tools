
undefined4 __fastcall FUN_0071fa00(int param_1,uint param_2,uint *param_3)

{
  uint uVar1;
  uint *puVar2;
  uint uVar3;
  int iVar4;
  int iVar5;
  undefined4 uVar6;
  uint local_c;
  
  puVar2 = param_3;
  uVar3 = param_3[1];
  if (param_2 < uVar3) {
LAB_0071fa16:
    uVar6 = 0x10a;
  }
  else {
    uVar1 = *param_3;
    if (uVar1 * 0x1c + uVar3 <= param_2) {
      if (uVar1 == 0) {
        uVar3 = 0;
      }
      else {
        uVar3 = uVar3 + param_1;
      }
      param_3[1] = uVar3;
      local_c = 0;
      if (uVar1 != 0) {
        param_3 = (uint *)0x0;
        do {
          iVar4 = puVar2[1] + (int)param_3;
          uVar3 = *(uint *)(iVar4 + 8);
          if (param_2 < uVar3) goto LAB_0071fa16;
          if (param_2 < uVar3 + *(int *)(iVar4 + 4) * 8) goto LAB_0071fa33;
          if (*(int *)(iVar4 + 4) == 0) {
            iVar5 = 0;
          }
          else {
            iVar5 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 8) = iVar5;
          uVar3 = *(uint *)(iVar4 + 0x10);
          if (param_2 < uVar3) goto LAB_0071fa16;
          if (param_2 < uVar3 + *(int *)(iVar4 + 0xc) * 4) goto LAB_0071fa33;
          if (*(int *)(iVar4 + 0xc) == 0) {
            iVar5 = 0;
          }
          else {
            iVar5 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 0x10) = iVar5;
          uVar3 = *(uint *)(iVar4 + 0x18);
          if (param_2 < uVar3) goto LAB_0071fa16;
          if (param_2 < uVar3 + *(int *)(iVar4 + 0x14) * 2) goto LAB_0071fa33;
          if (*(int *)(iVar4 + 0x14) == 0) {
            iVar5 = 0;
          }
          else {
            iVar5 = uVar3 + param_1;
          }
          *(int *)(iVar4 + 0x18) = iVar5;
          local_c = local_c + 1;
          param_3 = (uint *)((int)param_3 + 0x1c);
        } while (local_c < *puVar2);
      }
      return 1;
    }
LAB_0071fa33:
    uVar6 = 0x110;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f904,uVar6,&DAT_007f5634,0,1);
  return 0;
}


