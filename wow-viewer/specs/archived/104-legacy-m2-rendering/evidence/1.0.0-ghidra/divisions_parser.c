
undefined4 __fastcall FUN_0071f6f0(int param_1,uint param_2,uint *param_3)

{
  uint uVar1;
  uint uVar2;
  int iVar3;
  uint uVar4;
  int iVar5;
  undefined4 uVar6;
  
  uVar2 = param_3[1];
  if (param_2 < uVar2) {
    uVar6 = 0x10a;
  }
  else {
    uVar1 = *param_3;
    if (uVar1 * 0x2c + uVar2 <= param_2) {
      if (uVar1 == 0) {
        uVar2 = 0;
      }
      else {
        uVar2 = uVar2 + param_1;
      }
      uVar4 = 0;
      param_3[1] = uVar2;
      if (uVar1 != 0) {
        iVar5 = 0;
        do {
          iVar3 = FUN_0071e0b0(param_3[1] + iVar5);
          if (iVar3 == 0) {
            return 0;
          }
          uVar4 = uVar4 + 1;
          iVar5 = iVar5 + 0x2c;
        } while (uVar4 < *param_3);
      }
      return 1;
    }
    uVar6 = 0x110;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f904,uVar6,&DAT_007f5634,0,1);
  return 0;
}


