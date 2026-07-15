
undefined4 __fastcall FUN_0071f3c0(int param_1,uint param_2,int *param_3)

{
  uint uVar1;
  undefined4 uVar2;
  
  uVar1 = param_3[1];
  if (param_2 < uVar1) {
    uVar2 = 0x10a;
  }
  else {
    if (uVar1 + *param_3 * 0xc <= param_2) {
      if (*param_3 != 0) {
        param_3[1] = uVar1 + param_1;
        return 1;
      }
      param_3[1] = 0;
      return 1;
    }
    uVar2 = 0x110;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f904,uVar2,&DAT_007f5634,0,1);
  return 0;
}


