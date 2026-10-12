
undefined4 __fastcall FUN_0071e0b0(int param_1,uint param_2,int *param_3)

{
  uint uVar1;
  int iVar2;
  undefined4 uVar3;
  
  uVar1 = param_3[1];
  if (param_2 < uVar1) {
LAB_0071e165:
    uVar3 = 0x10a;
  }
  else {
    if (uVar1 + *param_3 * 2 <= param_2) {
      if (*param_3 == 0) {
        iVar2 = 0;
      }
      else {
        iVar2 = uVar1 + param_1;
      }
      param_3[1] = iVar2;
      uVar1 = param_3[3];
      if (param_2 < uVar1) goto LAB_0071e165;
      if (uVar1 + param_3[2] * 2 <= param_2) {
        if (param_3[2] == 0) {
          iVar2 = 0;
        }
        else {
          iVar2 = uVar1 + param_1;
        }
        param_3[3] = iVar2;
        uVar1 = param_3[5];
        if (param_2 < uVar1) goto LAB_0071e165;
        if (uVar1 + param_3[4] * 4 <= param_2) {
          if (param_3[4] == 0) {
            iVar2 = 0;
          }
          else {
            iVar2 = uVar1 + param_1;
          }
          param_3[5] = iVar2;
          uVar1 = param_3[7];
          if (param_2 < uVar1) goto LAB_0071e165;
          if (param_3[6] * 0x20 + uVar1 <= param_2) {
            if (param_3[6] == 0) {
              param_1 = 0;
            }
            else {
              param_1 = uVar1 + param_1;
            }
            param_3[7] = param_1;
            iVar2 = FUN_0071f340(param_3 + 8);
            if (iVar2 == 0) {
              return 0;
            }
            return 1;
          }
        }
      }
    }
    uVar3 = 0x110;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f904,uVar3,&DAT_007f5634,0,1);
  return 0;
}


