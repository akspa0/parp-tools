
undefined4 __fastcall FUN_0070e350(int param_1)

{
  int iVar1;
  ushort uVar2;
  
  if (*(int *)(param_1 + 0x10) == 0) {
    FUN_0070b920(s_HasAttachment_0083f434);
  }
  iVar1 = *(int *)(*(int *)(param_1 + 0x2c) + 0x130);
  uVar2 = FUN_0070b6c0();
  if (uVar2 == 0xffff) {
    return 0;
  }
  if (*(uint *)(iVar1 + 0x104) <= (uint)uVar2) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083ed44,0xb0c,
                 s_attachmentIndex_<_data_>attachme_0083f408,0,1);
  }
  return 1;
}


