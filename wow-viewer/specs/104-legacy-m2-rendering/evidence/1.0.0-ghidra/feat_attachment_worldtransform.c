
void __thiscall FUN_0070e500(int param_1,float *param_2)

{
  ushort uVar1;
  int iVar2;
  int iVar3;
  float *pfVar4;
  float *pfVar5;
  float local_44 [4];
  float local_34;
  float local_30;
  float local_2c;
  float local_24;
  float local_20;
  float local_1c;
  float local_14;
  float local_10;
  float local_c;
  
  if (*(int *)(param_1 + 0x10) == 0) {
    FUN_0070b920(s_GetAttachmentWorldTransform_0083f470);
  }
  iVar2 = *(int *)(*(int *)(param_1 + 0x2c) + 0x130);
  uVar1 = FUN_0070b6c0();
  if (*(uint *)(iVar2 + 0x104) <= (uint)uVar1) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083ed44,0xb44,
                 s_attachmentIndex_<_data_>attachme_0083f408,0,1);
  }
  FUN_0070f820();
  iVar2 = (uint)uVar1 * 0x30 + *(int *)(iVar2 + 0x108);
  pfVar4 = (float *)((uint)*(ushort *)(iVar2 + 4) * 0x40 + *(int *)(param_1 + 0x84));
  pfVar5 = local_44;
  for (iVar3 = 0x10; iVar3 != 0; iVar3 = iVar3 + -1) {
    *pfVar5 = *pfVar4;
    pfVar4 = pfVar4 + 1;
    pfVar5 = pfVar5 + 1;
  }
  local_14 = local_44[0] * *(float *)(iVar2 + 8) +
             local_34 * *(float *)(iVar2 + 0xc) + local_24 * *(float *)(iVar2 + 0x10) + local_14;
  local_10 = local_44[1] * *(float *)(iVar2 + 8) +
             local_30 * *(float *)(iVar2 + 0xc) + local_20 * *(float *)(iVar2 + 0x10) + local_10;
  local_c = local_44[2] * *(float *)(iVar2 + 8) +
            local_2c * *(float *)(iVar2 + 0xc) + local_1c * *(float *)(iVar2 + 0x10) + local_c;
  thunk_FUN_00726499(local_44,local_44,*(int *)(param_1 + 0x28) + 0xf8);
  pfVar4 = local_44;
  for (iVar2 = 0x10; iVar2 != 0; iVar2 = iVar2 + -1) {
    *param_2 = *pfVar4;
    pfVar4 = pfVar4 + 1;
    param_2 = param_2 + 1;
  }
  return;
}


