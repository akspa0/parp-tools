
void FUN_006976f0(void)

{
  int *piVar1;
  int iVar2;
  undefined1 local_15c [256];
  undefined1 local_5c [4];
  int local_58;
  undefined4 local_54;
  undefined4 local_50;
  undefined4 local_4c;
  undefined4 local_1c;
  undefined4 local_18;
  undefined4 local_14;
  int local_10;
  undefined4 local_c;
  int local_8;
  
  local_8 = 0;
  FUN_00643170(&DAT_00a93528,&local_8);
  if (local_8 == 0) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00835d80,0x15a,s_wdtFile_00835fe8,0,1
                );
  }
  local_10 = 0;
  local_c = 0;
  FUN_00643d30(local_8,&local_10,8,0,0,0);
  if (local_10 != 0x4d564552) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00835d80,0x15f,
                 s_iffChunk_token___MVER__00835fd0,0,1);
  }
  FUN_00643d30(local_8,&DAT_00a937b0,4,0,0,0);
  FUN_00643d30(local_8,&local_10,8,0,0,0);
  if (local_10 != 0x4d504844) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00835d80,0x164,
                 s_iffChunk_token___MPHD__00835fb8,0,1);
  }
  FUN_00643d30(local_8,&DAT_00a9b804,0x20,0,0,0);
  FUN_00643d30(local_8,&local_10,8,0,0,0);
  if (local_10 != 0x4d41494e) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00835d80,0x169,
                 s_iffChunk_token___MAIN__00835fa0,0,1);
  }
  FUN_00643d30(local_8,&DAT_00a873e8,0x8000,0,0,0);
  if ((DAT_00a9b804 & 1) != 0) {
    FUN_00643d30(local_8,&local_10,8,0,0,0);
    if (local_10 != 0x4d574d4f) {
      FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00835d80,0x171,
                   s_iffChunk_token_____MWMO__00835f84,0,1);
    }
    FUN_00643d30(local_8,local_15c,local_c,0,0,0);
    FUN_00643d30(local_8,&local_10,8,0,0,0);
    if (local_10 != 0x4d4f4446) {
      FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00835d80,0x176,
                   s_iffChunk_token_____MODF__00835f68,0,1);
    }
    local_54 = 0;
    local_50 = 0;
    local_4c = 0;
    FUN_00409150(0);
    FUN_0052f4a0(0);
    FUN_00643d30(local_8,local_5c,0x40,0,0,0);
    local_58 = DAT_00a977c0;
    DAT_00a977c0 = DAT_00a977c0 + -1;
    local_1c = 0;
    local_18 = 0;
    local_14 = 0;
    FUN_006988f0(&local_1c);
    iVar2 = FUN_006a8920();
    *(undefined4 *)(iVar2 + 8) = 0;
    piVar1 = (int *)(DAT_00a9b7f8 + iVar2);
    if (*(int *)(DAT_00a9b7f8 + iVar2) != 0) {
      FUN_006995f0();
    }
    *piVar1 = (int)DAT_00a9b7fc;
    piVar1[1] = DAT_00a9b7fc[1];
    DAT_00a9b7fc[1] = iVar2;
    DAT_00a9b828 = 1;
    DAT_00a9b7fc = piVar1;
  }
  FUN_00644030(local_8);
  return;
}


