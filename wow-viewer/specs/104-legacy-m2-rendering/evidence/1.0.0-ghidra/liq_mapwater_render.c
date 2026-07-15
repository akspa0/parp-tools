
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

void __thiscall
FUN_00687460(int param_1,float *param_2,int *param_3,int param_4,int param_5,int param_6)

{
  float *pfVar1;
  float fVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  float fVar6;
  float fVar7;
  float fVar8;
  float fVar9;
  float *pfVar10;
  uint uVar11;
  int *piVar12;
  int iVar13;
  uint uVar14;
  uint uVar15;
  uint uVar16;
  double dVar17;
  uint local_2c;
  uint local_28;
  int local_18;
  int local_10;
  uint local_8;
  
  pfVar10 = param_2;
  fVar2 = -(param_2[1] - _DAT_007cbb5c);
  fVar6 = -(*param_2 - _DAT_007cbb5c);
  if ((fVar2 < DAT_007cbb50) || (fVar6 < DAT_007cbb50)) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00834f64,0x28c,
                 s_mx_>__0_0f____my_>__0_0f_00835050,0,1);
  }
  if ((fVar2 < _DAT_007cbb58 == (NAN(fVar2) || NAN(_DAT_007cbb58))) ||
     (fVar6 < _DAT_007cbb58 == (NAN(fVar6) || NAN(_DAT_007cbb58)))) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00834f64,0x28d,
                 s_mx_<_((64*16)*((150_0f_36_0f)*8)_00835008,0,1);
  }
  fVar7 = (float)param_3 * _DAT_007dcd08;
  uVar11 = (int)ROUND(_DAT_007dcc48 * fVar2 - _DAT_00834d48) & 0xf;
  uVar16 = (int)ROUND(_DAT_007dcc48 * fVar6 - _DAT_00834d48) & 0xf;
  if (fVar7 < _DAT_007cba5c == (NAN(fVar7) || NAN(_DAT_007cba5c))) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_WoW_Source_Wor_00834f64,0x292,
                 s_radius____150_0f_36_0f__<_256_0f_008350d8,0,1);
  }
  dVar17 = ceil((double)fVar7);
  fVar2 = (float)dVar17;
  if (dVar17 < (double)DAT_007cbb50) {
    FUN_0063e0c0(0x85100000,s__________engine_source_Tempest_c_007fc2ac,0x93,0,1,
                 s___s____s____f_007f6030,s_x_>__0_0f_007fb240,&DAT_007f604c,(double)fVar2);
  }
  if (fVar2 < _DAT_007cd990 == (fVar2 == _DAT_007cd990)) {
    FUN_0063e0c0(0x85100000,s__________engine_source_Tempest_c_007fc2ac,0x94,0,1,
                 s___s____s____f_007f6030,s_x_<__255_9999f_007fb230,&DAT_007f604c,(double)fVar2);
  }
  local_8 = (uint)(fVar2 + _DAT_007cd98c) >> 0xe & 0xff;
  uVar14 = uVar16 - local_8;
  uVar14 = ((int)uVar14 < 1) - 1 & uVar14;
  uVar15 = uVar11 - local_8;
  uVar15 = uVar15 & ((int)uVar15 < 1) - 1;
  uVar16 = uVar16 + local_8;
  if (0xe < uVar16) {
    uVar16 = 0xf;
  }
  local_8 = local_8 + uVar11;
  if (0xe < local_8) {
    local_8 = 0xf;
  }
  if ((int)uVar14 <= (int)uVar16) {
    param_3 = (int *)(param_1 + 0x27c + (uVar14 * 0x10 + uVar15) * 4);
    local_18 = (uVar16 - uVar14) + 1;
    do {
      if ((int)uVar15 <= (int)local_8) {
        param_2 = (float *)param_3;
        iVar13 = (local_8 - uVar15) + 1;
        do {
          iVar3 = (int)*param_2;
          if (iVar3 != 0) {
            piVar12 = (int *)(iVar3 + 0x11c);
            local_10 = 4;
            do {
              iVar4 = *piVar12;
              if (iVar4 != 0) {
                local_28 = 0;
                do {
                  local_2c = 0;
                  do {
                    iVar5 = *(int *)(iVar4 + 0x10);
                    if (((((int)local_2c < 0) || (7 < local_2c)) || ((int)local_28 < 0)) ||
                       (7 < local_28)) {
                      FUN_0063d520(0x85100000,s_______common_MapDefs_h_0083506c,0x2d8,
                                   s_pos_x_>__0____pos_x_<_MD_LIQUID__00835088,0,1);
                    }
                    uVar16 = *(byte *)(iVar5 + local_28 * 8 + local_2c) & 0xf;
                    if (uVar16 != 0xf) {
                      *(undefined4 *)(param_4 + uVar16 * 4) = 1;
                      fVar6 = (*(float *)(iVar3 + 0x6c) - (float)(int)local_28 * _DAT_007dcc34) -
                              *pfVar10;
                      fVar8 = (*(float *)(iVar3 + 0x70) - (float)(int)local_2c * _DAT_007dcc34) -
                              pfVar10[1];
                      fVar9 = (*(float *)(iVar4 + 8) + *(float *)(iVar4 + 4)) * _DAT_007cbad0 -
                              pfVar10[2];
                      fVar7 = fVar9 * fVar9 + fVar6 * fVar6 + fVar8 * fVar8;
                      fVar2 = *(float *)(param_6 + uVar16 * 4);
                      if (fVar7 < fVar2 != (NAN(fVar7) || NAN(fVar2))) {
                        *(float *)(param_6 + uVar16 * 4) = fVar7;
                        pfVar1 = (float *)(param_5 + uVar16 * 0xc);
                        *pfVar1 = fVar6;
                        pfVar1[1] = fVar8;
                        pfVar1[2] = fVar9;
                      }
                    }
                    local_2c = local_2c + 1;
                  } while (local_2c < 8);
                  local_28 = local_28 + 1;
                } while (local_28 < 8);
              }
              piVar12 = piVar12 + 1;
              local_10 = local_10 + -1;
            } while (local_10 != 0);
          }
          param_2 = (float *)((int *)param_2 + 1);
          iVar13 = iVar13 + -1;
        } while (iVar13 != 0);
      }
      param_3 = param_3 + 0x10;
      local_18 = local_18 + -1;
    } while (local_18 != 0);
  }
  return;
}


