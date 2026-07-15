
/* WARNING: Globals starting with '_' overlap smaller symbols at the same address */

undefined4 __thiscall
FUN_00717da0(int param_1,float *param_2,float *param_3,float *param_4,int param_5)

{
  float *pfVar1;
  float *pfVar2;
  float fVar3;
  float fVar4;
  float fVar5;
  float fVar6;
  float fVar7;
  uint uVar8;
  float *pfVar9;
  int iVar10;
  int *piVar11;
  int iVar12;
  int *piVar13;
  float fVar14;
  byte *pbVar15;
  uint uVar16;
  ushort *puVar17;
  float10 fVar18;
  undefined4 uVar19;
  float local_d0;
  float local_cc;
  float local_c8;
  float local_b8;
  float local_b4;
  float local_b0;
  undefined4 local_ac;
  float local_a8;
  float local_a4;
  float local_a0;
  undefined4 local_9c;
  float local_98;
  float local_94;
  float local_90;
  undefined4 local_8c;
  float local_88;
  float local_84;
  float local_80;
  undefined4 local_7c;
  float local_78;
  float local_74;
  float local_70;
  float local_6c;
  int local_68;
  float local_64;
  ushort *local_60;
  int local_5c;
  uint local_58;
  int *local_54;
  int *local_50;
  float local_4c;
  float local_48;
  float local_44;
  int local_40;
  float local_3c;
  uint local_38;
  float local_34;
  float local_30;
  float local_2c;
  float local_28;
  float local_24;
  float local_20;
  float local_1c;
  float local_18;
  float local_14;
  float local_10;
  float local_c;
  float local_8;
  
  local_68 = param_1;
  if (*(int *)(param_1 + 0x148) == 0) {
    FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f784,0x31a,s_m_hitTest_0083f800,0
                 ,1);
  }
  if (*param_4 < _DAT_007d651c == (NAN(*param_4) || NAN(_DAT_007d651c))) {
    local_14 = *param_3 - *param_2;
    local_10 = param_3[1] - param_2[1];
    local_c = param_3[2] - param_2[2];
    fVar18 = (float10)FUN_00404fd0(local_14 * local_14 + local_10 * local_10 + local_c * local_c);
    local_2c = (float)fVar18;
    if (fVar18 < (float10)_DAT_007d651c == (NAN(fVar18) || NAN((float10)_DAT_007d651c))) {
      local_44 = _DAT_007cba60 / local_2c;
      local_4c = local_14 * local_44;
      local_48 = local_10 * local_44;
      local_44 = local_c * local_44;
      FUN_00410bb0();
      uVar16 = 0;
      for (iVar10 = *(int *)(param_1 + 0x14c); iVar10 != 0; iVar10 = *(int *)(iVar10 + 1000)) {
        uVar16 = uVar16 + 1;
      }
      if (*(uint *)(param_1 + 0x158) < uVar16) {
        if (*(int *)(param_1 + 0x150) != 0) {
          FUN_0063cda0(*(int *)(param_1 + 0x150),s_delete___007f6e74,0xffffffff,0);
        }
        if (*(int *)(param_1 + 0x154) != 0) {
          FUN_0063cda0(*(int *)(param_1 + 0x154),s_delete___007f6e74,0xffffffff,0);
        }
        if (*(int *)(param_1 + 0x158) == 0) {
          *(undefined4 *)(param_1 + 0x158) = 1;
        }
        uVar8 = *(uint *)(param_1 + 0x158);
        if (uVar8 < uVar16) {
          do {
            uVar8 = uVar8 << 1;
          } while (uVar8 < uVar16);
          *(uint *)(param_1 + 0x158) = uVar8;
        }
        uVar19 = FUN_0063cb90(*(int *)(param_1 + 0x158) << 4,s___AUM2HitRec___0083f7f0,0xfffffffe,0)
        ;
        *(undefined4 *)(param_1 + 0x150) = uVar19;
        uVar19 = FUN_0063cb90(*(int *)(param_1 + 0x158) << 2,s__I_007f6f24,0xfffffffe,0);
        *(undefined4 *)(param_1 + 0x154) = uVar19;
      }
      iVar10 = *(int *)(param_1 + 0x14c);
      local_14 = 0.0;
      local_10 = 0.0;
      local_c = 0.0;
      local_38 = 0;
      local_78 = 0.0;
      local_74 = 0.0;
      local_70 = 0.0;
      local_6c = 0.0;
      if (iVar10 != 0) {
        local_8 = 0.0;
        do {
          **(undefined4 **)(iVar10 + 0x3e4) = 0;
          *(undefined4 *)(iVar10 + 0x3e4) = 0;
          if ((*(int *)(iVar10 + 0x10) != 0) &&
             (*(int *)(iVar10 + 0x3c) == *(int *)(param_1 + 0x10))) {
            iVar12 = *(int *)(*(int *)(iVar10 + 0x2c) + 0x130);
            if (*(int *)(iVar10 + 0x3e0) == 1) {
              param_3 = (float *)(*(int *)(*(int *)(iVar10 + 0x80) + 0xa4) * 0x44 + 0x24 +
                                 *(int *)(iVar12 + 0x20));
            }
            else {
              param_3 = (float *)(iVar12 + 0xd0);
            }
            FUN_00459ed0(*param_3 + param_3[3],param_3[4] + param_3[1],param_3[5] + param_3[2]);
            FUN_006763a0(0x3f000000);
            local_6c = param_3[6];
            if ((NAN(local_6c) || NAN(DAT_007cbb50)) != (local_6c == DAT_007cbb50)) {
              FUN_00459ed0(*(float *)(iVar12 + 0xb4) + *(float *)(iVar12 + 0xc0),
                           *(float *)(iVar12 + 0xc4) + *(float *)(iVar12 + 0xb8),
                           *(float *)(iVar12 + 200) + *(float *)(iVar12 + 0xbc));
              FUN_006763a0(0x3f000000);
              local_6c = *(float *)(iVar12 + 0xcc);
            }
            pfVar9 = (float *)FUN_0078dea0((float *)(iVar10 + 0xec));
            fVar7 = DAT_007cbb50;
            local_78 = *pfVar9 - *param_2;
            local_74 = pfVar9[1] - param_2[1];
            local_70 = pfVar9[2] - param_2[2];
            fVar3 = *(float *)(iVar10 + 0xec);
            fVar4 = (*(float *)(iVar10 + 0xf4) * *(float *)(iVar10 + 0xf4) +
                    *(float *)(iVar10 + 0xf0) * *(float *)(iVar10 + 0xf0) + fVar3 * fVar3) *
                    local_6c * local_6c;
            fVar3 = local_78 * local_4c + local_74 * local_48 + local_70 * local_44;
            fVar14 = local_4c * fVar3 - local_78;
            fVar6 = local_48 * fVar3 - local_74;
            fVar5 = local_44 * fVar3 - local_70;
            fVar14 = fVar14 * fVar14 + fVar6 * fVar6 + fVar5 * fVar5;
            local_14 = local_78;
            local_10 = local_74;
            local_c = local_70;
            if (fVar14 <= fVar4) {
              fVar14 = SQRT(fVar4 - fVar14);
              local_64 = fVar3 + fVar14;
              if ((local_64 < DAT_007cbb50 == (NAN(local_64) || NAN(DAT_007cbb50))) &&
                 (fVar3 = fVar3 - fVar14, fVar3 <= local_2c)) {
                piVar13 = (int *)(*(int *)(param_1 + 0x150) + (int)local_8);
                *piVar13 = iVar10;
                fVar14 = DAT_007cbb50;
                if (fVar7 < fVar3) {
                  fVar14 = fVar3;
                }
                piVar13[1] = (int)fVar14;
                piVar13[2] = (int)local_64;
                piVar13[3] = *(int *)(iVar10 + 0x3f0);
                *(uint *)(*(int *)(param_1 + 0x154) + local_38 * 4) = local_38;
                local_38 = local_38 + 1;
                local_8 = (float)((int)local_8 + 0x10);
              }
            }
          }
          iVar10 = *(int *)(iVar10 + 1000);
        } while (iVar10 != 0);
      }
      local_64 = (float)local_38;
      FUN_00721080(local_38,*(undefined4 *)(param_1 + 0x150));
      piVar13 = (int *)0x0;
      local_50 = (int *)0x0;
      if (*param_4 < _DAT_007cba60 != (NAN(*param_4) || NAN(_DAT_007cba60))) {
        local_2c = local_2c * *param_4;
      }
      local_b8 = 1.0;
      local_b4 = 0.0;
      local_b0 = 0.0;
      local_ac = 0;
      local_a8 = 0.0;
      local_a4 = 1.0;
      local_a0 = 0.0;
      local_9c = 0;
      local_98 = 0.0;
      fVar3 = local_44 * param_2[2] + local_4c * *param_2 + local_48 * param_2[1];
      local_94 = 0.0;
      local_90 = 1.0;
      local_8c = 0;
      local_88 = 0.0;
      local_84 = 0.0;
      local_80 = 0.0;
      local_7c = 0x3f800000;
      local_58 = 0;
      do {
        local_38 = 0;
        param_3 = (float *)local_2c;
        if (local_64 != 0.0) {
          do {
            local_54 = (int *)(*(int *)(*(int *)(param_1 + 0x154) + local_38 * 4) * 0x10 +
                              *(int *)(param_1 + 0x150));
            if (local_58 == 0) {
              if ((float)param_3 <= (float)local_54[1]) break;
LAB_00718425:
              local_5c = *local_54;
              iVar10 = *(int *)(*(int *)(local_5c + 0x2c) + 0x130);
              local_40 = iVar10;
              if (*(int *)(local_5c + 0x3e0) == 1) {
                local_8 = *(float *)(iVar10 + 0x50);
                local_30 = 0.0;
                if (*(int *)((int)local_8 + 0x20) != 0) {
                  local_34 = 0.0;
                  do {
                    iVar10 = *(int *)((int)local_8 + 0x24) + (int)local_34;
                    if ((*(byte *)(*(int *)((int)local_8 + 0x24) + (int)local_34) & 8) == 0) {
                      iVar12 = *(int *)(local_5c + 0x8c);
                      if (iVar12 == 0) {
                        iVar12 = *(int *)(local_5c + 0x88);
                      }
                      if (*(int *)(iVar12 + (uint)*(ushort *)(iVar10 + 4) * 4) != 0) {
                        fVar14 = *(float *)(local_5c + 0x18c);
                        if ((uint)*(ushort *)(iVar10 + 8) < *(uint *)(local_40 + 0x54)) {
                          fVar14 = fVar14 * *(float *)((uint)*(ushort *)(iVar10 + 8) * 0x50 + 0x3c +
                                                      *(int *)(local_5c + 0x90));
                        }
                        if (*(short *)(iVar10 + 0xe) != 0) {
                          fVar14 = fVar14 * *(float *)((uint)*(ushort *)
                                                              (*(int *)(local_40 + 0xa8) +
                                                              (uint)*(ushort *)(iVar10 + 0x14) * 2)
                                                       * 0x20 + 0xc + *(int *)(local_5c + 0x98));
                        }
                        if ((fVar14 < DAT_007cbb50 == (NAN(fVar14) || NAN(DAT_007cbb50))) &&
                           ((NAN(fVar14) || NAN(DAT_007cbb50)) == (fVar14 == DAT_007cbb50))) {
                          fVar14 = (float)((uint)*(ushort *)(iVar10 + 4) * 0x20 +
                                          *(int *)((int)local_8 + 0x1c));
                          local_18 = fVar14;
                          if (*(uint *)(param_1 + 0x160) < (uint)*(ushort *)((int)fVar14 + 6)) {
                            if ((*(int *)(param_1 + 0x15c) != 0) &&
                               (iVar10 = *(int *)(param_1 + 0x15c) + -4, iVar10 != 0)) {
                              FUN_0063cda0(iVar10,s_delete___007f6e74,0xffffffff,0);
                            }
                            if (*(int *)(param_1 + 0x160) == 0) {
                              *(undefined4 *)(param_1 + 0x160) = 1;
                            }
                            if (*(uint *)(param_1 + 0x160) < (uint)*(ushort *)((int)fVar14 + 6)) {
                              do {
                                uVar16 = *(int *)(param_1 + 0x160) << 1;
                                *(uint *)(param_1 + 0x160) = uVar16;
                              } while (uVar16 < *(ushort *)((int)fVar14 + 6));
                            }
                            iVar10 = *(int *)(param_1 + 0x160);
                            piVar13 = (int *)FUN_0063cb90(iVar10 * 0xc + 4,
                                                          s___AVC3Vector_NTempest___007f6334,
                                                          0xfffffffe,0);
                            if (piVar13 == (int *)0x0) {
                              piVar13 = (int *)0x0;
                            }
                            else {
                              *piVar13 = iVar10;
                              piVar13 = piVar13 + 1;
                              piVar11 = piVar13;
                              if (-1 < iVar10 + -1) {
                                do {
                                  *piVar11 = 0;
                                  piVar11[1] = 0;
                                  piVar11[2] = 0;
                                  iVar10 = iVar10 + -1;
                                  piVar11 = piVar11 + 3;
                                } while (iVar10 != 0);
                              }
                            }
                            *(int **)(param_1 + 0x15c) = piVar13;
                          }
                          local_24 = 0.0;
                          local_28 = 0.0;
                          local_3c = 0.0;
                          if (*(short *)((int)fVar14 + 6) != 0) {
                            local_1c = 0.0;
                            do {
                              iVar10 = (uint)*(ushort *)
                                              (*(int *)((int)local_8 + 4) +
                                              ((uint)*(ushort *)((int)local_18 + 4) + (int)local_3c)
                                              * 2) * 0x30;
                              fVar14 = *(float *)(iVar10 + 0xc + *(int *)(local_40 + 0x48));
                              iVar10 = iVar10 + *(int *)(local_40 + 0x48);
                              if ((fVar14 != local_24) || (*(float *)(iVar10 + 0x10) != local_28)) {
                                local_28 = *(float *)(iVar10 + 0x10);
                                pfVar9 = (float *)(((uint)local_28 & 0xff) * 0x40 +
                                                  *(int *)(local_5c + 0x84));
                                local_20 = (float)(uint)*(byte *)(iVar10 + 0xc);
                                pbVar15 = (byte *)(iVar10 + 0xd);
                                local_80 = (float)(int)local_20 * _DAT_007cd524;
                                local_b8 = local_80 * *pfVar9;
                                local_b4 = local_80 * pfVar9[1];
                                local_b0 = local_80 * pfVar9[2];
                                local_a8 = local_80 * pfVar9[4];
                                local_a4 = local_80 * pfVar9[5];
                                local_a0 = local_80 * pfVar9[6];
                                local_98 = local_80 * pfVar9[8];
                                local_94 = local_80 * pfVar9[9];
                                local_90 = local_80 * pfVar9[10];
                                local_88 = local_80 * pfVar9[0xc];
                                local_84 = local_80 * pfVar9[0xd];
                                local_80 = local_80 * pfVar9[0xe];
                                do {
                                  local_24 = fVar14;
                                  if (*pbVar15 == 0) break;
                                  local_20 = (float)(uint)*pbVar15;
                                  pfVar9 = (float *)((uint)pbVar15[4] * 0x40 +
                                                    *(int *)(local_5c + 0x84));
                                  pbVar15 = pbVar15 + 1;
                                  fVar4 = (float)(int)local_20 * _DAT_007cd524;
                                  local_b8 = fVar4 * *pfVar9 + local_b8;
                                  local_b4 = fVar4 * pfVar9[1] + local_b4;
                                  local_b0 = fVar4 * pfVar9[2] + local_b0;
                                  local_a8 = fVar4 * pfVar9[4] + local_a8;
                                  local_a4 = fVar4 * pfVar9[5] + local_a4;
                                  local_a0 = fVar4 * pfVar9[6] + local_a0;
                                  local_98 = fVar4 * pfVar9[8] + local_98;
                                  local_94 = fVar4 * pfVar9[9] + local_94;
                                  local_90 = fVar4 * pfVar9[10] + local_90;
                                  local_88 = fVar4 * pfVar9[0xc] + local_88;
                                  local_84 = fVar4 * pfVar9[0xd] + local_84;
                                  local_80 = fVar4 * pfVar9[0xe] + local_80;
                                } while (pbVar15 + (-0xc - iVar10) < (byte *)0x4);
                              }
                              FUN_0078dea0(&local_b8);
                              if (local_58 != 0) {
                                local_14 = local_b8 * *(float *)(iVar10 + 0x14) +
                                           local_a8 * *(float *)(iVar10 + 0x18) +
                                           local_98 * *(float *)(iVar10 + 0x1c) + local_14;
                                local_10 = local_b4 * *(float *)(iVar10 + 0x14) +
                                           local_a4 * *(float *)(iVar10 + 0x18) +
                                           local_94 * *(float *)(iVar10 + 0x1c) + local_10;
                                local_c = local_c + local_b0 * *(float *)(iVar10 + 0x14) +
                                                    local_a0 * *(float *)(iVar10 + 0x18) +
                                                    local_90 * *(float *)(iVar10 + 0x1c);
                              }
                              pfVar9 = (float *)(*(int *)(local_68 + 0x15c) + (int)local_1c);
                              local_1c = (float)((int)local_1c + 0xc);
                              fVar14 = (local_c * local_44 +
                                       local_48 * local_10 + local_4c * local_14) - fVar3;
                              *pfVar9 = local_14 - local_4c * fVar14;
                              pfVar9[1] = local_10 - local_48 * fVar14;
                              pfVar9[2] = fVar14;
                              local_3c = (float)((int)local_3c + 1);
                            } while ((uint)local_3c < (uint)*(ushort *)((int)local_18 + 6));
                          }
                          puVar17 = (ushort *)
                                    (*(int *)((int)local_8 + 0xc) +
                                    (uint)*(ushort *)((int)local_18 + 8) * 2);
                          local_60 = puVar17 + *(ushort *)((int)local_18 + 10);
                          param_1 = local_68;
                          if (puVar17 < local_60) {
                            iVar10 = *(int *)(local_68 + 0x15c);
                            uVar16 = (uint)*(ushort *)((int)local_18 + 4);
                            do {
                              pfVar9 = (float *)(iVar10 + (*puVar17 - uVar16) * 0xc);
                              pfVar1 = (float *)(iVar10 + (puVar17[1] - uVar16) * 0xc);
                              pfVar2 = (float *)(iVar10 + (puVar17[2] - uVar16) * 0xc);
                              fVar14 = (*pfVar1 - *pfVar9) * (pfVar2[1] - pfVar9[1]) -
                                       (pfVar1[1] - pfVar9[1]) * (*pfVar2 - *pfVar9);
                              fVar4 = ABS(fVar14);
                              if (fVar4 < _DAT_007d651c == (NAN(fVar4) || NAN(_DAT_007d651c))) {
                                local_28 = pfVar2[1] - param_2[1];
                                local_18 = *pfVar1 - *param_2;
                                local_24 = pfVar1[1] - param_2[1];
                                local_1c = *pfVar2 - *param_2;
                                local_3c = _DAT_007cba60 / fVar14;
                                local_20 = (local_18 * local_28 - local_1c * local_24) * local_3c;
                                if (local_20 < DAT_007cbb50 == (NAN(local_20) || NAN(DAT_007cbb50)))
                                {
                                  local_1c = ((pfVar9[1] - param_2[1]) * local_1c -
                                             (*pfVar9 - *param_2) * local_28) * local_3c;
                                  if ((((local_1c < DAT_007cbb50 ==
                                         (NAN(local_1c) || NAN(DAT_007cbb50))) &&
                                       (fVar14 = ((*pfVar9 - *param_2) * local_24 -
                                                 (pfVar9[1] - param_2[1]) * local_18) * local_3c,
                                       fVar14 < DAT_007cbb50 == (NAN(fVar14) || NAN(DAT_007cbb50))))
                                      && (local_18 = local_20 * pfVar9[2] +
                                                     local_1c * pfVar1[2] + fVar14 * pfVar2[2],
                                         local_18 < DAT_007cbb50 ==
                                         (NAN(local_18) || NAN(DAT_007cbb50)))) &&
                                     (((local_58 != 0 &&
                                       ((local_50 == (int *)0x0 || (local_50[3] != local_54[3]))))
                                      || (local_18 <= (float)param_3)))) {
                                    local_50 = local_54;
                                    param_3 = (float *)local_18;
                                  }
                                }
                              }
                              puVar17 = puVar17 + 3;
                            } while (puVar17 < local_60);
                          }
                        }
                      }
                    }
                    local_30 = (float)((int)local_30 + 1);
                    local_34 = (float)((int)local_34 + 0x18);
                  } while ((uint)local_30 < (uint)*(float *)((int)local_8 + 0x20));
                }
              }
              else {
                if (*(uint *)(param_1 + 0x160) < *(uint *)(iVar10 + 0xf4)) {
                  if ((*(int *)(param_1 + 0x15c) != 0) &&
                     (iVar12 = *(int *)(param_1 + 0x15c) + -4, iVar12 != 0)) {
                    FUN_0063cda0(iVar12,s_delete___007f6e74,0xffffffff,0);
                  }
                  if (*(int *)(param_1 + 0x160) == 0) {
                    *(undefined4 *)(param_1 + 0x160) = 1;
                  }
                  if (*(uint *)(param_1 + 0x160) < *(uint *)(iVar10 + 0xf4)) {
                    do {
                      uVar16 = *(int *)(param_1 + 0x160) << 1;
                      *(uint *)(param_1 + 0x160) = uVar16;
                    } while (uVar16 < *(uint *)(iVar10 + 0xf4));
                  }
                  iVar10 = *(int *)(param_1 + 0x160);
                  piVar13 = (int *)FUN_0063cb90(iVar10 * 0xc + 4,s___AVC3Vector_NTempest___007f6334,
                                                0xfffffffe,0);
                  if (piVar13 == (int *)0x0) {
                    piVar13 = (int *)0x0;
                  }
                  else {
                    *piVar13 = iVar10;
                    piVar13 = piVar13 + 1;
                    piVar11 = piVar13;
                    if (-1 < iVar10 + -1) {
                      do {
                        *piVar11 = 0;
                        piVar11[1] = 0;
                        piVar11[2] = 0;
                        iVar10 = iVar10 + -1;
                        piVar11 = piVar11 + 3;
                      } while (iVar10 != 0);
                    }
                  }
                  *(int **)(param_1 + 0x15c) = piVar13;
                }
                uVar16 = 0;
                if (*(int *)(local_40 + 0xf4) != 0) {
                  local_60 = (ushort *)(local_5c + 0xec);
                  iVar10 = 0;
                  do {
                    FUN_0078dea0(local_60);
                    pfVar9 = (float *)(*(int *)(param_1 + 0x15c) + iVar10);
                    uVar16 = uVar16 + 1;
                    iVar10 = iVar10 + 0xc;
                    fVar14 = (local_cc * local_48 + local_d0 * local_4c + local_c8 * local_44) -
                             fVar3;
                    *pfVar9 = local_d0 - local_4c * fVar14;
                    pfVar9[1] = local_cc - local_48 * fVar14;
                    pfVar9[2] = fVar14;
                  } while (uVar16 < *(uint *)(local_40 + 0xf4));
                }
                puVar17 = *(ushort **)(local_40 + 0xf0);
                local_60 = puVar17 + *(int *)(local_40 + 0xec);
                for (; puVar17 < local_60; puVar17 = puVar17 + 3) {
                  iVar10 = *(int *)(param_1 + 0x15c);
                  pfVar9 = (float *)(iVar10 + (uint)*puVar17 * 0xc);
                  pfVar1 = (float *)(iVar10 + (uint)puVar17[1] * 0xc);
                  pfVar2 = (float *)(iVar10 + (uint)puVar17[2] * 0xc);
                  fVar14 = (*pfVar1 - *pfVar9) * (pfVar2[1] - pfVar9[1]) -
                           (pfVar1[1] - pfVar9[1]) * (*pfVar2 - *pfVar9);
                  fVar4 = ABS(fVar14);
                  if (fVar4 < _DAT_007d651c == (NAN(fVar4) || NAN(_DAT_007d651c))) {
                    local_20 = pfVar2[1] - param_2[1];
                    local_28 = *pfVar1 - *param_2;
                    local_1c = pfVar1[1] - param_2[1];
                    local_30 = *pfVar2 - *param_2;
                    local_34 = _DAT_007cba60 / fVar14;
                    local_24 = (local_28 * local_20 - local_30 * local_1c) * local_34;
                    if (local_24 < DAT_007cbb50 == (NAN(local_24) || NAN(DAT_007cbb50))) {
                      local_30 = ((pfVar9[1] - param_2[1]) * local_30 -
                                 (*pfVar9 - *param_2) * local_20) * local_34;
                      if ((((local_30 < DAT_007cbb50 == (NAN(local_30) || NAN(DAT_007cbb50))) &&
                           (fVar14 = ((*pfVar9 - *param_2) * local_1c -
                                     (pfVar9[1] - param_2[1]) * local_28) * local_34,
                           fVar14 < DAT_007cbb50 == (NAN(fVar14) || NAN(DAT_007cbb50)))) &&
                          (local_8 = local_24 * pfVar9[2] +
                                     local_30 * pfVar1[2] + fVar14 * pfVar2[2],
                          local_8 < DAT_007cbb50 == (NAN(local_8) || NAN(DAT_007cbb50)))) &&
                         (((local_58 != 0 &&
                           ((local_50 == (int *)0x0 || (local_50[3] != local_54[3])))) ||
                          (local_8 <= (float)param_3)))) {
                        local_50 = local_54;
                        param_3 = (float *)local_8;
                      }
                    }
                  }
                }
              }
            }
            else if ((piVar13 == (int *)0x0) ||
                    (((uint)piVar13[3] <= (uint)local_54[3] &&
                     ((piVar13[3] != local_54[3] || ((float)local_54[1] < (float)param_3))))))
            goto LAB_00718425;
            local_38 = local_38 + 1;
            piVar13 = local_50;
          } while (local_38 < (uint)local_64);
          if (piVar13 != (int *)0x0) break;
        }
        if (param_5 == 0) break;
        local_58 = local_58 + 1;
        param_3 = (float *)local_2c;
      } while (local_58 < 2);
      FUN_00410d20();
      if (*(int *)(param_1 + 0x14c) != 0) {
        FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f784,0x4d2,
                     s__m_hitList_0083f7d0,0,1);
      }
      *(undefined4 *)(param_1 + 0x148) = 0;
      if (piVar13 != (int *)0x0) {
        *param_4 = (float)param_3 / local_2c;
        return *(undefined4 *)(*piVar13 + 0x3ec);
      }
      return 0;
    }
    for (iVar10 = *(int *)(param_1 + 0x14c); iVar10 != 0; iVar10 = *(int *)(iVar10 + 1000)) {
      **(undefined4 **)(iVar10 + 0x3e4) = 0;
      *(undefined4 *)(iVar10 + 0x3e4) = 0;
    }
    if (*(int *)(param_1 + 0x14c) == 0) goto LAB_00717ebc;
    uVar19 = 0x337;
  }
  else {
    for (iVar10 = *(int *)(param_1 + 0x14c); iVar10 != 0; iVar10 = *(int *)(iVar10 + 1000)) {
      **(undefined4 **)(iVar10 + 0x3e4) = 0;
      *(undefined4 *)(iVar10 + 0x3e4) = 0;
    }
    if (*(int *)(param_1 + 0x14c) == 0) goto LAB_00717ebc;
    uVar19 = 0x327;
  }
  FUN_0063d520(0x85100000,s_C__build_buildWoW_Engine_Source__0083f784,uVar19,s__m_hitList_0083f7d0,0
               ,1);
LAB_00717ebc:
  *(undefined4 *)(param_1 + 0x148) = 0;
  return 0;
}


