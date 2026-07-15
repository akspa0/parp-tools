
undefined4 __fastcall FUN_0071eab0(int param_1)

{
  int iVar1;
  int iVar2;
  undefined4 uVar3;
  int *piVar4;
  undefined4 *puVar5;
  uint uVar6;
  uint uVar7;
  int iVar8;
  int *piVar9;
  uint local_10;
  uint local_c;
  uint local_8;
  
  local_c = 0x100;
  if (*(int *)(*(int *)(param_1 + 4) + 0x1024) != 0) {
    iVar1 = FUN_0058d450();
    uVar7 = (*(int *)(iVar1 + 0x60) - 0x1fU) / 3;
    if (uVar7 < 0x100) {
      local_c = uVar7;
    }
  }
  iVar1 = *(int *)(param_1 + 0x130);
  *(undefined4 *)(param_1 + 0x138) = 0;
  local_8 = 0;
  local_10 = 0;
  if (*(int *)(iVar1 + 0x4c) != 0) {
    iVar8 = 0;
    do {
      uVar7 = *(uint *)(*(int *)(iVar1 + 0x50) + 0x28 + iVar8);
      iVar2 = *(int *)(iVar1 + 0x50) + iVar8;
      if ((uVar7 <= local_c) && (local_8 <= uVar7)) {
        *(int *)(param_1 + 0x138) = iVar2;
        local_8 = *(uint *)(iVar2 + 0x28);
      }
      local_10 = local_10 + 1;
      iVar8 = iVar8 + 0x2c;
    } while (local_10 < *(uint *)(iVar1 + 0x4c));
  }
  iVar1 = *(int *)(param_1 + 0x138);
  if (iVar1 == 0) {
    return 0;
  }
  if (*(uint *)(iVar1 + 8) == 0) {
    uVar3 = 1;
  }
  else {
    uVar3 = (undefined4)(0x10000 / (ulonglong)*(uint *)(iVar1 + 8));
  }
  *(undefined4 *)(param_1 + 0x15c) = uVar3;
  local_10 = 0;
  if (*(int *)(iVar1 + 0x18) != 0) {
    iVar8 = 0;
    do {
      uVar7 = local_8 / *(ushort *)(*(int *)(iVar1 + 0x1c) + 0xc + iVar8);
      if (uVar7 < *(uint *)(param_1 + 0x15c)) {
        *(uint *)(param_1 + 0x15c) = uVar7;
      }
      local_10 = local_10 + 1;
      iVar8 = iVar8 + 0x20;
    } while (local_10 < *(uint *)(iVar1 + 0x18));
  }
  if (*(int *)(param_1 + 0x15c) == 0) {
    *(undefined4 *)(param_1 + 0x15c) = 1;
  }
  *(undefined4 *)(param_1 + 0x158) = 1;
  iVar1 = *(int *)(iVar1 + 0x18);
  if (iVar1 != 0) {
    piVar4 = (int *)FUN_0063cb90(iVar1 * 0x20 + 4,s___AUM2Region___0083f228,0xfffffffe,0);
    if (piVar4 == (int *)0x0) {
      piVar9 = (int *)0x0;
    }
    else {
      *piVar4 = iVar1;
      piVar9 = piVar4 + 1;
      if (-1 < iVar1 + -1) {
        do {
          piVar4[6] = 0;
          piVar4[7] = 0;
          piVar4[8] = 0;
          iVar1 = iVar1 + -1;
          piVar4 = piVar4 + 8;
        } while (iVar1 != 0);
      }
    }
    *(int **)(param_1 + 0x154) = piVar9;
    if (piVar9 == (int *)0x0) {
      return 0;
    }
    piVar4 = *(int **)(*(int *)(param_1 + 0x138) + 0x1c);
    for (uVar7 = (uint)(*(int *)(*(int *)(param_1 + 0x138) + 0x18) << 5) >> 2; uVar7 != 0;
        uVar7 = uVar7 - 1) {
      *piVar9 = *piVar4;
      piVar4 = piVar4 + 1;
      piVar9 = piVar9 + 1;
    }
    for (iVar1 = 0; iVar1 != 0; iVar1 = iVar1 + -1) {
      *(char *)piVar9 = (char)*piVar4;
      piVar4 = (int *)((int)piVar4 + 1);
      piVar9 = (int *)((int)piVar9 + 1);
    }
  }
  iVar1 = 0;
  if ((*(int *)(*(int *)(param_1 + 0x138) + 8) != 0) &&
     (*(int *)(*(int *)(param_1 + 4) + 0x1024) != 0)) {
    puVar5 = (undefined4 *)
             FUN_0063cb90(*(int *)(*(int *)(param_1 + 0x138) + 0x20) << 2,s__I_007f6f24,0xfffffffe,0
                         );
    *(undefined4 **)(param_1 + 0x150) = puVar5;
    if (puVar5 == (undefined4 *)0x0) {
      return 0;
    }
    for (uVar7 = *(uint *)(*(int *)(param_1 + 0x138) + 0x20) & 0x3fffffff; uVar7 != 0;
        uVar7 = uVar7 - 1) {
      *puVar5 = 0;
      puVar5 = puVar5 + 1;
    }
    for (iVar8 = 0; iVar8 != 0; iVar8 = iVar8 + -1) {
      *(undefined1 *)puVar5 = 0;
      puVar5 = (undefined4 *)((int)puVar5 + 1);
    }
    iVar8 = *(int *)(param_1 + 0x138);
    uVar7 = 0;
    if (*(int *)(iVar8 + 0x20) != 0) {
      do {
        uVar3 = FUN_0071f150(*(int *)(iVar8 + 0x24) + iVar1,
                             (uint)*(ushort *)(*(int *)(iVar8 + 0x24) + 4 + iVar1) * 0x20 +
                             *(int *)(iVar8 + 0x1c));
        *(undefined4 *)(*(int *)(param_1 + 0x150) + uVar7 * 4) = uVar3;
        iVar8 = *(int *)(param_1 + 0x138);
        uVar7 = uVar7 + 1;
        iVar1 = iVar1 + 0x18;
      } while (uVar7 < *(uint *)(iVar8 + 0x20));
    }
  }
  puVar5 = (undefined4 *)
           FUN_0063cb90(*(int *)(*(int *)(param_1 + 0x130) + 0x5c) << 2,s__PAUHTEXTURE_____00836c74,
                        0xfffffffe,0);
  *(undefined4 **)(param_1 + 0x13c) = puVar5;
  if (puVar5 != (undefined4 *)0x0) {
    for (uVar7 = *(uint *)(*(int *)(param_1 + 0x130) + 0x5c) & 0x3fffffff; uVar7 != 0;
        uVar7 = uVar7 - 1) {
      *puVar5 = 0;
      puVar5 = puVar5 + 1;
    }
    for (iVar1 = 0; iVar1 != 0; iVar1 = iVar1 + -1) {
      *(undefined1 *)puVar5 = 0;
      puVar5 = (undefined4 *)((int)puVar5 + 1);
    }
    iVar1 = *(int *)(param_1 + 0x130);
    uVar7 = **(uint **)(*(int *)(param_1 + 4) + 0x1030);
    local_8 = 0;
    if (*(int *)(iVar1 + 0x5c) != 0) {
      local_c = 0;
      do {
        if (1 < *(uint *)(*(int *)(iVar1 + 0x60) + 8 + local_c)) {
          uVar6 = (uint)*(ushort *)(*(int *)(iVar1 + 0x60) + local_c + 4) << 3;
          uVar7 = uVar7 ^ (uVar6 ^ uVar7) & 8;
          uVar3 = 1;
          uVar7 = uVar7 ^ (uVar6 ^ uVar7) & 0x10;
          FUN_0040fae0(1);
          uVar3 = FUN_0044dad0(uVar7,uVar3);
          *(undefined4 *)(*(int *)(param_1 + 0x13c) + local_8 * 4) = uVar3;
        }
        iVar1 = *(int *)(param_1 + 0x130);
        local_8 = local_8 + 1;
        local_c = local_c + 0x10;
      } while (local_8 < *(uint *)(iVar1 + 0x5c));
    }
    return 1;
  }
  return 0;
}


