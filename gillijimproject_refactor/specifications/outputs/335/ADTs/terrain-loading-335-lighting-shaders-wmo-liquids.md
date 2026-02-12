# Wow.exe 3.3.5.12340 - Lighting, Shaders, and WMO Liquids (Ghidra)

Scope: lighting/shader setup and WMO liquid selection, with evidence from the binary.

## WMO liquid selection and defaults

### Function addresses
- FUN_00793d20: WMO liquid instance creation and material selection
- FUN_008a1fa0: Liquid material bank (LiquidType.dbc)
- FUN_008a28f0: Liquid settings bank (LiquidMaterial.dbc)

### Decompiled evidence

```c
// FUN_00793d20
uVar4 = FUN_00431f30();              // returns *(param_1 + 0x144)
...
if (((int)uVar4 < DAT_00ad4074) || (DAT_00ad4070 < (int)uVar4) || (local_8 == 0)) {
  FUN_004b5040(2,2,"WMO: Liquid type [%d] not found, defaulting to water!",uVar4);
  uVar4 = 1;
  ... // fallback to water material
}
...
*(undefined4 *)(uVar10 + 0x68) = FUN_008a1b00();
**(undefined4 **)(uVar10 + 0x68) = FUN_008a1fa0(uVar4);   // material
*(undefined4 *)(*(int *)(uVar10 + 0x68) + 4) = FUN_008a28f0(uVar4); // settings
```

```c
// FUN_00431f30
return *(undefined4 *)(param_1 + 0x144); // liquid type id
```

```c
// FUN_008a1fa0
iVar3 = (**(code **)(*DAT_00d439ec + 4))(param_1); // LiquidType.dbc lookup
uVar1 = *(uint *)(iVar3 + 0x38);                   // material selector
... // choose material class based on type and caps
```

```c
// FUN_008a28f0
// creates settings using LiquidMaterial.dbc, defaults to water on miss
FUN_004b5040(2,2,"Settings Bank: Liquid type [%d] not found, defaulting to water!",param_1);
```

### Definitive answer
- WMO liquid type is read from a structure field at **offset 0x144** (via `FUN_00431f30`).
- That id is used to select:
  - **Material** via `LiquidType.dbc` (`FUN_008a1fa0`, uses field at offset `+0x38` in the LiquidType record).
  - **Settings** via `LiquidMaterial.dbc` (`FUN_008a28f0`).
- If the id is not found, the client **falls back to liquid type 1 (water)** and logs: `WMO: Liquid type [%d] not found, defaulting to water!`.

### Likely causes of wrong WMO liquids
- WMO liquid type id not matching `LiquidType.dbc` entries (bad mapping or wrong field).
- Using a name-based liquid mapping instead of the **numeric type id at offset 0x144**.

---

## WMO liquid rendering objects

### Function addresses
- FUN_007cf200: collects liquid chunks and builds a shared instance
- FUN_007cefd0: allocates chunk buffers
- FUN_007cf790: allocates chunk-liquid records

### Decompiled evidence

```c
// FUN_007cf200
puVar11 = (undefined4 *)FUN_008a1b00();
*puVar11 = FUN_008a1fa0(*(undefined4 *)(param_1 + 4));     // material
puVar11[1] = FUN_008a28f0(*(undefined4 *)(param_1 + 4));   // settings
puVar11[4] = FUN_007d5120(0);                              // client environment
...
*(undefined4 **)(DAT_00d2dd3c[uVar6] + 0x58) = puVar11;     // per-chunk bind
```

### Definitive answer
- WMO liquids are rendered by **chunk groups** that share a material/settings pair keyed by the liquid type id.
- Each chunk points to a **shared liquid instance** (`DAT_00d2dd3c[*] + 0x58`) created in `FUN_007cf200`.

---

## Liquid shaders

### Function addresses
- FUN_008a3e00: procedural water shaders
- FUN_008a3f70: standard water shaders

### Decompiled evidence

```c
// FUN_008a3e00
"vsLiquidProcWater%s"
"psLiquidProcWater%s"
```

```c
// FUN_008a3f70
"vsLiquidWater"
"psLiquidWater"
```

### Definitive answer
- Liquid rendering selects vertex/pixel shaders by name:
  - Procedural: `vsLiquidProcWater%s` / `psLiquidProcWater%s`
  - Standard: `vsLiquidWater` / `psLiquidWater`
- These shaders are loaded via the standard shader loader (`Shaders\\Vertex`, `Shaders\\Pixel`).

---

## Lighting controls and map object light LOD

### Function addresses
- FUN_0078e400: video option registration
- FUN_0078ded0: map object light LOD validation

### Decompiled evidence

```c
// FUN_0078e400
FUN_00767fc0("MaxLights","Max number of hardware lights",...);
FUN_00767fc0("mapObjLightLOD","Map object light LOD",...,FUN_0078ded0,...);
```

```c
// FUN_0078ded0
uVar1 = FUN_0076f0d0(param_3);
if (uVar1 < 3) { _DAT_00d1c414 = uVar1; return 1; }
FUN_00765270("MapObjLightLOD must be 0-2",0);
```

### Definitive answer
- Lighting and map-object light LOD are controlled via CVars (`MaxLights`, `mapObjLightLOD`).
- `mapObjLightLOD` is constrained to **0..2** in the client.
- The light DBs used by the client include `Light.dbc`, `LightParams.dbc`, `LightSkybox.dbc`, `LightIntBand.dbc`, `LightFloatBand.dbc`.

---

## Shader effect loading

### Function addresses
- FUN_00876d90: shader effect manager

### Decompiled evidence

```c
// FUN_00876d90
"Shaders\\Effects\\%s"
```

### Definitive answer
- Effect shaders are loaded from `Shaders\\Effects\\<name>` via `ShaderEffectManager`.

---

## Actionable checks for the WMO liquid bug

1. Ensure you read the **liquid type id** from the correct WMO data structure and map it through `LiquidType.dbc`.
2. Verify the id is not being treated as a material index or name hash.
3. If the type id is missing, the client defaults to **water type 1** (explains wrong liquids).

### Confidence
High for liquid type selection, defaults, and shader names. Medium for lighting pipeline integration (CVars and DBCs confirmed; full render path not traced here).
