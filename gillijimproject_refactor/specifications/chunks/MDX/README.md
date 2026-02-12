# MDX Chunks — Build 0.7.0.3694

This folder documents the MDX chunk model used by build **0.7.0.3694** from direct Ghidra decompilation.

## Chunks documented

- `VERS.md`
- `MODL.md`
- `SEQS.md`
- `GEOS.md`
- `GEOA.md`
- `TEXS.md`
- `MTLS.md`
- `ATCH.md`
- `BONE.md`
- `HTST.md`
- `PRE2.md`
- `RIBB.md`
- `LITE.md`
- `PIVT.md`
- `TXAN.md`
- `CLID.md`
- `CAMS.md`

## Notes

- `MDLX` magic is asserted in `FUN_004220e0`.
- Core loader dispatch is in `FUN_004211c0` and callees:
	- `FUN_0044cec0` (`TEXS`), `FUN_0044d100` (`MTLS`), `FUN_0044d730` (`GEOS` + `GEOA`)
	- `FUN_0044e7e0` (`ATCH`), `FUN_00447bf0` (`PRE2`), `FUN_0044a180` (`RIBB`)
	- `FUN_00449330` (`LITE`), `FUN_00421a00` (`MODL` + `SEQS`), `FUN_00421c60` (`PIVT`)
	- `FUN_00448b20` (`CAMS`), `FUN_004459c0` (`CLID`), `FUN_00421310` (`BONE` + `HTST` + `TXAN`)
