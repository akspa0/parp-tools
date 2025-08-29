# Offset & Size Safety Checklist

- All chunk offsets are 0 or within parent bounds.
- Emitted sizes equal computed payload sizes.
- Each block starts at 4-byte aligned offset.
- MCIN entries point to valid MCNK blocks.
- MH2O layer offsets (heightmap/mask) lie within MH2O chunk bounds.
- No `MCLQ` offsets present in MCNK/MHDR after write.
- Total file size equals last block end (aligned).
