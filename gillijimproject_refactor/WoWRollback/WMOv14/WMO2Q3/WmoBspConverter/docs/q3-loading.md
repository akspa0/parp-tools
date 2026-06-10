# Loading in Quake 3

Checklist:
- Use console: \sv_pure 0
- Check paths: \fs_basepath and \fs_homepath
- Place castle01.bsp under <baseq3>/maps of the active base path.
- If not found, pack a PK3:
  - zzz_castle01.pk3 with path maps/castle01.bsp
  - Optionally include scripts/ and textures/
- Then: \map castle01 (no .bsp extension).
