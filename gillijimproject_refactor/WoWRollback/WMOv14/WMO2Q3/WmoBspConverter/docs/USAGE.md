# WMO to Quake 3 Converter

Converts World of Warcraft WMO (World Map Object) v14 files to Quake 3 formats for use with GtkRadiant and Q3Map2.

## Quick Start

### Export WMO to ASE + Map

```bash
WmoBspConverter <wmo_file> --ase --split-groups -d <output_dir> -v
```

### Deploy to Quake 3

Copy the output folder contents to your Quake 3 `baseq3/` directory:

```
<output_dir>/
├── models/wmo/*.ase    → baseq3/models/wmo/
├── textures/wmo/*.tga  → baseq3/textures/wmo/
└── <name>.map          → baseq3/maps/
```

### Open in GtkRadiant

1. Launch GtkRadiant
2. File → Open → `baseq3/maps/<name>.map`
3. Geometry displays as wireframe misc_model entities

### Compile to BSP

**Option A: Via GtkRadiant**
- Bsp → BSP (full vis) or Bsp → BSP → -vis -light

**Option B: Via q3map2 CLI**
```bash
q3map2 -meta "baseq3/maps/<name>.map"
q3map2 -vis "baseq3/maps/<name>.bsp"
q3map2 -light "baseq3/maps/<name>.bsp"
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--ase` | Export ASE models + .map with misc_model entities |
| `--split-groups` | Export each WMO group as separate ASE file |
| `--map` | Export brush-based .map file (alternative to ASE) |
| `-d <dir>` | Output directory |
| `-v` | Verbose logging |
| `--list-groups` | List all groups in the WMO |
| `--to-v17` | Convert v14 WMO to v17 format |

## Output Structure

```
output/
├── models/
│   └── wmo/
│       ├── <name>_group000.ase
│       ├── <name>_group001.ase
│       └── ...
├── textures/
│   └── wmo/
│       ├── texture1.tga
│       └── ...
└── <name>.map
```

## Troubleshooting

### "PicoLoadModel: Failed loading model"
- Ensure ASE files are in `baseq3/models/wmo/`
- Verify the .map and models are in the same baseq3 tree

### "Failed to load shader" warnings
- Copy `textures/wmo/` to `baseq3/textures/wmo/`
- Textures should be `.tga` format

### Brush-based .map issues
If using `--map` instead of `--ase`, you may encounter plane errors. The ASE/misc_model approach is recommended for complex geometry.
