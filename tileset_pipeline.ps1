# =============================================================================
# tileset_pipeline.ps1
# Full tileset pattern mining pipeline:
#   1. Harvest BLPs from merged index -> PNGs (skip existing)
#   2. GPU FFT mining -> pattern_library_gpu.json
#   3. Dedup by design kit + scale bucket -> pattern_library_deduped.json
#   4. Render brush previews + collage -> brush_panel_deduped.png
# =============================================================================

param(
    [string]$MergedIndex   = "I:\parp\parp-tools\output\ml-training\v10_tileset_database\merged_tileset_index.json",
    [string]$PngDir        = "I:\parp\parp-tools\output\ml-training\v10_tileset_pngs",
    [string]$OutputDir     = "I:\parp\parp-tools\output\ml-training\v10_tileset_patterns",
    [string]$PythonExe     = "I:\parp\parp-tools\gillijimproject_refactor\.venv-train\Scripts\python.exe",
    [string]$ConverterProj = "I:\parp\parp-tools\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj",
    [switch]$SkipHarvest,
    [switch]$SkipMining,
    [switch]$SkipDedup,
    [switch]$SkipPreviews
)

$ErrorActionPreference = "Stop"
$start = Get-Date

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " tileset pattern mining pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Step 1: Harvest BLP -> PNG
# ---------------------------------------------------------------------------
if (-not $SkipHarvest) {
    Write-Host "[1/4] Harvesting tileset BLPs -> PNG..." -ForegroundColor Yellow
    Write-Host "      Input: $MergedIndex"
    Write-Host "      Output: $PngDir"
    Write-Host ""

    dotnet run --project $ConverterProj -c Debug -- `
        harvest-tileset-blps `
        --input $MergedIndex `
        --output-dir $PngDir

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Harvest failed with exit code $LASTEXITCODE" -ForegroundColor Red
        Write-Host "Continuing with whatever PNGs exist..." -ForegroundColor DarkYellow
    }
} else {
    Write-Host "[1/4] SKIPPING harvest (--SkipHarvest)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Step 1b: Regenerate manifest from existing PNGs
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "      Regenerating manifest from existing PNGs..." -ForegroundColor Yellow

& $PythonExe -c @"
import os, glob, json, time
d = r'$PngDir'
entries = []
for dn in sorted(os.listdir(d)):
    dp = os.path.join(d, dn)
    if not os.path.isdir(dp): continue
    for png in sorted(glob.glob(os.path.join(dp, '*.png'))):
        nm = os.path.splitext(os.path.basename(png))[0]
        entries.append({
            'name': nm,
            'png_path': png.replace('\\', '/'),
            'design_kit': dn,
            'era_tag': '',
            'relative_path': ''
        })
m = {'schema_version': 'v10', 'total_harvested': len(entries),
     'total_errors': 0, 'entries': entries}
mpath = os.path.join(d, 'harvest_manifest.json')
with open(mpath, 'w') as f: json.dump(m, f, indent=2)
print(f'Manifest: {len(entries)} entries')
"@
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Step 2: GPU pattern mining
# ---------------------------------------------------------------------------
if (-not $SkipMining) {
    Write-Host "[2/4] GPU pattern mining..." -ForegroundColor Yellow

    $MinerScript = "I:\parp\parp-tools\gillijimproject_refactor\src\WoWMapConverter\scripts\mine_patterns_gpu.py"
    $Manifest = Join-Path $PngDir "harvest_manifest.json"

    & $PythonExe $MinerScript `
        --manifest $Manifest `
        --output-dir $OutputDir `
        --device cuda `
        --min-periodicity 0.10 `
        --brush-size 96 `
        --no-previews

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: GPU mining failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[2/4] SKIPPING GPU mining (--SkipMining)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Step 3: Dedup + sort by design kit
# ---------------------------------------------------------------------------
if (-not $SkipDedup) {
    Write-Host ""
    Write-Host "[3/4] Deduplicating patterns by design kit + scale bucket..." -ForegroundColor Yellow

    $GpuJson = Join-Path $OutputDir "pattern_library_gpu.json"
    $DedupJson = Join-Path $OutputDir "pattern_library_deduped.json"

    & $PythonExe -c @"
import json, os

with open(r'$GpuJson') as f:
    data = json.load(f)
patterns = data['patterns']

from collections import defaultdict
groups = defaultdict(list)
for p in patterns:
    kit = p.get('design_kit', 'Unknown')
    tx, ty = p['tile_size_x'], p['tile_size_y']
    m = max(tx, ty)
    if m <= 8: bucket = 'micro'
    elif m <= 24: bucket = 'small'
    elif m <= 48: bucket = 'meso'
    elif m <= 96: bucket = 'large'
    else: bucket = 'macro'
    groups[f'{kit}|{bucket}'].append(p)

deduped = []
for key, entries in sorted(groups.items()):
    best = max(entries, key=lambda x: x['periodicity_score'])
    best['_group_key'] = key
    best['_group_count'] = len(entries)
    deduped.append(best)

deduped.sort(key=lambda x: (x['design_kit'], -x['periodicity_score']))

out = {
    'schema_version': 'v10-gpu-patterns-deduped.v1',
    'total_patterns': len(deduped),
    'original_count': len(patterns),
    'dedup_method': 'design_kit + scale_bucket, best periodicity per group',
    'patterns': deduped,
}
with open(r'$DedupJson', 'w') as f:
    json.dump(out, f, indent=2)

# Report
kit_counts = {}
for p in deduped:
    k = p['design_kit']
    kit_counts[k] = kit_counts.get(k, 0) + 1

print(f'Patterns: {len(patterns)} -> {len(deduped)} deduped')
print(f'Design kits: {len(kit_counts)}')
print()
for k, c in sorted(kit_counts.items(), key=lambda x: -x[1])[:15]:
    print(f'  {k}: {c}')
"@
    Write-Host ""
} else {
    Write-Host "[3/4] SKIPPING dedup (--SkipDedup)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Step 4: Brush previews + collage
# ---------------------------------------------------------------------------
if (-not $SkipPreviews) {
    Write-Host "[4/4] Rendering brush previews + collage..." -ForegroundColor Yellow

    $DedupJson = Join-Path $OutputDir "pattern_library_deduped.json"
    $Manifest = Join-Path $PngDir "harvest_manifest.json"
    $BrushDir = Join-Path $OutputDir "brushes_deduped"
    $Collage = Join-Path $OutputDir "brush_panel_deduped.png"

    & $PythonExe -c @"
import json, os, numpy as np
from PIL import Image, ImageDraw, ImageFont

with open(r'$DedupJson') as f: data = json.load(f)
with open(r'$Manifest') as f: mf = json.load(f)
png_map = {e['name']: e['png_path'] for e in mf['entries']}

BRUSH_DIR = r'$BrushDir'
BRUSH_SIZE = 80
COLS, OUTDIR = 8, r'$OutputDir'
os.makedirs(BRUSH_DIR, exist_ok=True)

def render_brush(png_path, tx, ty):
    img = Image.open(png_path).convert('RGBA')
    rgba = np.array(img, dtype=np.float32)
    stamp = rgba[:min(ty, rgba.shape[0]), :min(tx, rgba.shape[1])]
    gray = 0.299*stamp[:,:,0] + 0.587*stamp[:,:,1] + 0.114*stamp[:,:,2]
    gmin, gmax = gray.min(), gray.max()
    if gmax > gmin: gray = (gray - gmin) / (gmax - gmin)
    h, w = gray.shape
    dx = np.zeros_like(gray); dy = np.zeros_like(gray)
    dx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    dy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    shade = np.maximum(0.45, 1.0 - np.abs(dx)*0.6 - np.abs(dy)*0.4)
    b = np.clip(96 + 159*(gray - 0.5)*2*shade, 20, 235).astype(np.uint8)
    bump = np.stack([b, np.minimum(b+5,255), np.minimum(b+5,255)], axis=-1).astype(np.uint8)
    img = Image.fromarray(bump).resize((BRUSH_SIZE, BRUSH_SIZE), Image.LANCZOS)
    card = Image.new('RGB', (BRUSH_SIZE+16, BRUSH_SIZE+16), (20,20,24))
    card.paste((48,48,52), [0,0,card.width,1]); card.paste((48,48,52), [0,0,1,card.height])
    card.paste(img, (8,8))
    return card

patterns = sorted(data['patterns'], key=lambda p: (p['design_kit'], -p['periodicity_score']))
rendered = []
for p in patterns:
    ppath = png_map.get(p['texture_name'])
    if not ppath or not os.path.exists(ppath): continue
    try:
        b = render_brush(ppath, p['tile_size_x'], p['tile_size_y'])
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in p['texture_name'])
        fp = os.path.join(BRUSH_DIR, safe + '.png')
        b.save(fp)
        rendered.append((p['periodicity_score'], p['texture_name'], p['design_kit'],
                         p['tile_size_x'], p['tile_size_y'], fp, p.get('_group_count',1)))
    except Exception as e: pass

print(f'Rendered {len(rendered)} brushes')

# Collage
ROWS = (len(rendered) + COLS - 1) // COLS + 1
card_sz = BRUSH_SIZE + 16
header_h = 44
canvas = Image.new('RGB', (COLS*card_sz, ROWS*card_sz + header_h), (12,12,16))
try:
    font = ImageFont.truetype('consola.ttf', 8)
    font_sm = ImageFont.truetype('consola.ttf', 7)
except:
    font = font_sm = ImageFont.load_default()
draw = ImageDraw.Draw(canvas)
draw.rectangle([0, 0, canvas.width, header_h - 4], fill=(24,24,30))
draw.text((6, 4), 'WoWEdit Brush Panel  |  deduped pattern library  |  GPU FFT mining',
          fill=(180,180,180), font=font)
draw.text((6, 18), f'{len(rendered)} unique stamps across {len(set(p[2] for p in rendered))} design kits  |  woWViewer',
          fill=(100,100,100), font=font_sm)

kit_colors = {}
for idx, (score, name, kit, tx, ty, fp, gc) in enumerate(rendered[:COLS*(ROWS-1)]):
    if kit not in kit_colors:
        kit_colors[kit] = (40 + hash(kit)%40, 40 + hash(kit+'x')%40, 40 + hash(kit+'xx')%40)
    c = idx % COLS
    r = idx // COLS
    x = c * card_sz
    y = header_h + r * card_sz
    try:
        b = Image.open(fp).convert('RGB')
        canvas.paste(b, (x, y))
    except: pass
    draw.text((x+2, y+card_sz-15), name[:16], fill=(130,130,130), font=font)
    info = f'{kit[:14]}  t{tx}x{ty}  p{score:.0%}'
    if gc > 1: info += f' ({gc})'
    draw.text((x+2, y+card_sz-27), info, fill=(85,85,85), font=font_sm)

cp = os.path.join(OUTDIR, 'brush_panel_deduped.png')
canvas.save(cp)
print(f'Collage: {cp} ({canvas.width}x{canvas.height})')
"@
    Write-Host ""
} else {
    Write-Host "[4/4] SKIPPING previews (--SkipPreviews)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
$elapsed = (Get-Date) - $start
Write-Host "========================================" -ForegroundColor Green
Write-Host " Pipeline complete!" -ForegroundColor Green
Write-Host " Time: $($elapsed.TotalMinutes.ToString('F1')) min" -ForegroundColor Green
Write-Host " Harvested PNGs:     $PngDir" -ForegroundColor Green
Write-Host " Pattern library:    $OutputDir\pattern_library_deduped.json" -ForegroundColor Green
Write-Host " Brush collage:      $OutputDir\brush_panel_deduped.png" -ForegroundColor Green
Write-Host " Individual brushes: $OutputDir\brushes_deduped\" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
