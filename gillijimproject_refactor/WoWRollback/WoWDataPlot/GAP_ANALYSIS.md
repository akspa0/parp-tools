# Gap-Based Layer Detection

## What Changed

**Before (BROKEN):**
- Layers were fixed buckets: 0-99, 100-199, 200-299, etc.
- `--layer-size 100` meant "group into ranges of 100 IDs"
- **Problem:** Arbitrary buckets ignored actual development patterns

**After (FIXED):**
- Layers are **continuous clusters** of UniqueIDs separated by **gaps**
- `--gap-threshold 50` means "split into new layer if UniqueID jumps by more than 50"
- **Benefit:** Gaps tell the story of development pauses!

## How It Works

### Example: Tile [32, 28]

**UniqueIDs present:**
```
42, 43, 44, 45, ..., 87    ← Layer 1 (continuous)
[GAP of 68 IDs]
156, 157, 158, ..., 234    ← Layer 2 (continuous)
[GAP of 655 IDs]
890, 891, 892, ..., 923    ← Layer 3 (continuous)
```

**What the gaps mean:**
- **Gap 1 (88-155):** Development paused, or IDs used elsewhere
- **Gap 2 (235-889):** Major pause, version change, or content removal

**Layers detected:**
```
Layer 42-87:     46 placements (Blue)   ← Early work
Layer 156-234:   79 placements (Green)  ← Mid work
Layer 890-923:   34 placements (Red)    ← Late work
```

## Why Gaps Matter

**Gaps are data!** They tell you:

1. **Development Timeline**
   - Small gaps (< 10) = continuous work
   - Medium gaps (10-100) = daily/weekly pauses
   - Large gaps (100+) = version boundaries or content removal

2. **Content Evolution**
   - Densely packed IDs = rapid iteration
   - Sparse IDs with gaps = experimental/test content
   - Huge gaps = major refactors or removed content

3. **Collaboration Patterns**
   - Multiple tight clusters = different devs working in parallel
   - Single continuous range = one dev working sequentially

## CLI Parameters

### `--gap-threshold` (default: 50)

**What it does:** Splits into new layer when UniqueID jumps by more than this value

**Examples:**

**Small threshold (10):**
```bash
--gap-threshold 10
```
- Creates many small layers
- Detects even brief pauses
- Best for: Fine-grained timeline analysis

**Medium threshold (50 - RECOMMENDED):**
```bash
--gap-threshold 50
```
- Balanced layer detection
- Ignores small gaps, catches major pauses
- Best for: General archaeology

**Large threshold (200):**
```bash
--gap-threshold 200
```
- Creates few large layers
- Only major version boundaries detected
- Best for: High-level overview

## Real-World Example: Azeroth

Your Azeroth analysis with `--gap-threshold 50`:

**Before (broken fixed buckets):**
```
Layer 0-99:      scattered data
Layer 100-199:   scattered data
Layer 200-299:   scattered data
...
6,138 useless buckets with arbitrary splits
```

**After (gap-based detection):**
```
Tile [32, 28]:
  Layer 42-87:     Development burst A
  Layer 156-234:   Development burst B (after pause)
  Layer 890-923:   Development burst C (after long pause)

Tile [30, 35]:
  Layer 0-12:      Different timeline!
  Layer 500-723:   Completely different pattern!
```

**Each tile tells its own story!**

## Interpretation Guide

### Dense Continuous Layers
```
Layer 1000-1523:  524 placements
```
**Interpretation:** Intense focused development. One dev or team working steadily.

### Sparse Layers with Gaps
```
Layer 42-87:      12 placements
[GAP of 200]
Layer 288-301:    5 placements
[GAP of 1500]
Layer 1802-1809:  3 placements
```
**Interpretation:** Experimental content. Testing, removed, or placeholder objects.

### Version Boundaries
```
Layer 0-2345:     5,231 placements (Alpha 0.5.3)
[GAP of 10,000]
Layer 12,500-15,678: 2,143 placements (Alpha 0.6.0)
```
**Interpretation:** Clear version cutoff. Major content addition in later version.

## Usage

**Default (recommended):**
```bash
dotnet run --project WoWDataPlot -- visualize \
  --wdt Azeroth.wdt \
  --output-dir Azeroth_gaps \
  --gap-threshold 50
```

**Fine-grained (detect small pauses):**
```bash
dotnet run --project WoWDataPlot -- visualize \
  --wdt Azeroth.wdt \
  --output-dir Azeroth_detailed \
  --gap-threshold 10
```

**Coarse (major versions only):**
```bash
dotnet run --project WoWDataPlot -- visualize \
  --wdt Azeroth.wdt \
  --output-dir Azeroth_versions \
  --gap-threshold 500
```

## What You'll See

**Tile Images:**
- Each color = one continuous development burst
- More colors = more pauses/phases
- Single color = continuous work session

**Analysis JSON:**
```json
{
  "tiles": [
    {
      "tileX": 32,
      "tileY": 28,
      "layers": [
        {
          "name": "42-87",
          "minUniqueId": 42,
          "maxUniqueId": 87,
          "placementCount": 46
        },
        {
          "name": "156-234",
          "minUniqueId": 156,
          "maxUniqueId": 234,
          "placementCount": 79
        }
      ]
    }
  ]
}
```

**Now you can:**
1. See the actual UniqueID ranges used per tile
2. Identify gaps in development
3. Detect version boundaries
4. Find experimental content
5. Track collaboration patterns

**Gaps are not noise - they are the archaeological record of development!** 🏺
