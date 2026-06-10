# WoW Coordinate System Explained

## Overview

WoW uses a **right-handed coordinate system** with specific conventions that are important for correctly interpreting visualization outputs.

## WoW World Coordinate System

### Axis Definitions

- **X-axis**: Points NORTH (positive) / SOUTH (negative)
- **Y-axis**: Points WEST (positive) / EAST (negative)  
- **Z-axis**: Vertical height (0 = sea level)
- **Origin**: Center of the map

### Map Bounds

```
Top-left corner:     X = +17066.66,  Y = +17066.66  (Northwest)
Top-right corner:    X = +17066.66,  Y = -17066.66  (Northeast)
Bottom-left corner:  X = -17066.66,  Y = +17066.66  (Southwest)
Bottom-right corner: X = -17066.66,  Y = -17066.66  (Southeast)
```

**Total map size:** 34133.33 yards (17066.66 × 2)

### Map Grid Structure

- **64×64 blocks** (ADT tiles)
- Each block: **533.33 yards** (1600 feet)
- Each block divided into **16×16 chunks**
- Each chunk: **33.33 yards** (100 feet)

## Visualization Coordinate Transform

### What WoWDataPlot Does

When creating visualizations, we transform WoW world coordinates to plot coordinates:

```csharp
// WoW World → Plot Transform
plotX = -worldY  // Flip Y-axis so East is on right
plotY = -worldX  // Flip X-axis so North is on top
```

### Plot Orientation

Our 2D top-down plots show:

```
         North (top)
            ↑
            |
West ←------+------→ East
            |
            ↓
        South (bottom)
```

**In the plot:**
- **Top of image** = North (positive WoW X)
- **Right of image** = East (negative WoW Y)
- **Bottom of image** = South (negative WoW X)
- **Left of image** = West (positive WoW Y)

This matches how you see the world when looking at the in-game map!

## Tile Index Calculation

### Formula (from wowdev.wiki)

```csharp
tileX = floor(32 - (worldX / 533.33))
tileY = floor(32 - (worldY / 533.33))
```

### Examples

| World Coordinates | Tile Indices | Location |
|-------------------|--------------|----------|
| (17066, 17066) | [0, 0] | Northwest corner |
| (17066, -17066) | [0, 63] | Northeast corner |
| (-17066, 17066) | [63, 0] | Southwest corner |
| (-17066, -17066) | [63, 63] | Southeast corner |
| (0, 0) | [32, 32] | Map center |

## Player Perspective

If you're playing a character facing north:

```csharp
Forward = Vector3(1, 0, 0)   // North (positive X)
Right   = Vector3(0, -1, 0)  // East (negative Y)
Up      = Vector3(0, 0, 1)   // Vertical (positive Z)
```

## Practical Examples

### Example 1: Stormwind City

Approximate Stormwind coordinates: `(-8913, 895, 100)`

- **X = -8913**: South of map center (negative X)
- **Y = 895**: Slightly west of center (positive Y)  
- **Z = 100**: 100 yards above sea level
- **Tile**: `[floor(32 - (-8913/533.33)), floor(32 - (895/533.33))]` = `[48, 30]`

### Example 2: Interpreting Plots

If you see a cluster of points in the **top-right** of a plot:

- **Top** = North (positive X in WoW)
- **Right** = East (negative Y in WoW)
- **Result**: This is the **Northeast** region of the map

### Example 3: Layer Analysis

When analyzing layers per tile:

```
Tile [32, 32] (center) shows:
  Layer 0-999:    Blue dots scattered northeast
  Layer 1000-1999: Green dots in southwest quadrant
  Layer 2000-2999: Red dots filling entire tile
```

**Interpretation:**
- Early content (0-999) was placed in northeast part of center tile
- Mid content (1000-1999) added to southwest
- Later content (2000-2999) filled entire center area

## Common Confusions

### "Why is Y-axis backwards?"

WoW's Y-axis increases going **west**, which is counter-intuitive. We flip it in visualizations so that:
- East (the "right" direction on a compass) appears on the right of the plot
- This matches standard map conventions

### "Why flip both axes?"

To create a proper **top-down view** that matches what you see in-game:
- Without flips: North would be at bottom, west on right (confusing!)
- With flips: North at top, east on right (natural map view)

### "Do tile indices match in-game?"

**Partially.** The tile file naming format is:

```
<MapName>_<X>_<Y>.adt
```

Where X and Y are the tile indices calculated by our formula. However:
- **Our visualization uses plot coordinates** (transformed for clarity)
- **File paths use tile indices** (0-63 range)
- **World coordinates are absolute** (±17066 range)

## References

- [WoWDev Wiki - ADT v18 Coordinate System](https://wowdev.wiki/ADT/v18#An_important_note_about_the_coordinate_system_used)
- [Right-handed coordinate system](https://en.wikipedia.org/wiki/Right-hand_rule#Coordinates)

## TL;DR

**When looking at WoWDataPlot visualizations:**

✅ **Top = North** (where positive X is)  
✅ **Right = East** (where negative Y is)  
✅ **Bottom = South** (where negative X is)  
✅ **Left = West** (where positive Y is)

**This matches how you read a real-world map!**
