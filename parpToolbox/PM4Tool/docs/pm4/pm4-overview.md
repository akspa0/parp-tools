# PM4 File Overview

## What is a PM4 File?
PM4 files are navigation and mesh data files used in World of Warcraft, primarily for ADT (terrain) tiles. They encode complex navigation meshes, object placements, and pathfinding data for map tiles.

## Why PM4 Files Matter
- They provide the data needed to reconstruct navigation meshes and object placements for WoW maps.
- They are essential for reverse engineering, modding, and archival of WoW's world structure.
- PM4 files are a bridge between terrain (ADT) and world models (WMO/M2), enabling placement and navigation logic.

## PM4 in WoWToolbox
- WoWToolbox parses PM4 files to extract mesh geometry, boundary data, and object placements.
- The toolkit provides APIs and tools for analyzing, visualizing, and comparing PM4 data with WMO/ADT assets.
- PM4 analysis is central to reconstructing historical map layouts and understanding asset usage in WoW's development. 