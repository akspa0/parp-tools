# Feature Specification: Terrain Brush Signature Classification

**Feature Branch**: `132-terrain-brush-signature-classification`

**Created**: 2026-08-04

**Status**: Draft

**Input**: User description: "There is a class of terrain that is both textured and sculpted in 3d, that are matched pairs, and then there are far more instances of the texture work having been re-done, fully, removing this bond between the 3d digital paintbrush (WoWEdit itself, the texture tilesets were all painted in z-brush, to ensure that the tileset artwork was literally artwork they threw templates onto via photoshop actions, to build large datasets of textures, in the late 1990s!). We should be classifying terrain based on this relationship, in all our datasets and also to help train our models. Once we can attribute patterns in the image to known fractal brushes, building out a model that understands minimap terrain patterns will be able to rebuild based solely on the relationship of the terrain and painted artwork primitives. That's the pattern we weren't looking for, a macro pattern and relationship that exists and sometimes is a broken relationship that can be mended."

## Context

### The known pattern

The WoW terrain was built using fractal brushes in WoWEdit (circa 1999-2004). These brushes simultaneously affected:

1. **Heightmap** (MCVT) — the 3D sculpting of the terrain surface
2. **Alpha layers** (MCAL) — the texture blend masks that determine which tileset textures appear where
3. **Texture tilesets** — the actual BLP artwork painted onto the terrain

The relationship between these three is not random. A fractal brush stroke in the heightmap often has a corresponding signature in the alpha layers and texture placement. When textures were re-done (e.g., Cataclysm's full texture revamp), the alpha layers were replaced but the heightmap brush scars remained — creating a **broken relationship** between the 3D shape and the 2D texture.

### The weak signal connection

The archaeology work (spec 127) established that weak-signal tiles carry real relief that was compressed rather than erased. The same mechanism applies here: the Eraser tool in WoWEdit buried old brush data at lower precision levels rather than truly removing it. This creates **nested weak signals** — tiles that appear normal at full scale but contain progressively weaker brush signatures at finer precision levels.

### Three signal classes

| Class | Description | Example |
|-------|-------------|---------|
| **Strong** | Full-height terrain with intact brush-texture relationship | Most of Kalimdor/Azeroth |
| **Normal** | Terrain with visible relief but compressed or partially re-textured | Post-Cataclysm zones, re-tiled areas |
| **Weak** | Near-flat tiles with sub-metre relief, often abandoned work | Alpha squeezed tiles, ocean floor |

The "normal" class is the missing middle — tiles that have enough relief to be usable but whose brush-texture relationship is degraded or broken.

### The November 2001 rescale artifact

The WoW Diary documents a November 2001 world map rescale that added ~30% more landmass everywhere.
DeadminesInstance tiles show a measurable artifact of this rescale: a **horizontal roll of weak signals
at precisely 33.33%** — the first third of the top of the tile. This is the boundary where the old
tile was stretched to fit the new scale.

This pattern is a fossil of the development process. Tiles that predate the rescale carry this
33.33% boundary; tiles created after it do not. Detecting this pattern across all maps and builds
would reveal which tiles are pre-rescale originals and which are post-rescale additions — effectively
a **hidden jigsaw puzzle** of the world's development timeline.

### Concrete example: DeadminesInstance and Westfall

DeadminesInstance contains Westfall's **original alpha masks**, including extra bits that were removed from the live Westfall terrain. The data is rotated and mirrored relative to the original placement. This confirms the development process: artists literally copy/pasted tiny terrain patches from development zones into production maps, often without realigning to chunk or tile boundaries.

Westfall itself no longer has brush scars in the alpha masks — the textures were re-done, but DeadminesInstance preserves the originals. This is the key to understanding the lineage: the alpha masks in development/instance maps are the fossil record of what the production terrain used to look like.

### Why Wow:Classic is NOT the source of truth

Modern Wow:Classic clients do NOT include:
- The buggy geometry present in original retail clients
- Repainted or hidden terrain (the Eraser tool's buried data)
- The development-only zones and instance maps with original alpha masks

Only the original retail clients (1.0.0 through 3.3.5) and the pre-release/beta builds contain the real data. This project is acting as a **preservation archive** — farming the data that Blizzard's own classic re-releases discarded.

### What this enables

Classifying terrain by brush signature relationships allows:

1. **Identifying re-textured zones** — where the 3D shape predates the 2D texture (Westfall)
2. **Recovering original brush intent** — from preserved copies in development/instance maps (DeadminesInstance)
3. **Cross-map alignment** — finding copied-pasted terrain fragments that were rotated/mirrored and placed at non-aligned positions
4. **Tracking lineage** — establishing which tiles in which maps share a common origin, building the game's "DNA" tree
5. **Training a model** that understands the macro-pattern relationship between terrain shape and texture primitives
6. **Mending broken relationships** — reconstructing plausible textures for orphaned 3D shapes from their preserved relatives

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Classify tiles by signal strength into three tiers (Priority: P1)

A researcher can see every tile classified as strong, normal, or weak signal, with the classification criteria stated and the boundary between classes visible in the data.

**Why this priority**: Without the three-tier classification, the normal-signal tiles are invisible — they fall through the gap between "usable" and "weak" and are never examined for brush-texture relationships.

**Independent Test**: Load a map, run the three-tier classifier, and confirm that tiles previously classified as "usable" now split into "strong" and "normal" with measurable criteria.

**Acceptance Scenarios**:

1. **Given** a tile with full height range and intact alpha/texture layers, **When** classified, **Then** it is marked as "strong signal".
2. **Given** a tile with visible relief but re-textured alpha layers (mismatched brush signature), **When** classified, **Then** it is marked as "normal signal".
3. **Given** a tile with sub-metre height range, **When** classified, **Then** it is marked as "weak signal".
4. **Given** the same tile classified twice, **When** compared, **Then** the classification is identical.

---

### User Story 2 - Detect nested weak signals within weak signals (Priority: P2)

A researcher can see whether a weak-signal tile contains multiple tiers of progressively weaker brush data, or is simply a single compressed layer.

**Why this priority**: The Eraser tool hypothesis predicts nested data — each "undo" buried the previous brush stroke rather than removing it. If nested tiers exist, the recovery strategy is different from a single compression.

**Independent Test**: Select a weak-signal tile, run the nested-signal detector, and report how many distinct signal tiers exist and at what precision levels they appear.

**Acceptance Scenarios**:

1. **Given** a weak-signal tile with multiple compression tiers, **When** analyzed, **Then** each tier is reported with its height range and surviving level count.
2. **Given** a weak-signal tile with a single compression tier, **When** analyzed, **Then** only one tier is reported.
3. **Given** a tile with no weak signal, **When** analyzed, **Then** zero tiers are reported.

---

### User Story 3 - Correlate brush scars between heightmap and alpha layers (Priority: P2)

A researcher can see which alpha-layer patterns correlate with heightmap brush scars, and which tiles have broken this correlation (re-textured without re-sculpting).

**Why this priority**: This is the core insight — the brush-texture relationship is the macro-pattern that models should learn. Finding broken relationships identifies where the original data was overwritten.

**Independent Test**: Select a tile, run the brush-scar correlator, and report the correlation score between heightmap features and alpha-layer features.

**Acceptance Scenarios**:

1. **Given** a tile with intact brush-texture relationship, **When** correlated, **Then** the correlation score is high (>0.7).
2. **Given** a tile that was re-textured, **When** correlated, **Then** the correlation score is low (<0.3).
3. **Given** a tile with no alpha layers, **When** correlated, **Then** the result reports "no alpha data" rather than a spurious score.

---

### User Story 4 - Cross-map terrain fragment alignment (Priority: P2)

A researcher can take a tile from one map (e.g., DeadminesInstance) and search all other loaded maps
for matching terrain fragments, detecting rotated, mirrored, or non-aligned copies.

**Why this priority**: The DeadminesInstance example proves that terrain fragments were copy-pasted
between maps with rotation/mirroring and without alignment to chunk or tile boundaries. Finding these
relationships is the only way to establish the lineage of terrain data and recover original textures
for re-textured zones.

**Independent Test**: Take a known DeadminesInstance tile with original Westfall alpha masks, search
Westfall for the matching fragment, and confirm the rotation/mirror transform is detected.

**Acceptance Scenarios**:

1. **Given** a source tile and a target map, **When** the alignment tool searches the target, **Then**
   it returns ranked matches with the detected transform (rotation, mirror, scale, offset).
2. **Given** a terrain fragment that was copy-pasted with rotation, **When** aligned, **Then** the
   rotation angle is detected within 1 degree.
3. **Given** a terrain fragment that was mirrored, **When** aligned, **Then** the mirror axis is
   detected.
4. **Given** a fragment pasted at a non-chunk-aligned position, **When** aligned, **Then** the
   sub-tile offset is reported.
5. **Given** a search with no matching fragment, **When** completed, **Then** the result reports no
   match rather than a false positive.

---

### User Story 5 - Detect pre-rescale tile boundaries (Priority: P2)

A researcher can scan any tile for the 33.33% horizontal weak-signal roll that marks the November 2001
rescale boundary, and build a library of all pre-rescale tiles across all maps and builds.

**Why this priority**: This is a direct, testable prediction from the WoW Diary's documented rescale.
If the 33.33% boundary is real, it partitions the game's tiles into pre-rescale originals and
post-rescale additions — revealing the hidden development timeline.

**Independent Test**: Scan every tile in DeadminesInstance, report which tiles carry the 33.33%
horizontal roll, and confirm the pattern is absent from tiles known to be created after the rescale.

**Acceptance Scenarios**:

1. **Given** a pre-rescale tile, **When** scanned for the 33.33% boundary, **Then** the detector
   reports the boundary position and confidence.
2. **Given** a post-rescale tile, **When** scanned, **Then** the detector reports no boundary.
3. **Given** a tile with the boundary at a non-33.33% position, **When** scanned, **Then** the
   actual boundary position is reported rather than forcing a 33.33% match.
4. **Given** all scanned tiles, **When** the library is built, **Then** it is queryable by map,
   build version, and boundary position.

---

### User Story 6 - Build a model that predicts texture from heightmap shape (Priority: P3)

A researcher can train a model that, given a heightmap patch, predicts the most likely alpha-layer pattern and texture tileset based on learned brush-signature relationships.

**Why this priority**: This is the end goal — a model that can reconstruct plausible textures for orphaned 3D shapes, mending the broken relationship.

**Independent Test**: Train on tiles with intact brush-texture relationships, then test on tiles with broken relationships, and measure whether the predicted texture matches the original pre-retexture state.

**Acceptance Scenarios**:

1. **Given** a trained model and a heightmap patch, **When** queried, **Then** it returns a ranked list of likely alpha-layer patterns.
2. **Given** a tile with a broken brush-texture relationship, **When** the model predicts the texture, **Then** the prediction is closer to the original pre-retexture state than a random baseline.
3. **Given** a tile with no brush scars, **When** queried, **Then** the model reports low confidence rather than guessing.

---

### Edge Cases

- Tiles with zero alpha layers (no texture data to correlate)
- Tiles whose heightmap is flat but alpha layers contain brush scars (inverse relationship)
- Tiles where the brush signature is present in only one of the three channels (height, alpha, texture)
- Cross-tile brush strokes that span tile boundaries
- Builds where the alpha layer format differs (0.5.3 vs 1.x vs 3.x)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST classify every tile into one of three signal classes: strong, normal, or weak, with the classification criteria published alongside each result.
- **FR-002**: The normal-signal class MUST be defined by measurable criteria (height range, surviving height levels, alpha-texture correlation) and MUST NOT be a default catch-all.
- **FR-003**: The nested-signal detector MUST report the number of distinct signal tiers in a tile and the precision level of each tier.
- **FR-004**: The brush-scar correlator MUST compute a correlation score between heightmap features and alpha-layer features, and MUST report "no data" when either is absent.
- **FR-005**: The model MUST accept a heightmap patch as input and return a ranked list of likely alpha-layer patterns with confidence scores.
- **FR-006**: All classification and correlation results MUST be reproducible — the same input must produce the same output.
- **FR-007**: The system MUST handle tiles with missing alpha layers gracefully, reporting the absence rather than fabricating a correlation.
- **FR-008**: The cross-map alignment tool MUST detect rotation, mirror, scale, and sub-tile offset transforms between matching terrain fragments, and MUST report no match when none exists.
- **FR-009**: The rescale-boundary detector MUST scan a tile for the 33.33% horizontal weak-signal roll and report the boundary position, confidence, and whether the tile is classified as pre-rescale or post-rescale.

### Non-Functional Requirements

- **NFR-001**: Classification of a single tile must complete in under 1 second.
- **NFR-002**: The brush-scar correlator must work on any build from 0.5.3 through 3.3.5.
- **NFR-003**: The model must be trainable on a single GPU in under 24 hours.

## Success Criteria

1. **Three-tier classification**: Every tile in the development corpus (616 files) is classified as strong, normal, or weak, with the classification criteria published and reproducible.
2. **Nested signal detection**: At least one weak-signal tile is shown to contain multiple tiers of brush data, with the tier boundaries measured.
3. **Brush-texture correlation**: At least one re-textured zone (e.g., Westfall, DeadminesInstance) is identified where the heightmap brush scars do not match the alpha layers, with the correlation score reported.
4. **Cross-map alignment**: At least one known copy-pasted fragment (DeadminesInstance alpha masks in Westfall) is detected and the transform (rotation, mirror, offset) is reported correctly.
5. **Model prediction**: A trained model predicts alpha-layer patterns for orphaned heightmaps with >60% accuracy vs. a random baseline.
6. **Reproducibility**: All results are reproducible from the same input data without manual intervention.

## Key Entities

### TileSignalClass
- `tile_key`: string (map_tileX_tileY)
- `signal_class`: enum (strong, normal, weak)
- `height_range`: float
- `surviving_height_levels`: int
- `alpha_texture_correlation`: float (0-1, or null if no alpha data)
- `nested_tier_count`: int
- `classification_evidence`: string

### BrushScar
- `tile_key`: string
- `brush_id`: string (fingerprint of the brush pattern)
- `channel`: enum (heightmap, alpha, texture)
- `confidence`: float
- `bounding_box`: (min_x, min_y, max_x, max_y)

### BrushTextureCorrelation
- `tile_key`: string
- `correlation_score`: float
- `is_intact`: bool
- `broken_channel`: enum (alpha_replaced, texture_replaced, both)
- `evidence`: string

### TerrainFragmentMatch
- `source_map`: string
- `source_tile_key`: string
- `target_map`: string
- `target_tile_key`: string
- `detected_rotation_degrees`: float
- `detected_mirror_axis`: string (null if none)
- `sub_tile_offset_x`: float
- `sub_tile_offset_y`: float
- `confidence`: float
- `matched_channels`: list of string (heightmap, alpha, texture)

### LineageEdge
- `source_id`: string (tile_key or fragment_id)
- `target_id`: string
- `relationship`: enum (copy_paste, rotated_copy, mirrored_copy, retextured, resculpted)
- `transform`: string (description of the geometric transform)
- `confidence`: float
- `evidence_file`: string

## Assumptions

1. The Eraser tool in WoWEdit buried data rather than erasing it — this is the user's domain knowledge and is the motivating hypothesis.
2. Fractal brush signatures are detectable in both heightmap and alpha-layer data using existing signal-processing techniques.
3. The three-tier classification (strong/normal/weak) is sufficient for the initial analysis; finer gradations can be added later.
4. The model will be trained on tiles with intact brush-texture relationships and evaluated on tiles with broken relationships.
5. The existing harvest tool (WowViewer.Tool.Harvest) provides all necessary input signals (height_257, mcal_alpha_pack, mcly_texture_ids, minimap_rgb_256).
