# Cluster Data Enhancement Requirements

## Current State ✅
Clusters export basic aggregate data:
- `clusterId` - Cluster identifier
- `objectCount` - Number of objects in cluster
- `centroid` - Average position
- `bounds` - Bounding box
- `radius` - Cluster radius
- `isStamp` - Whether cluster represents a placement stamp
- `hasConsecutiveIds` - Whether UniqueIDs are consecutive

## Missing Data for Rich Popups ❌

### 1. Object List
**Requirement:** Export array of objects in each cluster

```json
{
  "clusterId": 0,
  "objectCount": 3,
  "objects": [
    {
      "uniqueId": 49415,
      "fileName": "AERIEPEAKSFIR02.M2",
      "assetPath": "WORLD\\LORDAERON\\...",
      "position": { "x": 555.45, "y": 57.8, "z": 721.78 },
      "scale": 1.0
    }
  ]
}
```

**Use Case:** Show list of objects in cluster popup, allow clicking to highlight

### 2. Object Relationships
**Requirement:** Export spatial/semantic relationships between objects

```json
{
  "relationships": [
    {
      "type": "SameAsset",
      "objects": [49415, 49416, 49417],
      "description": "3x same tree model"
    },
    {
      "type": "DesignKit",
      "objects": [49415, 49416],
      "kitName": "ForestDressing",
      "description": "Objects from same design kit"
    }
  ]
}
```

**Use Case:** 
- Identify prefab patterns
- Show design kit membership
- Highlight object hierarchies

### 3. UniqueID Range
**Requirement:** Export min/max UniqueID for filtering

```json
{
  "uniqueIdRange": {
    "min": 49415,
    "max": 49559
  }
}
```

**Use Case:** Filter sedimentary layers by cluster

### 4. Asset Distribution
**Requirement:** Count of unique assets

```json
{
  "assetDistribution": {
    "AERIEPEAKSFIR02.M2": 3,
    "AERIEPEAKSFIR03.M2": 2
  },
  "uniqueAssetCount": 2
}
```

**Use Case:** Show asset diversity in popup

## Implementation Priority

### High Priority (Phase 1)
1. ✅ Object list with basic info (uniqueId, fileName, position)
2. ✅ UniqueID range (min/max)

### Medium Priority (Phase 2)  
3. Asset distribution (count per model)
4. Basic relationships (SameAsset, proximity groups)

### Low Priority (Phase 3)
5. Design kit membership
6. Hierarchical relationships
7. Temporal analysis (if version data available)

## Export Format

```json
{
  "map": "development",
  "tile": { "row": 1, "col": 1 },
  "clusterCount": 5,
  "clusters": [
    {
      "clusterId": 0,
      "objectCount": 3,
      "centroid": {...},
      "bounds": {...},
      "radius": 23.09,
      "isStamp": true,
      "hasConsecutiveIds": false,
      
      // NEW: Detailed object data
      "uniqueIdRange": {
        "min": 49415,
        "max": 49559
      },
      "objects": [
        {
          "uniqueId": 49415,
          "fileName": "AERIEPEAKSFIR02.M2",
          "assetPath": "WORLD\\LORDAERON\\...",
          "position": { "x": 555.45, "y": 57.8, "z": 721.78 },
          "rotation": { "x": 0, "y": 0, "z": 0 },
          "scale": 1.0,
          "kind": "M2"
        }
      ],
      "assetDistribution": {
        "AERIEPEAKSFIR02.M2": 3
      },
      "relationships": [
        {
          "type": "SameAsset",
          "objectIds": [49415, 49416, 49417]
        }
      ]
    }
  ]
}
```

## Viewer Benefits

With enhanced data:
- ✅ Show scrollable object list in popup
- ✅ Click object → highlight on map
- ✅ Filter sedimentary layers by cluster UniqueID range
- ✅ Show asset reuse patterns
- ✅ Identify placement stamps vs. manual placement
- ✅ Cross-reference with design kit data

## Performance Considerations

- Object lists increase JSON size (~50-200 bytes per object)
- Gzip compression will handle repetitive data well
- Load on demand (already viewport-based)
- Consider separate detail endpoint for very large clusters (>100 objects)
