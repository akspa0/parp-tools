# Active Context: Return to Working Spatial Clustering

**CRITICAL FINDING:** The "Data Web" packed field hypothesis was a **dead end** that destroys spatial coherence and produces "imploding" fragmented geometry. We have abandoned this approach entirely.

## 🚫 **DEAD END APPROACHES - DO NOT REVISIT:**

### ❌ Byte-Splitting/Packed Field Grouping
- **SurfaceRefIndex[lower 8-bit] ↔ SurfaceKey[lower 8-bit]** linkage
- **Grouping by upper/lower bytes** of any PM4 fields
- **ParentIndex byte decomposition** and container theories
- **Result:** Destroys spatial relationships, produces fragmented "imploding" geometry
- **User Diagnosis:** "distributes the verts in a weird almost -imploding- style mess"

### Why Byte-Splitting Fails
1. **Destroys Natural Object Boundaries:** PM4's SurfaceKey grouping already represents coherent objects
2. **Artificial Fragmentation:** Splits vertices based on arbitrary bit patterns instead of spatial relationships
3. **Loses Geometric Coherence:** Creates meaningless objects that don't correspond to actual building geometry

## ✅ **CORRECT APPROACH: Working Spatial Clustering**

### 🏗️ What Actually Works
The **Pm4SpatialClusteringAssembler** already produces correct building-scale objects by:

1. **Respecting Natural PM4 Boundaries:** Uses **full SurfaceKey values** as natural grouping units
2. **Size-Based Filtering:** Applies intelligent triangle count limits (10-50k triangles) to avoid oversized/multi-building groups
3. **Spatial Coherence:** Maintains geometric relationships through proper clustering
4. **Cross-Tile Filtering:** Eliminates cross-tile references that break object boundaries

### 🔧 Technical Enhancements to Apply
While preserving the working spatial clustering logic, enhance it with discoveries from the Data Web exploration:

1. **Improved Linkage:** Better MSLK ↔ MSUR connections (without byte-splitting)
2. **Surface Deduplication:** Logic to avoid duplicate geometry when combining sources
3. **Robust Diagnostics:** Comprehensive logging and progress tracking
4. **Correct Field Usage:** Proper MSUR properties (IndexCount, MsviFirstIndex)
5. **Cycle Detection:** Robust handling of recursive/circular references

## Current Action Plan
1. **✅ DONE:** Register working spatial clustering command (`spatial-clustering`)
2. **🎯 NEXT:** Run and validate working spatial clustering baseline
3. **🔧 ENHANCE:** Apply technical improvements without destroying spatial coherence
4. **📊 MEASURE:** Compare enhanced vs baseline outputs for quality improvement
