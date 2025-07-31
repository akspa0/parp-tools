# Progress for parpToolbox

## Strategic Pivot: "Data Web" Analysis (2025-07-30)

All previous development on the "Data Web" analysis has been paused. A fundamental re-evaluation of our approach has led to a complete strategic pivot back to spatial clustering. Our previous assumptions about the "Data Web" were incorrect, leading to fragmented and incomplete results.

### What Works (Historical Context)

The following components were built based on previous, now-obsolete assumptions. They are preserved as historical context but are not part of the current development path.

-   **Various Exporter Implementations:** Multiple exporters (`Pm4SpatialClusteringAssembler`, `Pm4CrossTileObjectAssembler`, etc.) were created to test different hypotheses (spatial clustering, cross-tile linking). While they provided valuable insights, they failed to produce correct results because they did not address the core data structure.
-   **Analysis Tools:** Several analyzers (`Pm4DataBandingAnalyzer`, `Pm4IndexPatternAnalyzer`) were built, confirming that data is complex and interconnected across tiles. These tools paved the way for the new "Data Web" hypothesis.
-   **Core Infrastructure:** The project build system, CLI framework, and output management are stable.

### What Works

- **Pm4SpatialClusteringAssembler**: Produces correct building-scale objects using full SurfaceKey grouping
- PM4 file loading and basic chunk parsing
- Scene graph construction with vertices, links, surfaces  
- Cross-tile vertex reference resolution
- Size-based filtering (10-50k triangles) to avoid oversized groups
- CLI command infrastructure with working `spatial-clustering` command

### Abandoned Dead Ends

- **Byte-splitting/packed field grouping**: Destroys spatial coherence, produces "imploding" geometry
- **Data Web hypothesis**: Over-engineered approach that fragments natural object boundaries
- **SurfaceRefIndex[lower 8-bit] ↔ SurfaceKey[lower 8-bit]** linkage: Artificially splits coherent objects.

### What's Left to Build (Current Plan)

Our work has been reset to focus exclusively on spatial clustering. No exporter development will occur until this phase is complete.

1.  **[IN PROGRESS] Update Memory Bank:** Aligning all documentation (`activeContext.md`, `progress.md`, etc.) with the new "Data Web" strategy.
2.  **[NEXT] Develop a Comprehensive Key-Structure Analyzer:**
    *   **Goal:** Create a new CLI command (`analyze-pm4-keys`) to decode the "Data Web."
    *   **Functionality:** This tool will analyze any key field in any chunk, looking for packed hierarchical data and cross-chunk index relationships.
3.  **[PENDING] Map the Data Web:** Use the analyzer's output to create a definitive, visual map of the PM4 data structure.
4.  **[PENDING] Implement the Final, Correct Exporter:** Once the data is fully understood, build a new exporter from scratch based on the data map.

### Known Issues

-   **Fundamentally Incorrect Assumptions:** All previous work was based on a flawed understanding of the PM4 data model. This is the root cause of all export failures.
-   **Tool Fragmentation:** The codebase contains numerous obsolete exporters and analysis tools that need to be archived or removed once the new plan is fully implemented.
