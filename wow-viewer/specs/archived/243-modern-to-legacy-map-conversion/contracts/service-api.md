# Contract — ModernToLegacyMapConversionService

Namespace: `WowViewer.Core.IO.Maps` (library-first, Constitution II).

```csharp
public sealed class ModernToLegacyMapConversionService
{
    // FR-008: validate the route before any write.
    public MapConversionValidationResult ValidateRoute(
        MapConversionSourceFormat source,
        MapConversionTargetFormat target);

    // FR-004/FR-006: convert a batch of maps for one target under a generated project folder.
    public ConversionRun Convert(
        IReadOnlyList<SourceMap> maps,
        MapConversionTargetFormat target,
        string outputRoot,
        ModernToLegacyConversionOptions options);

    // FR-007: resolve + copy referenced assets and produce the manifest.
    public AssetManifest BuildAssetManifest(
        SourceMap map,
        string outputRoot,
        bool includeAssets);
}

public sealed record ModernToLegacyConversionOptions(
    bool IncludeAssets = false,
    int TargetLayerCapacity = 4,
    bool Verbose = false);
```

## Guarantees

- **Deterministic**: identical inputs → byte-identical outputs (SC-004).
- **Isolated**: one map's failure never aborts the batch (FR-004).
- **Non-destructive**: `outputRoot` is a generated project folder; client data is never written
  (FR-006).
- **Reported**: every chunk emits a `MergeRecord`; every unresolved texture is listed (FR-003).
- **No god-class growth**: the service owns all state; `ViewerApp`/`WorldScene` gain no members
  (AGENTS.md §10).

## Errors

| Condition | Behaviour |
|---|---|
| Unsupported (source, target) pair | `ValidateRoute` returns `IsSupported: false`; `Convert` throws before writing (FR-008). |
| Map has zero occupied tiles | `MapConversionResult.Status = Skipped`; run continues. |
| Texture FileDataID unresolved | Recorded in `MergeRecord.UnresolvedTextures`; tile still converts. |
| Asset unresolved (CDN-less) | Recorded in `AssetManifest.Unresolved` with a reason; map output still written. |
