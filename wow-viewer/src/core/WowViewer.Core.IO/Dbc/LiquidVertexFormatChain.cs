using System.Collections.Concurrent;
using DBCD;
using DBCD.Providers;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Dbc;

/// <summary>Where a resolved liquid vertex format came from.</summary>
public enum LiquidVertexFormatSource
{
    /// <summary>The field held a liquid vertex format directly (pre-Cataclysm meaning, values 0-3).</summary>
    VertexFormatField = 0,

    /// <summary>The field held a LiquidObject id, resolved through LiquidType and LiquidMaterial.</summary>
    LiquidObjectChain = 1,

    /// <summary>Nothing resolved it. The caller must degrade and report; it must not guess.</summary>
    Unresolved = 2,
}

/// <summary>The outcome of resolving one <c>liquid_object_or_lvf</c> field.</summary>
public readonly record struct LiquidVertexFormatResolution(
    ushort RawValue,
    LiquidVertexFormatSource Source,
    AdtLiquidVertexFormat Format,
    bool Resolved,
    int LiquidTypeId,
    int MaterialId,
    string? FailureReason);

/// <summary>
/// Resolves the second <c>uint16</c> of <c>SMLiquidInstance</c> to a liquid vertex format.
/// </summary>
/// <remarks>
/// <para>
/// In Cataclysm and later that field is <c>liquid_object_or_lvf</c>. Values 0-3 are a vertex format
/// directly; values at or above <see cref="LiquidObjectIdThreshold"/> are a <c>LiquidObject.dbc</c>
/// row id, and the vertex format has to be resolved through
/// LiquidObject -&gt; LiquidType -&gt; LiquidMaterial -&gt; LVF.
/// </para>
/// <para>
/// <b>Why this class exists.</b> Both MH2O decoders in this repository cast the field straight to a
/// vertex-format enum and <c>switch</c> on it <b>with no <c>default</c></b>. Measured on
/// <c>MoPBeta</c> / <c>HawaiiMainLand</c> - 80 root ADTs, 17,461 liquid layers -
/// <b>100% of layers carry a LiquidObject id</b>, so every layer falls through, the height array
/// stays null, and the surface renders as a flat plane at the header's minimum height. The 144 river
/// layers (ids 2325/2333/2372) carry real sloped heightmaps with spreads up to 163 world units, all
/// discarded. Ocean (id 42) is genuinely depth-only, so flat is correct there.
/// See <c>specs/205-mh2o-liquid-object-vertex-format/</c>.
/// </para>
/// <para>
/// <b>An unresolved value is never guessed.</b> It is counted, reported once per distinct value, and
/// the caller degrades to the pre-existing behaviour. The float-plausibility probe that made the
/// diagnosis possible is deliberately not used as a decode rule: it misread 18 of 6,194 ocean layers
/// as height-bearing, which would put bogus geometry on real water.
/// </para>
/// </remarks>
public sealed class LiquidVertexFormatChain
{
    /// <summary>Values at or above this are a LiquidObject.dbc id, not a vertex format.</summary>
    public const int LiquidObjectIdThreshold = 42;

    /// <summary>Highest value that is a liquid vertex format in its own right.</summary>
    public const int MaxDirectVertexFormat = 3;

    private readonly DbcLiquidObjectTable _liquidObjects;
    private readonly DbcLiquidMaterialTable _liquidMaterials;
    private readonly IReadOnlyDictionary<int, int> _materialIdByLiquidTypeId;
    private readonly ConcurrentDictionary<ushort, int> _unresolvedCounts = new();

    private LiquidVertexFormatChain(
        DbcLiquidObjectTable liquidObjects,
        DbcLiquidMaterialTable liquidMaterials,
        IReadOnlyDictionary<int, int> materialIdByLiquidTypeId)
    {
        _liquidObjects = liquidObjects;
        _liquidMaterials = liquidMaterials;
        _materialIdByLiquidTypeId = materialIdByLiquidTypeId;
    }

    /// <summary>
    /// A chain with no tables. Values 0-3 still resolve directly; every LiquidObject id is reported
    /// unresolved. This is the no-DBC degradation path, and it reproduces today's behaviour exactly.
    /// </summary>
    public static LiquidVertexFormatChain Empty { get; } =
        new(DbcLiquidObjectTable.Empty, DbcLiquidMaterialTable.Empty, new Dictionary<int, int>());

    /// <summary>True when at least one link of the chain carries rows.</summary>
    public bool HasTables => _liquidObjects.RowCount > 0 || _liquidMaterials.RowCount > 0;

    public int LiquidObjectRowCount => _liquidObjects.RowCount;

    public int LiquidMaterialRowCount => _liquidMaterials.RowCount;

    public int LiquidTypeMaterialRowCount => _materialIdByLiquidTypeId.Count;

    /// <summary>Distinct raw values that failed to resolve, with how often each was seen.</summary>
    public IReadOnlyDictionary<ushort, int> UnresolvedCounts => _unresolvedCounts;

    /// <summary>Compose a chain from already-built tables. Used by tests and by the inspect tool.</summary>
    public static LiquidVertexFormatChain FromTables(
        DbcLiquidObjectTable liquidObjects,
        DbcLiquidMaterialTable liquidMaterials,
        IReadOnlyDictionary<int, int> materialIdByLiquidTypeId)
    {
        ArgumentNullException.ThrowIfNull(liquidObjects);
        ArgumentNullException.ThrowIfNull(liquidMaterials);
        ArgumentNullException.ThrowIfNull(materialIdByLiquidTypeId);

        return new LiquidVertexFormatChain(liquidObjects, liquidMaterials, materialIdByLiquidTypeId);
    }

    /// <summary>
    /// Load the whole chain for <paramref name="buildVersion"/>. Any link that fails to load leaves
    /// that link empty rather than throwing: a partial chain still resolves what it can and reports
    /// the rest.
    /// </summary>
    public static LiquidVertexFormatChain Load(
        IDBCProvider dbcProvider,
        string definitionsDirectory,
        string buildVersion,
        out IReadOnlyList<string> diagnostics)
    {
        ArgumentNullException.ThrowIfNull(dbcProvider);

        List<string> messages = [];
        DbcLiquidObjectTable objects = DbcLiquidObjectTable.Empty;
        DbcLiquidMaterialTable materials = DbcLiquidMaterialTable.Empty;
        Dictionary<int, int> materialByType = [];

        try
        {
            objects = DbcLiquidObjectTable.Load(dbcProvider, definitionsDirectory, buildVersion);
            messages.Add($"LiquidObject: {objects.RowCount} rows via column '{objects.LiquidTypeIdColumn ?? "<none found>"}'.");
        }
        catch (Exception ex)
        {
            messages.Add($"LiquidObject: FAILED to load for build {buildVersion}: {ex.Message}");
        }

        try
        {
            materials = DbcLiquidMaterialTable.Load(dbcProvider, definitionsDirectory, buildVersion);
            messages.Add($"LiquidMaterial: {materials.RowCount} rows via column '{materials.LvfColumn ?? "<none found>"}'.");
        }
        catch (Exception ex)
        {
            messages.Add($"LiquidMaterial: FAILED to load for build {buildVersion}: {ex.Message}");
        }

        try
        {
            materialByType = LoadMaterialIdByLiquidTypeId(dbcProvider, definitionsDirectory, buildVersion, out string? column);
            messages.Add($"LiquidType: {materialByType.Count} rows carry a material id via column '{column ?? "<none found>"}'.");
        }
        catch (Exception ex)
        {
            messages.Add($"LiquidType: FAILED to load for build {buildVersion}: {ex.Message}");
        }

        diagnostics = messages;
        return new LiquidVertexFormatChain(objects, materials, materialByType);
    }

    /// <summary>
    /// Read <c>LiquidType</c>'s MaterialID column. This is the middle link, and it is read through
    /// DBCD rather than the byte offset <see cref="DbcLiquidTypeTable"/> uses, because that offset is
    /// only correct for the 3.1.0-5.4.8 record layout.
    /// </summary>
    private static Dictionary<int, int> LoadMaterialIdByLiquidTypeId(
        IDBCProvider dbcProvider,
        string definitionsDirectory,
        string buildVersion,
        out string? materialColumn)
    {
        IDBCDStorage storage = DbcTableLoader.Load(dbcProvider, definitionsDirectory, buildVersion, "LiquidType");
        materialColumn = DbcTableLoader.DetectColumn(storage, ["MaterialID", "MaterialId", "Material"]);
        string? idColumn = DbcTableLoader.DetectColumn(storage, DbcTableLoader.IdColumns);

        Dictionary<int, int> map = [];
        if (materialColumn is null)
            return map;

        foreach (DBCDRow row in storage.Values)
        {
            int? materialId = DbcTableLoader.TryGetInt(row, materialColumn);
            if (materialId is null or <= 0)
                continue;

            map[DbcTableLoader.ResolveRowId(row, idColumn)] = materialId.Value;
        }

        return map;
    }

    /// <summary>
    /// Resolve one <c>liquid_object_or_lvf</c> value. Never throws, never guesses; an unresolved
    /// value is recorded in <see cref="UnresolvedCounts"/> and returned with
    /// <see cref="LiquidVertexFormatResolution.Resolved"/> false.
    /// </summary>
    public LiquidVertexFormatResolution Resolve(ushort rawValue)
    {
        if (rawValue <= MaxDirectVertexFormat)
        {
            return new LiquidVertexFormatResolution(
                rawValue,
                LiquidVertexFormatSource.VertexFormatField,
                (AdtLiquidVertexFormat)rawValue,
                Resolved: true,
                LiquidTypeId: 0,
                MaterialId: 0,
                FailureReason: null);
        }

        if (rawValue < LiquidObjectIdThreshold)
        {
            return Unresolved(
                rawValue,
                $"value {rawValue} is neither a vertex format (0-{MaxDirectVertexFormat}) nor a LiquidObject id (>= {LiquidObjectIdThreshold})");
        }

        if (!_liquidObjects.TryGetLiquidTypeId(rawValue, out int liquidTypeId))
        {
            return Unresolved(
                rawValue,
                $"LiquidObject id {rawValue} is not present in the loaded LiquidObject table ({_liquidObjects.RowCount} rows)");
        }

        if (!_materialIdByLiquidTypeId.TryGetValue(liquidTypeId, out int materialId))
        {
            return Unresolved(
                rawValue,
                $"LiquidObject {rawValue} -> LiquidType {liquidTypeId}, which carries no material id in the loaded LiquidType table ({_materialIdByLiquidTypeId.Count} rows)",
                liquidTypeId);
        }

        if (!_liquidMaterials.TryGetVertexFormat(materialId, out AdtLiquidVertexFormat format))
        {
            string detail = _liquidMaterials.TryGetRawLvf(materialId, out int rawLvf)
                ? $"LVF {rawLvf} is outside the defined range 0-{MaxDirectVertexFormat}"
                : $"material is not present in the loaded LiquidMaterial table ({_liquidMaterials.RowCount} rows)";
            return Unresolved(
                rawValue,
                $"LiquidObject {rawValue} -> LiquidType {liquidTypeId} -> LiquidMaterial {materialId}, but {detail}",
                liquidTypeId,
                materialId);
        }

        return new LiquidVertexFormatResolution(
            rawValue,
            LiquidVertexFormatSource.LiquidObjectChain,
            format,
            Resolved: true,
            liquidTypeId,
            materialId,
            FailureReason: null);
    }

    private LiquidVertexFormatResolution Unresolved(ushort rawValue, string reason, int liquidTypeId = 0, int materialId = 0)
    {
        _unresolvedCounts.AddOrUpdate(rawValue, 1, static (_, count) => count + 1);
        return new LiquidVertexFormatResolution(
            rawValue,
            LiquidVertexFormatSource.Unresolved,
            default,
            Resolved: false,
            liquidTypeId,
            materialId,
            reason);
    }
}
