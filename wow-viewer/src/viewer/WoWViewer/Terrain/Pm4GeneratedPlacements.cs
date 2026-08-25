using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text.Json;

namespace WoWViewer.Terrain;

/// <summary>
/// Optional side-car of asset names recovered from PM4 geometry, used to identify objects the scene
/// cannot identify on its own.
/// </summary>
/// <remarks>
/// The scene resolves a PM4 object by finding the loaded WMO instance that placed it. That works only
/// where the tile still HAS placement data - on a tile whose ADT is gone the object stays an anonymous
/// key, which is precisely the case worth labelling. This fills that gap from
/// <c>pm4-generated-placements.json</c>, produced by <c>pm4 generate-placements</c>.
///
/// <para>Names from here are <b>guesses</b>, right at rank one about half the time, so every entry
/// carries its score and callers are expected to mark them as inferred rather than present them the way
/// a real placement is presented.</para>
///
/// <para>Lookup is by placement key plus position rather than by tile name. The file records tiles as
/// <c>&lt;first&gt;_&lt;second&gt;</c> while the scene keys them as X and Y, and that pairing is
/// reversed relative to the obvious reading - matching on it would be a silent one-tile-off bug. A key
/// collision across tiles is resolved by taking the nearest candidate and rejecting anything further
/// than a few units, so a wrong match fails closed instead of mislabelling.</para>
/// </remarks>
internal static class Pm4GeneratedPlacements
{
    private const float MaxCentreDistance = 8f;

    private static readonly Dictionary<uint, List<Entry>> ByKey = [];
    private static bool _attempted;

    private readonly record struct Entry(string Asset, double Score, Vector3 Centre);

    public static int Count { get; private set; }

    public static string? SourcePath { get; private set; }

    /// <summary>
    /// Loads the side-car if it is present. Absent or malformed input is not an error - the viewer runs
    /// exactly as before without it.
    /// </summary>
    public static void EnsureLoaded(string? explicitPath = null)
    {
        if (_attempted && explicitPath is null)
            return;

        _attempted = true;
        ByKey.Clear();
        Count = 0;
        SourcePath = null;

        string path = explicitPath ?? Path.Combine("output", "pm4-placements", "pm4-generated-placements.json");
        if (!File.Exists(path))
            return;

        try
        {
            using FileStream stream = File.OpenRead(path);
            using JsonDocument doc = JsonDocument.Parse(stream);
            if (!TryProp(doc.RootElement, "tiles", out JsonElement tiles))
                return;

            foreach (JsonElement tile in tiles.EnumerateArray())
            {
                if (!TryProp(tile, "placements", out JsonElement rows))
                    continue;

                foreach (JsonElement row in rows.EnumerateArray())
                {
                    string? asset = TryProp(row, "asset", out JsonElement a) ? a.GetString() : null;
                    if (string.IsNullOrWhiteSpace(asset) || asset.Equals("missingwmo.wmo", StringComparison.OrdinalIgnoreCase))
                        continue;

                    if (!TryProp(row, "objectKey", out JsonElement k)
                        || k.GetString() is not string keyText
                        || !TryParseKey(keyText, out uint packed))
                    {
                        continue;
                    }

                    if (!TryProp(row, "boundsMin", out JsonElement lo)
                        || !TryProp(row, "boundsMax", out JsonElement hi))
                    {
                        continue;
                    }

                    Vector3 centre = (ReadXyz(lo) + ReadXyz(hi)) * 0.5f;
                    double score = TryProp(row, "score", out JsonElement s) ? s.GetDouble() : double.NaN;

                    // The scene's ck24 is the placement float shifted down a byte.
                    uint ck24 = packed >> 8;
                    if (!ByKey.TryGetValue(ck24, out List<Entry>? list))
                    {
                        list = [];
                        ByKey[ck24] = list;
                    }

                    list.Add(new Entry(asset, score, centre));
                    Count++;
                }
            }

            SourcePath = Path.GetFullPath(path);
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            ByKey.Clear();
            Count = 0;
            SourcePath = null;
        }
    }

    /// <summary>
    /// Finds a recovered name for an object the scene could not identify, or false when nothing matches
    /// closely enough to be worth showing.
    /// </summary>
    public static bool TryResolve(uint ck24, Vector3 boundsMin, Vector3 boundsMax, out string? asset, out double score)
    {
        asset = null;
        score = double.NaN;

        if (ck24 == 0 || ByKey.Count == 0 || !ByKey.TryGetValue(ck24, out List<Entry>? entries))
            return false;

        Vector3 centre = (boundsMin + boundsMax) * 0.5f;
        float best = float.MaxValue;
        foreach (Entry entry in entries)
        {
            float d = Vector3.Distance(centre, entry.Centre);
            if (d < best)
            {
                best = d;
                asset = entry.Asset;
                score = entry.Score;
            }
        }

        if (best <= MaxCentreDistance)
            return true;

        asset = null;
        score = double.NaN;
        return false;
    }

    /// <summary>
    /// Case-insensitive property lookup.
    /// </summary>
    /// <remarks>
    /// <c>TryGetProperty</c> is case-sensitive, and the generator wrote PascalCase while this read
    /// camelCase - which loads zero entries and reports nothing wrong. The generator now emits
    /// camelCase, and this stays tolerant so a file produced before that change still works.
    /// </remarks>
    private static bool TryProp(JsonElement source, string name, out JsonElement value)
    {
        if (source.TryGetProperty(name, out value))
            return true;

        foreach (JsonProperty property in source.EnumerateObject())
        {
            if (string.Equals(property.Name, name, StringComparison.OrdinalIgnoreCase))
            {
                value = property.Value;
                return true;
            }
        }

        value = default;
        return false;
    }

    private static bool TryParseKey(string text, out uint value) =>
        text.StartsWith("0x", StringComparison.OrdinalIgnoreCase)
            ? uint.TryParse(text.AsSpan(2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out value)
            : uint.TryParse(text, out value);

    private static Vector3 ReadXyz(JsonElement e) => new(
        TryProp(e, "x", out JsonElement x) ? x.GetSingle() : 0f,
        TryProp(e, "y", out JsonElement y) ? y.GetSingle() : 0f,
        TryProp(e, "z", out JsonElement z) ? z.GetSingle() : 0f);
}
