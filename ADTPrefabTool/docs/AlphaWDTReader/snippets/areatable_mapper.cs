// docs/AlphaWDTReader/snippets/areatable_mapper.cs
// Purpose: Remap Alpha (0.5.3) AreaIDs to 3.3.5 by name using normalization and simple disambiguation.
// This snippet is DBCD-agnostic: feed it rows loaded by your preferred DBC reader (e.g., DBCD).

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Snippets
{
    public sealed class AreaRow
    {
        public int Id;
        public string Name = string.Empty;
        public int MapId;
        public int ParentId;
        public string ParentName = string.Empty; // optional convenience
    }

    public sealed class MapContext
    {
        public int MapId;      // continent id for tile if known
        public int TileX;      // optional
        public int TileY;      // optional
    }

    public sealed class RemapResult
    {
        public int AlphaId;
        public string AlphaName = string.Empty;
        public int LkId;
        public string LkName = string.Empty;
        public string Status = "mapped"; // mapped | ambiguous | unmapped
        public string Note = string.Empty;
    }

    public sealed class AreaTableMapper
    {
        private readonly Dictionary<int, AreaRow> _alphaById;
        private readonly Dictionary<string, List<AreaRow>> _lkByNameNorm;

        private AreaTableMapper(Dictionary<int, AreaRow> alphaById, Dictionary<string, List<AreaRow>> lkByNameNorm)
        {
            _alphaById = alphaById;
            _lkByNameNorm = lkByNameNorm;
        }

        public static AreaTableMapper Create(IEnumerable<AreaRow> alphaRows, IEnumerable<AreaRow> lkRows)
        {
            var alphaById = alphaRows.ToDictionary(r => r.Id);
            var lkByNameNorm = new Dictionary<string, List<AreaRow>>();
            foreach (var r in lkRows)
            {
                string key = Normalize(r.Name);
                if (!lkByNameNorm.TryGetValue(key, out var list))
                    lkByNameNorm[key] = list = new List<AreaRow>();
                list.Add(r);
            }
            return new AreaTableMapper(alphaById, lkByNameNorm);
        }

        public RemapResult RemapAreaId(int alphaId, MapContext ctx)
        {
            if (!_alphaById.TryGetValue(alphaId, out var a))
            {
                return new RemapResult { AlphaId = alphaId, LkId = alphaId, Status = "unmapped", Note = "alpha id not found" };
            }

            string key = Normalize(a.Name);
            if (!_lkByNameNorm.TryGetValue(key, out var candidates) || candidates.Count == 0)
            {
                return new RemapResult { AlphaId = alphaId, AlphaName = a.Name, LkId = alphaId, LkName = a.Name, Status = "unmapped", Note = "no name match" };
            }

            // Disambiguation
            AreaRow chosen = candidates[0];
            string note = string.Empty;

            // 1) Prefer same MapId
            if (ctx != null)
            {
                var sameMap = candidates.Where(c => c.MapId == ctx.MapId).ToList();
                if (sameMap.Count == 1) chosen = sameMap[0];
                else if (sameMap.Count > 1)
                {
                    candidates = sameMap; note = "multiple candidates same map";
                }
            }

            // 2) Prefer same parent name if available
            if (!string.IsNullOrEmpty(a.ParentName))
            {
                string pKey = Normalize(a.ParentName);
                var sameParent = candidates.Where(c => Normalize(c.ParentName) == pKey).ToList();
                if (sameParent.Count == 1) chosen = sameParent[0];
                else if (sameParent.Count > 1)
                {
                    candidates = sameParent; note = string.IsNullOrEmpty(note) ? "multiple candidates same parent" : note + "; same parent";
                }
            }

            string status = candidates.Count > 1 ? "ambiguous" : "mapped";
            return new RemapResult
            {
                AlphaId = a.Id,
                AlphaName = a.Name,
                LkId = chosen.Id,
                LkName = chosen.Name,
                Status = status,
                Note = note
            };
        }

        public static string Normalize(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) return string.Empty;
            string s = name.Trim().ToLowerInvariant();
            // collapse whitespace
            s = Regex.Replace(s, "\\s+", " ");
            // strip simple color codes like |cAARRGGBB and |r
            s = Regex.Replace(s, "\\|c[0-9a-fA-F]{8}", string.Empty);
            s = s.Replace("|r", string.Empty);
            return s;
        }
    }
}
