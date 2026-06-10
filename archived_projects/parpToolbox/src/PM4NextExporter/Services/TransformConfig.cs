using System;
using System.Collections.Generic;

namespace PM4NextExporter.Services
{
    internal enum ExporterKind
    {
        Object,
        PerTileObj,
        PerTileObjects,
        Mscn,
        ObjectMscnSidecar
    }

    internal static class TransformConfig
    {
        // Decides whether an exporter should mirror X at all
        public static bool ShouldMirrorX(ExporterKind kind, bool legacyParity, bool alignWithMscn)
        {
            // When aligning with MSCN, meshes keep mirroring to fix orientation,
            // but MSCN point outputs should not mirror.
            if (alignWithMscn)
            {
                return kind switch
                {
                    ExporterKind.Mscn => false,
                    ExporterKind.ObjectMscnSidecar => false,
                    ExporterKind.PerTileObjects => true, // still mirror, but do it centered
                    ExporterKind.Object => true,         // force mirror; exporter will center it
                    ExporterKind.PerTileObj => true,     // force mirror; exporter will center it
                    _ => !legacyParity,
                };
            }

            // Default (current behavior preservation)
            return kind switch
            {
                ExporterKind.Object => !legacyParity,
                ExporterKind.PerTileObj => !legacyParity,
                ExporterKind.PerTileObjects => true, // matches existing const true
                ExporterKind.Mscn => legacyParity,   // MSCN mirrored only for legacy parity
                ExporterKind.ObjectMscnSidecar => !legacyParity,
                _ => !legacyParity,
            };
        }

        // Apply centered mirror if mirror==true
        public static float MirrorX(float x, float centerX, bool mirror)
            => mirror ? (2f * centerX - x) : x;
    }
}
