using System;
using System.Collections.Generic;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// Portal references chunk. Links portals to groups.
    /// Count is defined by sum of MOGP.portals_used.
    /// </summary>
    public class MOPR : IChunk
    {
        public const int PORTAL_REF_SIZE = 8;

        public class PortalReference
        {
            public ushort PortalIndex { get; set; }  // Index into MOPT
            public ushort GroupIndex { get; set; }   // The other group
            public short Side { get; set; }          // Positive or negative
            public ushort Filler { get; set; }       // Unused

            /// <summary>
            /// Validates that the portal reference has valid indices.
            /// </summary>
            public bool ValidateIndices(int portalCount, int groupCount)
            {
                return PortalIndex < portalCount && GroupIndex < groupCount;
            }

            /// <summary>
            /// Gets the direction of the portal relative to the group.
            /// </summary>
            public bool IsPositiveSide => Side > 0;
        }

        /// <summary>
        /// Gets the list of portal references.
        /// </summary>
        public List<PortalReference> References { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Clear existing data
            References.Clear();

            // Read portal references
            var refCount = (int)size / PORTAL_REF_SIZE;
            for (int i = 0; i < refCount; i++)
            {
                var reference = new PortalReference
                {
                    PortalIndex = reader.ReadUInt16(),
                    GroupIndex = reader.ReadUInt16(),
                    Side = reader.ReadInt16(),
                    Filler = reader.ReadUInt16()
                };

                References.Add(reference);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var reference in References)
            {
                writer.Write(reference.PortalIndex);
                writer.Write(reference.GroupIndex);
                writer.Write(reference.Side);
                writer.Write(reference.Filler);
            }
        }

        /// <summary>
        /// Gets all portal references for a specific group.
        /// </summary>
        public List<PortalReference> GetReferencesForGroup(int groupIndex)
        {
            return References.FindAll(r => r.GroupIndex == groupIndex);
        }

        /// <summary>
        /// Gets all portal references for a specific portal.
        /// </summary>
        public List<PortalReference> GetReferencesForPortal(int portalIndex)
        {
            return References.FindAll(r => r.PortalIndex == portalIndex);
        }

        /// <summary>
        /// Validates all portal references against portal and group counts.
        /// </summary>
        public bool ValidateReferences(MOPT mopt, MOGI mogi)
        {
            if (mopt == null || mogi == null)
                return false;

            foreach (var reference in References)
            {
                if (!reference.ValidateIndices(mopt.Portals.Count, mogi.Groups.Length))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Validates portal topology (each portal should connect exactly two groups).
        /// </summary>
        public bool ValidateTopology(MOPT mopt)
        {
            if (mopt == null)
                return false;

            // Check each portal
            for (int i = 0; i < mopt.Portals.Count; i++)
            {
                var refs = GetReferencesForPortal(i);

                // Each portal should have exactly two references
                if (refs.Count != 2)
                    return false;

                // References should have opposite sides
                if (refs[0].Side * refs[1].Side >= 0)
                    return false;

                // References should be to different groups
                if (refs[0].GroupIndex == refs[1].GroupIndex)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets a validation report for portal references.
        /// </summary>
        public string GetValidationReport(MOPT mopt, MOGI mogi, MOGN mogn)
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOPR Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Reference Count: {References.Count}");
            report.AppendLine();

            if (mopt != null && mogi != null)
            {
                // Group portal counts
                var groupPortals = new Dictionary<int, int>();
                foreach (var reference in References)
                {
                    if (!groupPortals.ContainsKey(reference.GroupIndex))
                        groupPortals[reference.GroupIndex] = 0;
                    groupPortals[reference.GroupIndex]++;
                }

                report.AppendLine("Group Portal Counts:");
                foreach (var kvp in groupPortals)
                {
                    string groupName = "Unknown";
                    if (mogn != null && kvp.Key < mogi.Groups.Length)
                    {
                        var nameOffset = mogi.Groups[kvp.Key].NameOffset;
                        if (nameOffset >= 0)
                            groupName = mogn.GetNameByOffset(nameOffset) ?? "Invalid Name";
                    }
                    report.AppendLine($"  Group {kvp.Key} ({groupName}): {kvp.Value} portals");
                }
                report.AppendLine();

                // Portal validation
                report.AppendLine("Portal References:");
                for (int i = 0; i < mopt.Portals.Count; i++)
                {
                    var refs = GetReferencesForPortal(i);
                    report.AppendLine($"  Portal {i}:");
                    report.AppendLine($"    Reference Count: {refs.Count}");

                    foreach (var reference in refs)
                    {
                        string groupName = "Unknown";
                        if (mogn != null && reference.GroupIndex < mogi.Groups.Length)
                        {
                            var nameOffset = mogi.Groups[reference.GroupIndex].NameOffset;
                            if (nameOffset >= 0)
                                groupName = mogn.GetNameByOffset(nameOffset) ?? "Invalid Name";
                        }

                        report.AppendLine($"    -> Group {reference.GroupIndex} ({groupName})");
                        report.AppendLine($"       Side: {(reference.IsPositiveSide ? "Positive" : "Negative")}");
                    }

                    // Validate topology for this portal
                    bool validTopology = refs.Count == 2 &&
                                       refs[0].Side * refs[1].Side < 0 &&
                                       refs[0].GroupIndex != refs[1].GroupIndex;

                    if (!validTopology)
                    {
                        report.AppendLine("    WARNING: Invalid portal topology");
                        if (refs.Count != 2)
                            report.AppendLine("      - Portal does not connect exactly two groups");
                        else
                        {
                            if (refs[0].Side * refs[1].Side >= 0)
                                report.AppendLine("      - Portal sides are not opposite");
                            if (refs[0].GroupIndex == refs[1].GroupIndex)
                                report.AppendLine("      - Portal connects same group to itself");
                        }
                    }

                    report.AppendLine();
                }
            }

            return report.ToString();
        }

        /// <summary>
        /// Builds an adjacency graph of groups connected by portals.
        /// </summary>
        public Dictionary<int, HashSet<int>> BuildGroupGraph()
        {
            var graph = new Dictionary<int, HashSet<int>>();

            foreach (var reference in References)
            {
                // Ensure both groups exist in the graph
                if (!graph.ContainsKey(reference.GroupIndex))
                    graph[reference.GroupIndex] = new HashSet<int>();

                // Get the other group this portal connects to
                var otherRefs = GetReferencesForPortal(reference.PortalIndex);
                foreach (var otherRef in otherRefs)
                {
                    if (otherRef != reference)
                    {
                        graph[reference.GroupIndex].Add(otherRef.GroupIndex);
                    }
                }
            }

            return graph;
        }
    }
} 