using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using WoWToolbox.Core.Navigation.PM4;

namespace WoWToolbox.MPRRExplorer
{
    class Program
    {
        static readonly HashSet<int> ExpectedMprrFlags = new HashSet<int> {
            0x0300,0x0301,0x0302,0x0303,0x0304,0x0305,0x0306,0x0307,0x0308,0x0309,0x030a,0x030b,0x030c,0x030d,0x030e,0x0310,0x0311,0x0314,0x0500,0x0501,0x0502,0x1100
        };

        static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: WoWToolbox.MPRRExplorer <pm4 file path>");
                return;
            }

            string pm4Path = args[0];
            if (!File.Exists(pm4Path))
            {
                Console.WriteLine($"File not found: {pm4Path}");
                return;
            }

            PM4File pm4;
            try
            {
                pm4 = PM4File.FromFile(pm4Path);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load PM4 file: {ex.Message}");
                return;
            }

            // --- Load MSVI and MSVT for geometry cross-referencing ---
            var msvi = pm4.MSVI?.Indices ?? new List<uint>();
            var msvt = pm4.MSVT?.Vertices ?? new List<WoWToolbox.Core.Navigation.PM4.Chunks.MsvtVertex>();

            if (pm4.MPRR == null)
            {
                Console.WriteLine("No MPRR chunk found in file.");
                return;
            }

            var sequences = pm4.MPRR.Sequences;
            Console.WriteLine($"MPRR Chunk: {sequences.Count} sequences found.\n");

            // Output directory: always use project root output folder
            string projectRoot = "I:\\parp-scripts\\WoWToolbox_v3";
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string outputDir = Path.Combine(projectRoot, "output", $"mprr_{timestamp}");
            Directory.CreateDirectory(outputDir);
            string csvPath = Path.Combine(outputDir, "mprr_sequences.csv");
            string crossrefCsvPath = Path.Combine(outputDir, "mprr_crossref_summary.csv");
            string logPath = Path.Combine(outputDir, "mprr_analysis.log");
            string linkageCsvPath = Path.Combine(outputDir, "mprr_mslk_linkage.csv");
            string graphPath = Path.Combine(outputDir, "mprr_mslk_graph.dot");
            string seqLenCsvPath = Path.Combine(outputDir, "mprr_sequence_lengths.csv");
            string groupConsCsvPath = Path.Combine(outputDir, "mprr_group_consistency.csv");
            string pairPropsCsvPath = Path.Combine(outputDir, "mprr_pair_properties.csv");
            string nodeFreqCsvPath = Path.Combine(outputDir, "mprr_node_frequency.csv");
            string topNodesCsvPath = Path.Combine(outputDir, "mprr_top_nodes_crossref.csv");
            string topNodesMermaidPath = Path.Combine(outputDir, "mprr_top_nodes_mermaid.md");
            string topNodesSummaryPath = Path.Combine(outputDir, "mprr_top_nodes_summary.txt");

            // --- Find newest batch output directory ---
            string batchRoot = Path.Combine(projectRoot, "output");
            string newestBatchDir = null;
            DateTime newestTime = DateTime.MinValue;
            foreach (var dir in Directory.GetDirectories(batchRoot))
            {
                string batchOutput = Path.Combine(dir, "PM4_BatchOutput");
                if (Directory.Exists(batchOutput))
                {
                    var dirInfo = new DirectoryInfo(dir);
                    if (dirInfo.LastWriteTime > newestTime)
                    {
                        newestTime = dirInfo.LastWriteTime;
                        newestBatchDir = batchOutput;
                    }
                }
            }
            if (newestBatchDir == null)
            {
                Console.WriteLine($"No PM4_BatchOutput directory found in {batchRoot}");
                return;
            }

            // Find the OBJ and CSV files in the newest batch output directory
            string pm4Base = Path.GetFileNameWithoutExtension(pm4Path);
            string mslkObjPath = Path.Combine(newestBatchDir, pm4Base + "_pm4_mslk_nodes.obj");
            string mprlCsvPath = Path.Combine(newestBatchDir, pm4Base + "_mprl_data.csv");
            if (!File.Exists(mslkObjPath))
            {
                Console.WriteLine($"MSLK OBJ file not found: {mslkObjPath}");
                return;
            }
            if (!File.Exists(mprlCsvPath))
            {
                Console.WriteLine($"MPRL CSV file not found: {mprlCsvPath}");
                return;
            }

            using var logWriter = new StreamWriter(logPath);
            logWriter.WriteLine($"MPRR/MPRL/MSLK Analysis Log - {DateTime.Now}");

            // Parse node indices and Unk10 anchor indices from OBJ comments
            var nodeIndices = new HashSet<int>();
            var unk10Anchors = new HashSet<int>();
            var nodeIdxToUnk10 = new Dictionary<int, int>();
            var nodeIdxRegex = new Regex(@"Node Idx=(\d+).+Unk10=(\d+)");
            var mslk00 = new Dictionary<int, int>(); // nodeIdx -> 0x00
            var mslk01 = new Dictionary<int, int>(); // nodeIdx -> 0x01
            var mslk02 = new Dictionary<int, int>(); // nodeIdx -> 0x02
            var mslk0c = new Dictionary<int, int>(); // nodeIdx -> 0x0C
            var mslk0e = new Dictionary<int, int>(); // nodeIdx -> 0x0E
            var mslk12 = new Dictionary<int, int>(); // nodeIdx -> 0x12
            var mslkLineRegex = new Regex(@"Node Idx=(\d+) Grp=0x([0-9a-fA-F]+) Unk00=0x([0-9a-fA-F]+) Unk01=0x([0-9a-fA-F]+) Unk10=(\d+) Unk12=0x([0-9a-fA-F]+)");
            foreach (var line in File.ReadLines(mslkObjPath))
            {
                var match = mslkLineRegex.Match(line);
                if (match.Success)
                {
                    int nodeIdx = int.Parse(match.Groups[1].Value);
                    int grp = Convert.ToInt32(match.Groups[2].Value, 16);
                    int unk00 = Convert.ToInt32(match.Groups[3].Value, 16);
                    int unk01 = Convert.ToInt32(match.Groups[4].Value, 16);
                    int unk10 = int.Parse(match.Groups[5].Value);
                    int unk12 = Convert.ToInt32(match.Groups[6].Value, 16);
                    nodeIndices.Add(nodeIdx);
                    unk10Anchors.Add(unk10);
                    nodeIdxToUnk10[nodeIdx] = unk10;
                    mslk00[nodeIdx] = unk00;
                    mslk01[nodeIdx] = unk01;
                    mslk02[nodeIdx] = 0; // Not present in OBJ, assume 0
                    mslk0c[nodeIdx] = grp;
                    mslk0e[nodeIdx] = 0xFFFF; // Not present in OBJ, assume 0xFFFF
                    mslk12[nodeIdx] = unk12;
                }
            }

            // Parse MPRL CSV: expect columns like Index,Unk00,Unk02,Unk04,Unk06,...
            var mprl00 = new Dictionary<int, int>(); // index -> Unk00
            var mprl02 = new Dictionary<int, int>(); // index -> Unk02
            var mprl04 = new Dictionary<int, int>(); // index -> Unk04
            var mprl06 = new Dictionary<int, int>(); // index -> Unk06
            var mprlHeader = File.ReadLines(mprlCsvPath).FirstOrDefault()?.Split(',');
            if (mprlHeader == null || mprlHeader.Length < 7)
            {
                logWriter.WriteLine($"MPRL CSV header is missing or malformed: {mprlCsvPath}");
            }
            else
            {
                foreach (var line in File.ReadLines(mprlCsvPath).Skip(1))
                {
                    var parts = line.Split(',');
                    if (parts.Length < 7) continue;
                    int idx = int.Parse(parts[0]);
                    int unk00 = int.Parse(parts[1]);
                    int unk02 = int.Parse(parts[2]);
                    int unk04 = int.Parse(parts[3]);
                    int unk06 = int.Parse(parts[4]);
                    mprl00[idx] = unk00;
                    mprl02[idx] = unk02;
                    mprl04[idx] = unk04;
                    mprl06[idx] = unk06;
                }
            }

            // --- Node frequency analysis (for top 50 nodes) and role summaries ---
            var nodeFreq = new Dictionary<int, int>();
            var nodeToSeqs = new Dictionary<int, HashSet<int>>();
            var nodeToConnections = new Dictionary<int, HashSet<int>>();
            var nodeToGroups = new Dictionary<int, HashSet<int>>();
            var nodeToStart = new Dictionary<int, int>();
            var nodeToEnd = new Dictionary<int, int>();
            var nodeToMiddle = new Dictionary<int, int>();
            // --- Sentinel leaf detection ---
            var sentinelLeafNodes = new HashSet<int>();
            foreach (var seq in sequences)
            {
                var nodeIdxs = seq.Where(val => nodeIndices.Contains(val)).ToList();
                if (nodeIdxs.Count > 0)
                {
                    // The last node before the 0xFFFF terminator is a sentinel leaf
                    sentinelLeafNodes.Add(nodeIdxs.Last());
                }
            }
            foreach (var (seq, seqIdx) in sequences.Select((s, i) => (s, i)))
            {
                var nodeIdxs = seq.Where(val => nodeIndices.Contains(val)).ToList();
                for (int i = 0; i < nodeIdxs.Count; i++)
                {
                    int idx = nodeIdxs[i];
                    if (!nodeFreq.ContainsKey(idx)) nodeFreq[idx] = 0;
                    nodeFreq[idx]++;
                    if (!nodeToSeqs.ContainsKey(idx)) nodeToSeqs[idx] = new HashSet<int>();
                    nodeToSeqs[idx].Add(seqIdx);
                    if (!nodeToConnections.ContainsKey(idx)) nodeToConnections[idx] = new HashSet<int>();
                    foreach (var other in nodeIdxs)
                        if (other != idx) nodeToConnections[idx].Add(other);
                    if (!nodeToGroups.ContainsKey(idx)) nodeToGroups[idx] = new HashSet<int>();
                    foreach (var other in nodeIdxs)
                        if (mslk0c.ContainsKey(other)) nodeToGroups[idx].Add(mslk0c[other]);
                    // Start/middle/end roles
                    if (i == 0)
                    {
                        if (!nodeToStart.ContainsKey(idx)) nodeToStart[idx] = 0;
                        nodeToStart[idx]++;
                    }
                    else if (i == nodeIdxs.Count - 1)
                    {
                        if (!nodeToEnd.ContainsKey(idx)) nodeToEnd[idx] = 0;
                        nodeToEnd[idx]++;
                    }
                    else
                    {
                        if (!nodeToMiddle.ContainsKey(idx)) nodeToMiddle[idx] = 0;
                        nodeToMiddle[idx]++;
                    }
                }
            }
            var topNodes = nodeFreq.OrderByDescending(kv => kv.Value).Take(50).Select(kv => kv.Key).ToList();
            // --- Role-based output files ---
            var roleToNodes = new Dictionary<string, List<int>>
            {
                { "Hub", new List<int>() },
                { "Leaf", new List<int>() },
                { "Root", new List<int>() }
            };
            var nodeSummaries = new Dictionary<int, List<string>>();
            // --- Node type mapping by (unk00, unk01) ---
            var typeToNodes = new Dictionary<(int, int), List<int>>();
            foreach (var idx in topNodes)
            {
                int unk00 = mslk00.ContainsKey(idx) ? mslk00[idx] : -1;
                int unk01 = mslk01.ContainsKey(idx) ? mslk01[idx] : -1;
                var typeKey = (unk00, unk01);
                if (!typeToNodes.ContainsKey(typeKey)) typeToNodes[typeKey] = new List<int>();
                typeToNodes[typeKey].Add(idx);
            }
            // --- Type breakdown for summary ---
            using (var topNodesSummaryWriter = new StreamWriter(topNodesSummaryPath))
            {
                topNodesSummaryWriter.WriteLine("Node Type Breakdown (by unk00, unk01):");
                foreach (var type in typeToNodes.Keys.OrderBy(t => t.Item1).ThenBy(t => t.Item2))
                {
                    topNodesSummaryWriter.WriteLine($"  Type unk00={type.Item1}, unk01={type.Item2}: {typeToNodes[type].Count} nodes");
                }
                topNodesSummaryWriter.WriteLine("");
                // Output nodes grouped by type, then by group
                foreach (var type in typeToNodes.Keys.OrderBy(t => t.Item1).ThenBy(t => t.Item2))
                {
                    var nodesInType = typeToNodes[type].OrderBy(idx => mslk0c.ContainsKey(idx) ? mslk0c[idx] : int.MaxValue).ToList();
                    topNodesSummaryWriter.WriteLine($"=== Type unk00={type.Item1}, unk01={type.Item2} ===");
                    foreach (var idx in nodesInType)
                    {
                        string group = mslk0c.ContainsKey(idx) ? mslk0c[idx].ToString() : "";
                        string unk12 = mslk12.ContainsKey(idx) ? mslk12[idx].ToString() : "";
                        int degree = nodeToConnections.ContainsKey(idx) ? nodeToConnections[idx].Count : 0;
                        string groupsConnected = nodeToGroups.ContainsKey(idx) ? string.Join(", ", nodeToGroups[idx].OrderBy(x => x)) : "";
                        var neighborCounts = new Dictionary<int, int>();
                        if (nodeToConnections.ContainsKey(idx))
                        {
                            foreach (var n in nodeToConnections[idx])
                            {
                                if (!neighborCounts.ContainsKey(n)) neighborCounts[n] = 0;
                                neighborCounts[n] += nodeToSeqs.ContainsKey(n) ? nodeToSeqs[n].Count : 1;
                            }
                        }
                        var topNeighbors = neighborCounts.OrderByDescending(kv => kv.Value).Take(5).Select(kv => $"{kv.Key}({kv.Value})");
                        int appearsAsStart = nodeToStart.ContainsKey(idx) ? nodeToStart[idx] : 0;
                        int appearsAsEnd = nodeToEnd.ContainsKey(idx) ? nodeToEnd[idx] : 0;
                        int appearsAsMiddle = nodeToMiddle.ContainsKey(idx) ? nodeToMiddle[idx] : 0;
                        var roles = new List<string>();
                        bool isRoot = appearsAsStart > 10;
                        bool isSentinelLeaf = sentinelLeafNodes.Contains(idx);
                        if (isRoot) roles.Add("Root");
                        else if (isSentinelLeaf) roles.Add("Leaf");
                        if (degree >= 5) roles.Add("Hub");
                        if (isRoot) roleToNodes["Root"].Add(idx);
                        else if (isSentinelLeaf) roleToNodes["Leaf"].Add(idx);
                        if (degree >= 5) roleToNodes["Hub"].Add(idx);
                        var summaryLines = new List<string>
                        {
                            $"Node {idx} (Group {group}):",
                            $"  Frequency: {nodeFreq[idx]}",
                            $"  Degree: {degree}",
                            $"  Groups connected: {groupsConnected}",
                            $"  Most common neighbors: {string.Join(", ", topNeighbors)}",
                            $"  Appears as start: {appearsAsStart}, end: {appearsAsEnd}, middle: {appearsAsMiddle}",
                            $"  Role: {string.Join(", ", roles)}",
                            $"  Unk00: {type.Item1}, Unk01: {type.Item2}, Unk12: {unk12}",
                            ""
                        };
                        nodeSummaries[idx] = summaryLines;
                        foreach (var line in summaryLines) topNodesSummaryWriter.WriteLine(line);
                    }
                }
            }
            // Write role-based files, grouped by type then group
            foreach (var role in roleToNodes.Keys)
            {
                string rolePath = Path.Combine(outputDir, $"mprr_top_nodes_{role.ToLower()}.txt");
                using var writer = new StreamWriter(rolePath);
                var nodes = roleToNodes[role].Distinct().ToList();
                var nodesByType = nodes.GroupBy(idx => (mslk00.ContainsKey(idx) ? mslk00[idx] : -1, mslk01.ContainsKey(idx) ? mslk01[idx] : -1))
                    .OrderBy(g => g.Key.Item1).ThenBy(g => g.Key.Item2);
                foreach (var typeGroup in nodesByType)
                {
                    writer.WriteLine($"=== Type unk00={typeGroup.Key.Item1}, unk01={typeGroup.Key.Item2} ===");
                    foreach (var idx in typeGroup.OrderBy(idx => mslk0c.ContainsKey(idx) ? mslk0c[idx] : int.MaxValue))
                        foreach (var line in nodeSummaries[idx])
                            writer.WriteLine(line);
                }
            }
            // --- Type-centric hierarchy output ---
            string typeHierarchyPath = Path.Combine(outputDir, "mprr_type_hierarchy.txt");
            using (var writer = new StreamWriter(typeHierarchyPath))
            {
                writer.WriteLine("Type-Centric Hierarchy of Top Nodes (by unk00, unk01, then object id):\n");
                // Group nodes by (unk00, unk01)
                var nodesByType = topNodes.GroupBy(idx => (mslk00.ContainsKey(idx) ? mslk00[idx] : -1, mslk01.ContainsKey(idx) ? mslk01[idx] : -1))
                    .OrderBy(g => g.Key.Item1).ThenBy(g => g.Key.Item2);
                foreach (var typeGroup in nodesByType)
                {
                    var typeKey = typeGroup.Key;
                    var nodes = typeGroup.ToList();
                    writer.WriteLine($"=== Type unk00={typeKey.Item1}, unk01={typeKey.Item2} ({nodes.Count} nodes) ===");
                    // Group by object id (mslk0c)
                    var nodesByObj = nodes.GroupBy(idx => mslk0c.ContainsKey(idx) ? mslk0c[idx] : -1)
                        .OrderBy(g => g.Key);
                    foreach (var objGroup in nodesByObj)
                    {
                        int objId = objGroup.Key;
                        writer.WriteLine($"  ObjectId {objId}:");
                        foreach (var idx in objGroup)
                        {
                            writer.WriteLine($"    Node {idx}");
                        }
                    }
                    writer.WriteLine("");
                }
            }
            Console.WriteLine($"Type-centric hierarchy written to: {typeHierarchyPath} (grouped by unk00, unk01, then object id, no roles)");

            // Print summary to console
            int minLenFinal = sequences.Count > 0 ? sequences.Min(s => s.Count) : 0;
            int maxLenFinal = sequences.Count > 0 ? sequences.Max(s => s.Count) : 0;
            double avgLen = sequences.Count > 0 ? (double)sequences.Sum(s => s.Count) / sequences.Count : 0;
            Console.WriteLine("\n--- Summary ---");
            Console.WriteLine($"Total sequences: {sequences.Count}");
            Console.WriteLine($"Average sequence length: {avgLen:F2}");
            Console.WriteLine($"Min sequence length: {minLenFinal}");
            Console.WriteLine($"Max sequence length: {maxLenFinal}");
            Console.WriteLine($"\nCSV output written to: {csvPath}");
            Console.WriteLine($"Cross-reference summary written to: {crossrefCsvPath}");
            Console.WriteLine($"Linkage CSV written to: {linkageCsvPath}");
            Console.WriteLine($"Graphviz .dot file written to: {graphPath}");
            Console.WriteLine($"Sequence length/type CSV written to: {seqLenCsvPath}");
            Console.WriteLine($"Group consistency CSV written to: {groupConsCsvPath}");
            Console.WriteLine($"Pair property CSV written to: {pairPropsCsvPath}");
            Console.WriteLine($"Node frequency CSV written to: {nodeFreqCsvPath}");
            Console.WriteLine($"Top 50 node crossref CSV written to: {topNodesCsvPath}");
            Console.WriteLine($"Top 10 node Mermaid graph written to: {topNodesMermaidPath}");
            Console.WriteLine($"Top 50 node summary written to: {topNodesSummaryPath}");
            Console.WriteLine($"Role-based node summaries written to: mprr_top_nodes_hub.txt, leaf.txt, root.txt");
            Console.WriteLine($"Analysis log written to: {logPath}");

            // --- Object ID (mslk0c) mapping output ---
            string objectIdMapPath = Path.Combine(outputDir, "mprr_object_id_map.txt");
            var objectIdToNodes = new Dictionary<int, List<int>>();
            foreach (var idx in topNodes)
            {
                if (mslk0c.ContainsKey(idx))
                {
                    int objectId = mslk0c[idx];
                    if (!objectIdToNodes.ContainsKey(objectId)) objectIdToNodes[objectId] = new List<int>();
                    objectIdToNodes[objectId].Add(idx);
                }
            }
            using (var writer = new StreamWriter(objectIdMapPath))
            {
                writer.WriteLine("Object ID to Node Mapping (by mslk0c):\n");
                foreach (var objectId in objectIdToNodes.Keys.OrderBy(x => x))
                {
                    writer.WriteLine($"=== Object ID {objectId} ===");
                    var nodes = objectIdToNodes[objectId]
                        .OrderBy(idx => mslk00.ContainsKey(idx) ? mslk00[idx] : -1)
                        .ThenBy(idx => mslk01.ContainsKey(idx) ? mslk01[idx] : -1)
                        .ToList();
                    foreach (var idx in nodes)
                    {
                        int unk00 = mslk00.ContainsKey(idx) ? mslk00[idx] : -1;
                        int unk01 = mslk01.ContainsKey(idx) ? mslk01[idx] : -1;
                        string unk12 = mslk12.ContainsKey(idx) ? mslk12[idx].ToString() : "";
                        string roles = nodeSummaries.ContainsKey(idx) ? nodeSummaries[idx].FirstOrDefault(l => l.TrimStart().StartsWith("Role:")) ?? "" : "";
                        // Geometry references (if available)
                        // (Placeholder: actual geometry mapping would require more data, but we can note the node index for now)
                        writer.WriteLine($"  Node {idx}: unk00={unk00}, unk01={unk01}, unk12={unk12} {roles}");
                    }
                    writer.WriteLine("");
                }
            }
            Console.WriteLine($"Object ID map written to: {objectIdMapPath} (maps each object id to its nodes and types)");

            // --- Cross-reference file: node to MSVT vertex ---
            string typeObjectVertexMapPath = Path.Combine(outputDir, "mprr_type_object_vertex_map.txt");
            using (var writer = new StreamWriter(typeObjectVertexMapPath))
            {
                writer.WriteLine("Node to MSVT Vertex Mapping (by type and object id):\n");
                var nodesByType = topNodes.GroupBy(idx => (mslk00.ContainsKey(idx) ? mslk00[idx] : -1, mslk01.ContainsKey(idx) ? mslk01[idx] : -1))
                    .OrderBy(g => g.Key.Item1).ThenBy(g => g.Key.Item2);
                foreach (var typeGroup in nodesByType)
                {
                    var typeKey = typeGroup.Key;
                    var nodes = typeGroup.ToList();
                    writer.WriteLine($"=== Type unk00={typeKey.Item1}, unk01={typeKey.Item2} ===");
                    var nodesByObj = nodes.GroupBy(idx => mslk0c.ContainsKey(idx) ? mslk0c[idx] : -1)
                        .OrderBy(g => g.Key);
                    foreach (var objGroup in nodesByObj)
                    {
                        int objId = objGroup.Key;
                        writer.WriteLine($"  ObjectId {objId}:");
                        foreach (var idx in objGroup)
                        {
                            int unk10 = nodeIdxToUnk10.ContainsKey(idx) ? nodeIdxToUnk10[idx] : -1;
                            string vertexStr = "(no vertex)";
                            if (unk10 >= 0 && unk10 < msvi.Count)
                            {
                                uint vtxIdx = msvi[unk10];
                                if (vtxIdx < msvt.Count)
                                {
                                    var vtx = msvt[(int)vtxIdx];
                                    var world = vtx.ToWorldCoordinates();
                                    vertexStr = $"({world.X:F3}, {world.Y:F3}, {world.Z:F3})";
                                }
                            }
                            writer.WriteLine($"    Node {idx}: Unk10={unk10} Vertex {vertexStr}");
                        }
                    }
                    writer.WriteLine("");
                }
            }
            Console.WriteLine($"Node-to-vertex map written to: {typeObjectVertexMapPath}");

            // --- OBJ export for each type/object id group ---
            foreach (var typeGroup in topNodes.GroupBy(idx => (mslk00.ContainsKey(idx) ? mslk00[idx] : -1, mslk01.ContainsKey(idx) ? mslk01[idx] : -1)))
            {
                var typeKey = typeGroup.Key;
                var nodesByObj = typeGroup.GroupBy(idx => mslk0c.ContainsKey(idx) ? mslk0c[idx] : -1)
                    .OrderBy(g => g.Key);
                foreach (var objGroup in nodesByObj)
                {
                    int objId = objGroup.Key;
                    string objFileName = $"mprr_type_{typeKey.Item1}_{typeKey.Item2}_object_{objId}.obj";
                    string objFilePath = Path.Combine(outputDir, objFileName);
                    using var objWriter = new StreamWriter(objFilePath);
                    objWriter.WriteLine($"# OBJ export for Type unk00={typeKey.Item1}, unk01={typeKey.Item2}, ObjectId {objId}");
                    foreach (var idx in objGroup)
                    {
                        int unk10 = nodeIdxToUnk10.ContainsKey(idx) ? nodeIdxToUnk10[idx] : -1;
                        if (unk10 >= 0 && unk10 < msvi.Count)
                        {
                            uint vtxIdx = msvi[unk10];
                            if (vtxIdx < msvt.Count)
                            {
                                var vtx = msvt[(int)vtxIdx];
                                var world = vtx.ToWorldCoordinates();
                                objWriter.WriteLine($"v {world.X.ToString(System.Globalization.CultureInfo.InvariantCulture)} {world.Y.ToString(System.Globalization.CultureInfo.InvariantCulture)} {world.Z.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                            }
                        }
                    }
                }
            }
            Console.WriteLine($"OBJ files written for each type/object id group (see output directory, named as in mprr_type_hierarchy.txt)");

            // --- Load MSUR for surface/faces extraction ---
            var msur = pm4.MSUR?.Entries ?? new List<WoWToolbox.Core.Navigation.PM4.Chunks.MsurEntry>();
            // --- MSUR surface summary ---
            string msurSummaryPath = Path.Combine(outputDir, "msur_surface_summary.txt");
            using (var writer = new StreamWriter(msurSummaryPath))
            {
                writer.WriteLine("MSUR Surface Summary:\n");
                for (int i = 0; i < msur.Count; i++)
                {
                    var entry = msur[i];
                    writer.WriteLine($"Surface {i}: MsviFirstIndex={entry.MsviFirstIndex}, IndexCount={entry.IndexCount}, Flags=0x{entry.FlagsOrUnknown_0x00:X2}, MdosIndex={entry.MdosIndex}");
                    // List the MSVI indices
                    var indices = new List<uint>();
                    for (int j = 0; j < entry.IndexCount; j++)
                    {
                        int msviIdx = (int)entry.MsviFirstIndex + j;
                        if (msviIdx >= 0 && msviIdx < msvi.Count)
                            indices.Add(msvi[msviIdx]);
                    }
                    writer.WriteLine($"  MSVI Indices: {string.Join(", ", indices)}");
                }
            }
            Console.WriteLine($"MSUR surface summary written to: {msurSummaryPath}");
            // --- OBJ export for each MSUR surface ---
            for (int i = 0; i < msur.Count; i++)
            {
                var entry = msur[i];
                var indices = new List<uint>();
                for (int j = 0; j < entry.IndexCount; j++)
                {
                    int msviIdx = (int)entry.MsviFirstIndex + j;
                    if (msviIdx >= 0 && msviIdx < msvi.Count)
                        indices.Add(msvi[msviIdx]);
                }
                // Only export if at least 3 indices (triangle)
                if (indices.Count >= 3)
                {
                    string objFileName = $"msur_surface_{i}.obj";
                    string objFilePath = Path.Combine(outputDir, objFileName);
                    using var objWriter = new StreamWriter(objFilePath);
                    objWriter.WriteLine($"# OBJ export for MSUR Surface {i}");
                    // Write vertices
                    foreach (var vtxIdx in indices)
                    {
                        if (vtxIdx < msvt.Count)
                        {
                            var vtx = msvt[(int)vtxIdx];
                            var world = vtx.ToWorldCoordinates();
                            objWriter.WriteLine($"v {world.X.ToString(System.Globalization.CultureInfo.InvariantCulture)} {world.Y.ToString(System.Globalization.CultureInfo.InvariantCulture)} {world.Z.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                        }
                    }
                    // Write faces (as a fan for now)
                    for (int f = 1; f + 1 < indices.Count; f++)
                    {
                        objWriter.WriteLine($"f 1 {f + 1} {f + 2}");
                    }
                }
            }
            Console.WriteLine($"OBJ files written for each MSUR surface (see output directory, msur_surface_{{index}}.obj)");

            // --- Automated cross-referencing: assign MSUR surfaces to object id/type groups ---
            // Find all node indices for mapping, not just topNodes
            var allNodeIndices = nodeIndices.ToList();
            // Build a mapping from vertex index to (type, object id) groupings based on ALL node anchor points
            var vertexToGroups = new Dictionary<uint, List<(int unk00, int unk01, int objId)>>();
            foreach (var idx in allNodeIndices)
            {
                int unk00 = mslk00.ContainsKey(idx) ? mslk00[idx] : -1;
                int unk01 = mslk01.ContainsKey(idx) ? mslk01[idx] : -1;
                int objId = mslk0c.ContainsKey(idx) ? mslk0c[idx] : -1;
                int unk10 = nodeIdxToUnk10.ContainsKey(idx) ? nodeIdxToUnk10[idx] : -1;
                if (unk10 >= 0 && unk10 < msvi.Count)
                {
                    uint vtxIdx = msvi[unk10];
                    if (!vertexToGroups.ContainsKey(vtxIdx)) vertexToGroups[vtxIdx] = new List<(int, int, int)>();
                    vertexToGroups[vtxIdx].Add((unk00, unk01, objId));
                }
            }
            // For each MSUR surface, count which group its vertices belong to
            var surfaceToGroup = new List<(int surfaceIdx, int unk00, int unk01, int objId, double percent, int totalVerts, int matchedVerts, List<uint> indices)>();
            var groupToSurfaces = new Dictionary<(int, int, int), List<int>>();
            int assignedCount = 0;
            int unassignedCount = 0;
            var unassignedSurfaces = new List<(int surfaceIdx, List<uint> indices)>();
            for (int i = 0; i < msur.Count; i++)
            {
                var entry = msur[i];
                var indices = new List<uint>();
                for (int j = 0; j < entry.IndexCount; j++)
                {
                    int msviIdx = (int)entry.MsviFirstIndex + j;
                    if (msviIdx >= 0 && msviIdx < msvi.Count)
                        indices.Add(msvi[msviIdx]);
                }
                // Count group matches
                var groupCounts = new Dictionary<(int, int, int), int>();
                foreach (var vtxIdx in indices)
                {
                    if (vertexToGroups.TryGetValue(vtxIdx, out var groups))
                    {
                        foreach (var g in groups)
                        {
                            if (!groupCounts.ContainsKey(g)) groupCounts[g] = 0;
                            groupCounts[g]++;
                        }
                    }
                }
                // Assign to group with most matches
                (int, int, int) bestGroup = (-1, -1, -1);
                int bestCount = 0;
                foreach (var kv in groupCounts)
                {
                    if (kv.Value > bestCount)
                    {
                        bestGroup = kv.Key;
                        bestCount = kv.Value;
                    }
                }
                double percent = indices.Count > 0 ? (double)bestCount / indices.Count : 0.0;
                surfaceToGroup.Add((i, bestGroup.Item1, bestGroup.Item2, bestGroup.Item3, percent, indices.Count, bestCount, indices));
                if (bestGroup.Item1 != -1)
                {
                    assignedCount++;
                    if (!groupToSurfaces.ContainsKey(bestGroup)) groupToSurfaces[bestGroup] = new List<int>();
                    groupToSurfaces[bestGroup].Add(i);
                }
                else
                {
                    unassignedCount++;
                    if (unassignedSurfaces.Count < 10)
                        unassignedSurfaces.Add((i, indices));
                }
            }
            // Output mapping summary
            string msurObjectMapPath = Path.Combine(outputDir, "msur_surface_object_map.txt");
            using (var writer = new StreamWriter(msurObjectMapPath))
            {
                writer.WriteLine("MSUR Surface to Object/Type Mapping:\n");
                foreach (var s in surfaceToGroup)
                {
                    writer.WriteLine($"Surface {s.surfaceIdx}: Type unk00={s.unk00}, unk01={s.unk01}, ObjectId={s.objId}, Match={s.percent:P1} ({s.matchedVerts}/{s.totalVerts})");
                }
            }
            // Output assignment summary
            string assignmentSummaryPath = Path.Combine(outputDir, "msur_surface_assignment_summary.txt");
            using (var writer = new StreamWriter(assignmentSummaryPath))
            {
                writer.WriteLine($"Total MSUR surfaces: {msur.Count}");
                writer.WriteLine($"Assigned to group: {assignedCount} ({(100.0 * assignedCount / msur.Count):F1}%)");
                writer.WriteLine($"Unassigned: {unassignedCount} ({(100.0 * unassignedCount / msur.Count):F1}%)\n");
                writer.WriteLine("First 10 unassigned surfaces (surfaceIdx: vertex indices):");
                foreach (var (surfaceIdx, indices) in unassignedSurfaces)
                {
                    writer.WriteLine($"  Surface {surfaceIdx}: {string.Join(", ", indices)}");
                }
            }
            Console.WriteLine($"MSUR surface assignment summary written to: {assignmentSummaryPath}");
            Console.WriteLine($"Assigned: {assignedCount} ({(100.0 * assignedCount / msur.Count):F1}%), Unassigned: {unassignedCount} ({(100.0 * unassignedCount / msur.Count):F1}%)");

            // Export combined OBJ for each group
            foreach (var group in groupToSurfaces.Keys)
            {
                string objFileName = $"msur_group_type_{group.Item1}_{group.Item2}_object_{group.Item3}.obj";
                string objFilePath = Path.Combine(outputDir, objFileName);
                using var objWriter = new StreamWriter(objFilePath);
                objWriter.WriteLine($"# OBJ export for Type unk00={group.Item1}, unk01={group.Item2}, ObjectId {group.Item3}");
                var allVertices = new List<(float X, float Y, float Z)>();
                var allFaces = new List<List<int>>();
                var vertexMap = new Dictionary<uint, int>();
                int vtxCounter = 1;
                // Gather all unique vertices and build faces
                foreach (var surfIdx in groupToSurfaces[group])
                {
                    var entry = msur[surfIdx];
                    var indices = new List<uint>();
                    for (int j = 0; j < entry.IndexCount; j++)
                    {
                        int msviIdx = (int)entry.MsviFirstIndex + j;
                        if (msviIdx >= 0 && msviIdx < msvi.Count)
                            indices.Add(msvi[msviIdx]);
                    }
                    var faceIndices = new List<int>();
                    foreach (var vtxIdx in indices)
                    {
                        if (!vertexMap.ContainsKey(vtxIdx) && vtxIdx < msvt.Count)
                        {
                            var vtx = msvt[(int)vtxIdx];
                            var world = vtx.ToWorldCoordinates();
                            allVertices.Add((world.X, world.Y, world.Z));
                            vertexMap[vtxIdx] = vtxCounter++;
                        }
                        if (vertexMap.ContainsKey(vtxIdx))
                            faceIndices.Add(vertexMap[vtxIdx]);
                    }
                    if (faceIndices.Count >= 3)
                    {
                        for (int f = 1; f + 1 < faceIndices.Count; f++)
                        {
                            allFaces.Add(new List<int> { faceIndices[0], faceIndices[f], faceIndices[f + 1] });
                        }
                    }
                }
                // Write vertices
                foreach (var v in allVertices)
                {
                    objWriter.WriteLine($"v {v.X.ToString(System.Globalization.CultureInfo.InvariantCulture)} {v.Y.ToString(System.Globalization.CultureInfo.InvariantCulture)} {v.Z.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                }
                // Write faces
                foreach (var face in allFaces)
                {
                    objWriter.WriteLine($"f {string.Join(" ", face)}");
                }
                // Annotate with summary
                objWriter.WriteLine($"# Surfaces: {groupToSurfaces[group].Count}, Vertices: {allVertices.Count}, Faces: {allFaces.Count}");
            }
            // Print summary to console
            Console.WriteLine("\n--- Combined OBJ Group Summary ---");
            foreach (var group in groupToSurfaces.Keys)
            {
                Console.WriteLine($"Group unk00={group.Item1}, unk01={group.Item2}, ObjectId={group.Item3}: Surfaces={groupToSurfaces[group].Count}");
            }

            // --- Spatial clustering of MSUR surfaces ---
            // 1. Compute centroids for each surface
            var surfaceCentroids = new List<(int surfaceIdx, float X, float Y, float Z)>();
            for (int i = 0; i < msur.Count; i++)
            {
                var entry = msur[i];
                var indices = new List<uint>();
                for (int j = 0; j < entry.IndexCount; j++)
                {
                    int msviIdx = (int)entry.MsviFirstIndex + j;
                    if (msviIdx >= 0 && msviIdx < msvi.Count)
                        indices.Add(msvi[msviIdx]);
                }
                var verts = indices.Where(vtxIdx => vtxIdx < msvt.Count)
                                   .Select(vtxIdx => msvt[(int)vtxIdx].ToWorldCoordinates())
                                   .ToList();
                if (verts.Count > 0)
                {
                    float cx = verts.Average(v => v.X);
                    float cy = verts.Average(v => v.Y);
                    float cz = verts.Average(v => v.Z);
                    surfaceCentroids.Add((i, cx, cy, cz));
                }
            }
            // 2. Simple clustering: group centroids within threshold distance
            float threshold = 10.0f; // units, can be adjusted
            var clusterAssignments = new int[msur.Count];
            for (int i = 0; i < clusterAssignments.Length; i++) clusterAssignments[i] = -1;
            int nextClusterId = 0;
            for (int i = 0; i < surfaceCentroids.Count; i++)
            {
                if (clusterAssignments[surfaceCentroids[i].surfaceIdx] != -1) continue;
                // Start new cluster
                int clusterId = nextClusterId++;
                var queue = new Queue<int>();
                queue.Enqueue(i);
                clusterAssignments[surfaceCentroids[i].surfaceIdx] = clusterId;
                while (queue.Count > 0)
                {
                    int idx = queue.Dequeue();
                    var (surfIdxA, ax, ay, az) = surfaceCentroids[idx];
                    for (int j = 0; j < surfaceCentroids.Count; j++)
                    {
                        var (surfIdxB, bx, by, bz) = surfaceCentroids[j];
                        if (clusterAssignments[surfIdxB] != -1) continue;
                        float dx = ax - bx, dy = ay - by, dz = az - bz;
                        float dist = (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);
                        if (dist <= threshold)
                        {
                            clusterAssignments[surfIdxB] = clusterId;
                            queue.Enqueue(j);
                        }
                    }
                }
            }
            // 3. Export combined OBJ for each cluster
            var clusterToSurfaces = new Dictionary<int, List<int>>();
            for (int i = 0; i < clusterAssignments.Length; i++)
            {
                int cid = clusterAssignments[i];
                if (cid == -1) continue;
                if (!clusterToSurfaces.ContainsKey(cid)) clusterToSurfaces[cid] = new List<int>();
                clusterToSurfaces[cid].Add(i);
            }
            foreach (var cid in clusterToSurfaces.Keys)
            {
                string objFileName = $"msur_cluster_{cid}.obj";
                string objFilePath = Path.Combine(outputDir, objFileName);
                using var objWriter = new StreamWriter(objFilePath);
                objWriter.WriteLine($"# OBJ export for spatial cluster {cid}");
                var allVertices = new List<(float X, float Y, float Z)>();
                var allFaces = new List<List<int>>();
                var vertexMap = new Dictionary<uint, int>();
                int vtxCounter = 1;
                foreach (var surfIdx in clusterToSurfaces[cid])
                {
                    var entry = msur[surfIdx];
                    var indices = new List<uint>();
                    for (int j = 0; j < entry.IndexCount; j++)
                    {
                        int msviIdx = (int)entry.MsviFirstIndex + j;
                        if (msviIdx >= 0 && msviIdx < msvi.Count)
                            indices.Add(msvi[msviIdx]);
                    }
                    var faceIndices = new List<int>();
                    foreach (var vtxIdx in indices)
                    {
                        if (!vertexMap.ContainsKey(vtxIdx) && vtxIdx < msvt.Count)
                        {
                            var vtx = msvt[(int)vtxIdx];
                            var world = vtx.ToWorldCoordinates();
                            allVertices.Add((world.X, world.Y, world.Z));
                            vertexMap[vtxIdx] = vtxCounter++;
                        }
                        if (vertexMap.ContainsKey(vtxIdx))
                            faceIndices.Add(vertexMap[vtxIdx]);
                    }
                    if (faceIndices.Count >= 3)
                    {
                        for (int f = 1; f + 1 < faceIndices.Count; f++)
                        {
                            allFaces.Add(new List<int> { faceIndices[0], faceIndices[f], faceIndices[f + 1] });
                        }
                    }
                }
                foreach (var v in allVertices)
                {
                    objWriter.WriteLine($"v {v.X.ToString(System.Globalization.CultureInfo.InvariantCulture)} {v.Y.ToString(System.Globalization.CultureInfo.InvariantCulture)} {v.Z.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                }
                foreach (var face in allFaces)
                {
                    objWriter.WriteLine($"f {string.Join(" ", face)}");
                }
                objWriter.WriteLine($"# Surfaces: {clusterToSurfaces[cid].Count}, Vertices: {allVertices.Count}, Faces: {allFaces.Count}");
            }
            // 4. Output cluster summary
            string clusterSummaryPath = Path.Combine(outputDir, "msur_cluster_summary.txt");
            using (var writer = new StreamWriter(clusterSummaryPath))
            {
                writer.WriteLine($"Total clusters: {clusterToSurfaces.Count}");
                int maxSize = clusterToSurfaces.Values.Max(l => l.Count);
                int minSize = clusterToSurfaces.Values.Min(l => l.Count);
                writer.WriteLine($"Largest cluster size: {maxSize}");
                writer.WriteLine($"Smallest cluster size: {minSize}\n");
                foreach (var cid in clusterToSurfaces.Keys.OrderBy(x => x))
                {
                    var surfaces = clusterToSurfaces[cid];
                    var centroid = surfaceCentroids.FirstOrDefault(sc => sc.surfaceIdx == surfaces[0]);
                    writer.WriteLine($"Cluster {cid}: Surfaces={surfaces.Count}, Centroid=({centroid.X:F2}, {centroid.Y:F2}, {centroid.Z:F2})");
                    writer.WriteLine($"  Surfaces: {string.Join(", ", surfaces)}");
                }
            }
            Console.WriteLine($"Spatial clustering complete. {clusterToSurfaces.Count} clusters written to OBJ files. See msur_cluster_summary.txt for details.");

            // --- Export OBJs grouped by (unk00, unk01) ---
            var groupByUnk00Unk01 = new Dictionary<(int, int), List<int>>();
            for (int i = 0; i < surfaceToGroup.Count; i++)
            {
                var s = surfaceToGroup[i];
                if (s.unk00 == -1 || s.unk01 == -1) continue;
                var key = (s.unk00, s.unk01);
                if (!groupByUnk00Unk01.ContainsKey(key)) groupByUnk00Unk01[key] = new List<int>();
                groupByUnk00Unk01[key].Add(s.surfaceIdx);
            }
            foreach (var key in groupByUnk00Unk01.Keys)
            {
                string objFileName = $"msur_group_unk00_{key.Item1}_unk01_{key.Item2}.obj";
                string objFilePath = Path.Combine(outputDir, objFileName);
                using var objWriter = new StreamWriter(objFilePath);
                objWriter.WriteLine($"# OBJ export for group unk00={key.Item1}, unk01={key.Item2}");
                var allVertices = new List<(float X, float Y, float Z)>();
                var allFaces = new List<List<int>>();
                var vertexMap = new Dictionary<uint, int>();
                int vtxCounter = 1;
                foreach (var surfIdx in groupByUnk00Unk01[key])
                {
                    var entry = msur[surfIdx];
                    var indices = new List<uint>();
                    for (int j = 0; j < entry.IndexCount; j++)
                    {
                        int msviIdx = (int)entry.MsviFirstIndex + j;
                        if (msviIdx >= 0 && msviIdx < msvi.Count)
                            indices.Add(msvi[msviIdx]);
                    }
                    var faceIndices = new List<int>();
                    foreach (var vtxIdx in indices)
                    {
                        if (!vertexMap.ContainsKey(vtxIdx) && vtxIdx < msvt.Count)
                        {
                            var vtx = msvt[(int)vtxIdx];
                            var world = vtx.ToWorldCoordinates();
                            allVertices.Add((world.X, world.Y, world.Z));
                            vertexMap[vtxIdx] = vtxCounter++;
                        }
                        if (vertexMap.ContainsKey(vtxIdx))
                            faceIndices.Add(vertexMap[vtxIdx]);
                    }
                    if (faceIndices.Count >= 3)
                    {
                        for (int f = 1; f + 1 < faceIndices.Count; f++)
                        {
                            allFaces.Add(new List<int> { faceIndices[0], faceIndices[f], faceIndices[f + 1] });
                        }
                    }
                }
                foreach (var v in allVertices)
                {
                    objWriter.WriteLine($"v {v.X.ToString(System.Globalization.CultureInfo.InvariantCulture)} {v.Y.ToString(System.Globalization.CultureInfo.InvariantCulture)} {v.Z.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                }
                foreach (var face in allFaces)
                {
                    objWriter.WriteLine($"f {string.Join(" ", face)}");
                }
                objWriter.WriteLine($"# Surfaces: {groupByUnk00Unk01[key].Count}, Vertices: {allVertices.Count}, Faces: {allFaces.Count}");
            }
            Console.WriteLine("\n--- OBJ Grouped by (unk00, unk01) ---");
            foreach (var key in groupByUnk00Unk01.Keys)
            {
                Console.WriteLine($"Group unk00={key.Item1}, unk01={key.Item2}: Surfaces={groupByUnk00Unk01[key].Count}");
            }
            // --- Export OBJs grouped by unk00 (WMO root) ---
            var groupByUnk00 = new Dictionary<int, List<int>>();
            for (int i = 0; i < surfaceToGroup.Count; i++)
            {
                var s = surfaceToGroup[i];
                if (s.unk00 == -1) continue;
                if (!groupByUnk00.ContainsKey(s.unk00)) groupByUnk00[s.unk00] = new List<int>();
                groupByUnk00[s.unk00].Add(s.surfaceIdx);
            }
            foreach (var unk00 in groupByUnk00.Keys)
            {
                string objFileName = $"msur_wmo_root_{unk00}.obj";
                string objFilePath = Path.Combine(outputDir, objFileName);
                using var objWriter = new StreamWriter(objFilePath);
                objWriter.WriteLine($"# OBJ export for WMO root unk00={unk00}");
                var allVertices = new List<(float X, float Y, float Z)>();
                var allFaces = new List<List<int>>();
                var vertexMap = new Dictionary<uint, int>();
                int vtxCounter = 1;
                foreach (var surfIdx in groupByUnk00[unk00])
                {
                    var entry = msur[surfIdx];
                    var indices = new List<uint>();
                    for (int j = 0; j < entry.IndexCount; j++)
                    {
                        int msviIdx = (int)entry.MsviFirstIndex + j;
                        if (msviIdx >= 0 && msviIdx < msvi.Count)
                            indices.Add(msvi[msviIdx]);
                    }
                    var faceIndices = new List<int>();
                    foreach (var vtxIdx in indices)
                    {
                        if (!vertexMap.ContainsKey(vtxIdx) && vtxIdx < msvt.Count)
                        {
                            var vtx = msvt[(int)vtxIdx];
                            var world = vtx.ToWorldCoordinates();
                            allVertices.Add((world.X, world.Y, world.Z));
                            vertexMap[vtxIdx] = vtxCounter++;
                        }
                        if (vertexMap.ContainsKey(vtxIdx))
                            faceIndices.Add(vertexMap[vtxIdx]);
                    }
                    if (faceIndices.Count >= 3)
                    {
                        for (int f = 1; f + 1 < faceIndices.Count; f++)
                        {
                            allFaces.Add(new List<int> { faceIndices[0], faceIndices[f], faceIndices[f + 1] });
                        }
                    }
                }
                foreach (var v in allVertices)
                {
                    objWriter.WriteLine($"v {v.X.ToString(System.Globalization.CultureInfo.InvariantCulture)} {v.Y.ToString(System.Globalization.CultureInfo.InvariantCulture)} {v.Z.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                }
                foreach (var face in allFaces)
                {
                    objWriter.WriteLine($"f {string.Join(" ", face)}");
                }
                objWriter.WriteLine($"# Surfaces: {groupByUnk00[unk00].Count}, Vertices: {allVertices.Count}, Faces: {allFaces.Count}");
            }
            Console.WriteLine("\n--- OBJ Grouped by unk00 (WMO root) ---");
            foreach (var unk00 in groupByUnk00.Keys)
            {
                Console.WriteLine($"WMO root unk00={unk00}: Surfaces={groupByUnk00[unk00].Count}");
            }
        }
    }
}
