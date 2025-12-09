// PM4-WMO Pathfinding Comparison Tool with MPQ Support
// Extracts clean WMO v17 from 2.4.3 MPQ archive for comparison

using System.Numerics;
using StormLibSharp;
using WoWRollback.Core.Services.Archive;
using WoWRollback.PM4Module;

Console.WriteLine("=== PM4-WMO Pathfinding Comparison Tool ===\n");

// === Part 1: PM4 Analysis ===
var pm4DataPath = @"I:\parp-tools\pm4next-branch\parp-tools\gillijimproject_refactor\test_data\development";
var pm4File = Path.Combine(pm4DataPath, "development_14_50.pm4");

Console.WriteLine("--- PM4 Pathfinding Data ---\n");

if (File.Exists(pm4File))
{
    var pm4 = PM4File.FromFile(pm4File);
    Console.WriteLine($"File: {Path.GetFileName(pm4File)}");
    Console.WriteLine($"Total Surfaces: {pm4.Surfaces.Count}");
    Console.WriteLine($"Total Mesh Verts: {pm4.MeshVertices.Count}");
    
    // Group by CK24 (distinct objects)
    var pm4Objects = pm4.Surfaces
        .Where(s => s.CK24 != 0) // Exclude 0x000000 (M2/terrain)
        .GroupBy(s => s.CK24)
        .Select(g => CreatePm4Fingerprint(pm4, g.Key, g.ToList()))
        .OrderByDescending(f => f.SurfaceCount)
        .ToList();
    
    Console.WriteLine($"WMO Objects (CK24 != 0): {pm4Objects.Count}\n");
    
    Console.WriteLine("| CK24 | Surfaces | Verts | Size WxLxH | Type |");
    Console.WriteLine("|------|----------|-------|------------|------|");
    foreach (var obj in pm4Objects.Take(10))
    {
        string objType = ClassifyPm4Object(obj);
        Console.WriteLine($"| 0x{obj.CK24:X6} | {obj.SurfaceCount,8} | {obj.VertexCount,5} | {obj.Size.X:F0}x{obj.Size.Y:F0}x{obj.Size.Z:F0} | {objType} |");
    }
}
else
{
    Console.WriteLine($"PM4 file not found: {pm4File}");
}

// === Part 2: WMO Analysis from MPQ ===
Console.WriteLine("\n--- WMO Pathfinding Extraction (from 2.4.3 MPQ) ---\n");

var clientPath = @"G:\WoW\WoWArchive-0.X-3.X\Mount\2.X_Retail_Windows_enUS_2.4.3.8606\World of Warcraft";
var wmoPath = @"World\wmo\Azeroth\Buildings\Stormwind\Stormwind.wmo";

if (Directory.Exists(clientPath))
{
    Console.WriteLine($"Client: {clientPath}");
    Console.WriteLine($"Extracting: {wmoPath}\n");
    
    try
    {
        var mpqs = ArchiveLocator.LocateMpqs(clientPath);
        Console.WriteLine($"Found {mpqs.Count} MPQ archives");
        
        byte[]? wmoData = null;
        foreach (var mpqPath in mpqs.Reverse()) // Reverse to get highest patch priority first
        {
            try
            {
                using var mpq = new MpqArchive(mpqPath, FileAccess.Read);
                if (mpq.HasFile(wmoPath))
                {
                    using var stream = mpq.OpenFile(wmoPath);
                    if (stream != null && stream.CanRead)
                    {
                        wmoData = new byte[stream.Length];
                        stream.Read(wmoData, 0, wmoData.Length);
                        Console.WriteLine($"Found in: {Path.GetFileName(mpqPath)}");
                        Console.WriteLine($"WMO Size: {wmoData.Length:N0} bytes");
                        break;
                    }
                }
            }
            catch { /* Try next MPQ */ }
        }
        
        if (wmoData != null)
        {
            // Extract WMO and analyze
            var extractor = new WmoPathfindingExtractor();
            var wmoResult = extractor.ExtractFromBytes(wmoData, wmoPath);
            
            Console.WriteLine($"\nWMO Analysis:");
            Console.WriteLine($"  Walkable Surfaces: {wmoResult.SurfaceCount}");
            Console.WriteLine($"  Unique Vertices: {wmoResult.VertexCount}");
            if (wmoResult.SurfaceCount > 0)
            {
                Console.WriteLine($"  Bounding Box: {wmoResult.BoundsMin} -> {wmoResult.BoundsMax}");
                Console.WriteLine($"  Size: {wmoResult.Size.X:F1} x {wmoResult.Size.Y:F1} x {wmoResult.Size.Z:F1}");
            }
        }
        else
        {
            Console.WriteLine($"WMO not found in any MPQ: {wmoPath}");
            
            // List some available WMOs
            Console.WriteLine("\nSearching for available WMOs...");
            var wmos = new List<string>();
            foreach (var mpqPath in mpqs.Take(3))
            {
                try
                {
                    using var mpq = new MpqArchive(mpqPath, FileAccess.Read);
                    if (mpq.HasFile("(listfile)"))
                    {
                        using var stream = mpq.OpenFile("(listfile)");
                        using var reader = new StreamReader(stream);
                        while (!reader.EndOfStream)
                        {
                            var line = reader.ReadLine();
                            if (line?.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase) == true && 
                                !line.Contains("_00") && wmos.Count < 20)
                            {
                                wmos.Add(line);
                            }
                        }
                    }
                }
                catch { }
            }
            
            if (wmos.Count > 0)
            {
                Console.WriteLine("Sample WMO root files:");
                foreach (var w in wmos.Take(10))
                    Console.WriteLine($"  {w}");
            }
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"MPQ Error: {ex.Message}");
    }
}
else
{
    Console.WriteLine($"Client path not found: {clientPath}");
    Console.WriteLine("Please update the path to your 2.4.3 client installation.");
}

// === Helper Functions ===

PM4ObjectFingerprint CreatePm4Fingerprint(PM4File pm4, uint ck24, List<MsurEntry> surfaces)
{
    var fp = new PM4ObjectFingerprint { CK24 = ck24, SurfaceCount = surfaces.Count };
    
    var usedVerts = new HashSet<int>();
    foreach (var surf in surfaces)
    {
        for (int i = 0; i < surf.IndexCount; i++)
        {
            int idx = (int)(surf.MsviFirstIndex + i);
            if (idx < pm4.MeshIndices.Count)
            {
                usedVerts.Add((int)pm4.MeshIndices[idx]);
            }
        }
    }
    
    fp.VertexCount = usedVerts.Count;
    
    var verts = usedVerts
        .Where(idx => idx < pm4.MeshVertices.Count)
        .Select(idx => pm4.MeshVertices[idx])
        .ToList();
    
    if (verts.Count > 0)
    {
        fp.BoundsMin = new Vector3(verts.Min(v => v.X), verts.Min(v => v.Y), verts.Min(v => v.Z));
        fp.BoundsMax = new Vector3(verts.Max(v => v.X), verts.Max(v => v.Y), verts.Max(v => v.Z));
    }
    
    fp.HeightRange = (surfaces.Min(s => s.Height), surfaces.Max(s => s.Height));
    
    return fp;
}

string ClassifyPm4Object(PM4ObjectFingerprint fp)
{
    var aspectXY = fp.Size.X > 0 ? fp.Size.Y / fp.Size.X : 1;
    var heightRatio = fp.Size.Z > 0 ? Math.Max(fp.Size.X, fp.Size.Y) / fp.Size.Z : 1;
    
    if (fp.Size.Z > 80 && heightRatio < 2) return "Tower";
    if (fp.Size.Z > 40) return "Building";
    if (fp.SurfaceCount < 50) return "Small";
    if (aspectXY > 3 || aspectXY < 0.33) return "Bridge/Wall";
    return "Structure";
}

class PM4ObjectFingerprint
{
    public uint CK24 { get; set; }
    public int SurfaceCount { get; set; }
    public int VertexCount { get; set; }
    public Vector3 BoundsMin { get; set; }
    public Vector3 BoundsMax { get; set; }
    public Vector3 Size => BoundsMax - BoundsMin;
    public (float min, float max) HeightRange { get; set; }
}
