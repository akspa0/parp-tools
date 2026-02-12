### Task 4: WMO Path Investigation RESOLVED

**Status**: COMPLETED

**User Decision**: Use `world\wmo\dungeon\test\test.wmo` for WMO replacements

**Findings**:
- WMO replacement model: `world\wmo\dungeon\test\test.wmo` (30 chars)
- Description: Three nested square boxes for coordinate system testing
- Confirmed: Exists in all WoW versions (Alpha → 3.3.5 → 12.0)
- Small footprint: Minimal geometry, perfect for testing/invisible purposes
- Same padding strategy applies to WMO as M2 (null-padding to original length)

**Implementation**:
File size preserved exactly (no offset changes)  
No crashes or visual corruption

**Future Optimization**:
See `05-future-null-out-optimization.md` for research on potentially nulling out model names entirely instead of replacing. Requires manual ADT testing and Noggit code review first.  

### Step 1: Verify Debug Models ✅ (Quick Check)
**Time estimate**: 5 minutes

- [ ] Navigate to `test_data/0.5.3.3368/tree/World/SPELLS/`
- [ ] Confirm `Invisible.m2` exists
- [ ] Confirm `ErrorCube.m2` exists (fallback)
- [ ] Note file sizes

**If models don't exist**: Extract from Alpha MPQ or research alternative debug models

---

### Step 2: Create AlphaWDT Parser (Core Infrastructure)
**Time estimate**: 2-3 hours

**File**: `WoWRollback.Core/Services/Parsing/AlphaWdtParser.cs`

#### Requirements
1. Read WDT file in chunks
2. Parse standard WDT chunks:
   - MVER (version)
   - MPHD (header)
   - MAIN (tile flags)
3. Parse M2 data:
   - MMDX (model name strings, null-separated)
   - MMID (offsets into MMDX)
   - MDDF (placement data with UniqueID)
4. Parse WMO data:
   - MWMO (model name strings, null-separated)
   - MWID (offsets into MWMO)
   - MODF (placement data)

#### Implementation Template
```csharp
public class AlphaWdtParser
{
    public AlphaWdtData Parse(string wdtPath)
    {
        using var fs = File.OpenRead(wdtPath);
        using var br = new BinaryReader(fs);
        
        var data = new AlphaWdtData();
        
        while (br.BaseStream.Position < br.BaseStream.Length)
        {
            string chunkId = Encoding.ASCII.GetString(br.ReadBytes(4));
            int chunkSize = br.ReadInt32();
            byte[] chunkData = br.ReadBytes(chunkSize);
            
            switch (chunkId)
            {
                case "MVER": data.MverChunk = chunkData; break;
                case "MPHD": data.MphdChunk = chunkData; break;
                case "MAIN": data.MainChunk = chunkData; break;
                case "MMDX": data.MmdxChunk = chunkData; break;
                case "MMID": data.MmidChunk = chunkData; break;
                case "MDDF": data.MddfChunk = chunkData; break;
                case "MWMO": data.MwmoChunk = chunkData; break;
                case "MWID": data.MwidChunk = chunkData; break;
                case "MODF": data.ModfChunk = chunkData; break;
            }
        }
        
        ParseM2Placements(data);
        ParseWmoPlacements(data);
        
        return data;
    }
    
    private void ParseM2Placements(AlphaWdtData data)
    {
        // Parse MDDF chunk (36 bytes per entry)
        // Offset 0: MMID index (4 bytes)
        // Offset 4: UniqueID (4 bytes)
        // Offset 8-35: Position, rotation, scale, flags, etc.
        
        data.M2Placements = new List<M2Placement>();
        
        using var ms = new MemoryStream(data.MddfChunk);
        using var br = new BinaryReader(ms);
        
        while (br.BaseStream.Position < br.BaseStream.Length)
        {
            var placement = new M2Placement
            {
                NameIndex = br.ReadUInt32(),
                UniqueId = br.ReadUInt32(),
                Position = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                Rotation = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                Scale = br.ReadUInt16(),
                Flags = br.ReadUInt16()
            };
            
            data.M2Placements.Add(placement);
        }
    }
    
    public Dictionary<int, string> GetModelNames(byte[] mmdxChunk, byte[] mmidChunk)
    {
        // Parse MMDX: null-separated strings
        var names = new List<string>();
        var current = new StringBuilder();
        
        foreach (byte b in mmdxChunk)
        {
            if (b == 0)
            {
                if (current.Length > 0)
                {
                    names.Add(current.ToString());
                    current.Clear();
                }
            }
            else
            {
                current.Append((char)b);
            }
        }
        
        // Parse MMID: 4-byte offsets into MMDX
        var result = new Dictionary<int, string>();
        using var ms = new MemoryStream(mmidChunk);
        using var br = new BinaryReader(ms);
        
        int index = 0;
        while (br.BaseStream.Position < br.BaseStream.Length)
        {
            int offset = br.ReadInt32();
            result[index] = names[index]; // Map index to name
            index++;
        }
        
        return result;
    }
}

public class AlphaWdtData
{
    public byte[] MverChunk { get; set; }
    public byte[] MphdChunk { get; set; }
    public byte[] MainChunk { get; set; }
    
    public byte[] MmdxChunk { get; set; }
    public byte[] MmidChunk { get; set; }
    public byte[] MddfChunk { get; set; }
    
    public byte[] MwmoChunk { get; set; }
    public byte[] MwidChunk { get; set; }
    public byte[] ModfChunk { get; set; }
    
    public List<M2Placement> M2Placements { get; set; }
    public List<WmoPlacement> WmoPlacements { get; set; }
}

public class M2Placement
{
    public uint NameIndex { get; set; }
    public uint UniqueId { get; set; }
    public Vector3 Position { get; set; }
    public Vector3 Rotation { get; set; }
    public ushort Scale { get; set; }
    public ushort Flags { get; set; }
}
```

#### Testing
```csharp
// Test parsing on real WDT
var parser = new AlphaWdtParser();
var data = parser.Parse("test_data/0.5.3.3368/tree/World/Maps/Azeroth/Azeroth.wdt");

Console.WriteLine($"M2 Placements: {data.M2Placements.Count}");
Console.WriteLine($"WMO Placements: {data.WmoPlacements.Count}");

var modelNames = parser.GetModelNames(data.MmdxChunk, data.MmidChunk);
Console.WriteLine($"Unique M2 models: {modelNames.Count}");
```

---

### Step 3: Build AlphaWDT Patcher
**Time estimate**: 3-4 hours

**File**: `WoWRollback.Core/Services/Patching/AlphaWdtPatcher.cs`

#### Requirements
1. Accept rollback configuration (UniqueID ranges to KEEP)
2. Identify objects to remove (UniqueIDs NOT in config)
3. Replace model paths with `SPELLS\Invisible.m2` + null padding
4. Write patched WDT preserving file size

#### Implementation Template
```csharp
public class AlphaWdtPatcher
{
    private const string INVISIBLE_MODEL = "SPELLS\\Invisible.m2";
    private readonly AlphaWdtParser parser;
    
    public void PatchWdt(string wdtPath, RollbackConfig config, string outputPath)
    {
        // 1. Parse original WDT
        var data = parser.Parse(wdtPath);
        
        // 2. Get model names
        var modelNames = parser.GetModelNames(data.MmdxChunk, data.MmidChunk);
        
        // 3. Identify objects to remove
        var uniqueIdsToRemove = new HashSet<uint>();
        
        foreach (var placement in data.M2Placements)
        {
            if (!IsUniqueIdSelected(placement.UniqueId, config))
            {
                uniqueIdsToRemove.Add(placement.UniqueId);
            }
        }
        
        // 4. Get unique model names to replace
        var modelsToReplace = new HashSet<string>();
        
        foreach (var placement in data.M2Placements)
        {
            if (uniqueIdsToRemove.Contains(placement.UniqueId))
            {
                string modelName = modelNames[(int)placement.NameIndex];
                modelsToReplace.Add(modelName);
            }
        }
        
        // 5. Replace model paths in MMDX chunk
        var patchedMmdx = PatchModelNames(data.MmdxChunk, modelsToReplace);
        
        // 6. Write patched WDT
        WriteWdt(outputPath, data, patchedMmdx);
        
        // 7. Report
        Console.WriteLine($"Objects to remove: {uniqueIdsToRemove.Count}");
        Console.WriteLine($"Models replaced: {modelsToReplace.Count}");
    }
    
    private byte[] PatchModelNames(byte[] mmdxChunk, HashSet<string> modelsToReplace)
    {
        // Create mutable copy
        byte[] patched = new byte[mmdxChunk.Length];
        Array.Copy(mmdxChunk, patched, mmdxChunk.Length);
        
        // Find and replace each model name
        int offset = 0;
        var current = new StringBuilder();
        int startOffset = 0;
        
        for (int i = 0; i < patched.Length; i++)
        {
            if (patched[i] == 0)
            {
                if (current.Length > 0)
                {
                    string modelName = current.ToString();
                    
                    if (modelsToReplace.Contains(modelName))
                    {
                        // Replace with invisible model + null padding
                        ReplaceModelPath(patched, startOffset, modelName.Length);
                    }
                    
                    current.Clear();
                }
                
                startOffset = i + 1;
            }
            else
            {
                current.Append((char)patched[i]);
            }
        }
        
        return patched;
    }
    
    private void ReplaceModelPath(byte[] chunkData, int offset, int originalLength)
    {
        // Build replacement: "SPELLS\Invisible.m2" + null padding
        string replacement = INVISIBLE_MODEL.PadRight(originalLength, '\0');
        byte[] replacementBytes = Encoding.ASCII.GetBytes(replacement);
        
        // Replace in-place
        Array.Copy(replacementBytes, 0, chunkData, offset, originalLength);
    }
    
    private bool IsUniqueIdSelected(uint uniqueId, RollbackConfig config)
    {
        foreach (var range in config.SelectedRanges)
        {
            if (uniqueId >= range.Min && uniqueId <= range.Max)
                return true;
        }
        return false;
    }
    
    private void WriteWdt(string outputPath, AlphaWdtData data, byte[] patchedMmdx)
    {
        using var fs = File.Create(outputPath);
        using var bw = new BinaryWriter(fs);
        
        // Write chunks in order
        WriteChunk(bw, "MVER", data.MverChunk);
        WriteChunk(bw, "MPHD", data.MphdChunk);
        WriteChunk(bw, "MAIN", data.MainChunk);
        WriteChunk(bw, "MWMO", data.MwmoChunk);
        WriteChunk(bw, "MWID", data.MwidChunk);
        WriteChunk(bw, "MODF", data.ModfChunk);
        WriteChunk(bw, "MMDX", patchedMmdx);  // Patched!
        WriteChunk(bw, "MMID", data.MmidChunk);
        WriteChunk(bw, "MDDF", data.MddfChunk);
    }
    
    private void WriteChunk(BinaryWriter bw, string chunkId, byte[] data)
    {
        bw.Write(Encoding.ASCII.GetBytes(chunkId));  // 4 bytes
        bw.Write(data.Length);                       // 4 bytes (size)
        bw.Write(data);                              // N bytes (data)
    }
}
```

---

### Step 4: Add CLI Command with Lineage Tracking
**Time estimate**: 2 hours (including lineage report generation)

**File**: `WoWRollback.Cli/Commands/RollbackCommands.cs`

```csharp
[Command("rollback patch-alpha-wdt")]
public class PatchAlphaWdtCommand
{
    [Option("--wdt", Description = "Path to Alpha WDT file")]
    public string WdtPath { get; set; }
    
    [Option("--config", Description = "Path to rollback config JSON")]
    public string ConfigPath { get; set; }
    
    [Option("--output", Description = "Output path for patched WDT")]
    public string OutputPath { get; set; }
    
    [Option("--report", Description = "Generate lineage report (default: true)")]
    public bool GenerateReport { get; set; } = true;
    
    public async Task<int> OnExecuteAsync()
    {
        // Load config
        var config = JsonSerializer.Deserialize<RollbackConfig>(
            await File.ReadAllTextAsync(ConfigPath));
        
        Console.WriteLine($"Patching WDT: {Path.GetFileName(WdtPath)}");
        Console.WriteLine($"Config: {Path.GetFileName(ConfigPath)}");
        Console.WriteLine($"Output: {OutputPath}");
        
        // Patch WDT
        var patcher = new AlphaWdtPatcher();
        var stats = patcher.PatchWdt(WdtPath, config, OutputPath);
        
        Console.WriteLine($"\n✅ Patched WDT written to: {OutputPath}");
        Console.WriteLine($"   Objects kept: {stats.ObjectsKept}");
        Console.WriteLine($"   Objects replaced: {stats.ObjectsReplaced}");
        Console.WriteLine($"   M2 replaced: {stats.M2Replaced}");
        Console.WriteLine($"   WMO replaced: {stats.WmoReplaced}");
        
        // Generate lineage report
        if (GenerateReport)
        {
            var reportPath = OutputPath.Replace(".wdt", "_lineage.md");
            await GenerateLineageReport(config, stats, reportPath);
            Console.WriteLine($"   Lineage report: {reportPath}");
        }
        
        return 0;
    }
    
    private async Task GenerateLineageReport(
        RollbackConfig config, 
        PatchingStatistics stats, 
        string reportPath)
    {
        var report = new StringBuilder();
        
        report.AppendLine("# Rollback Lineage Report");
        report.AppendLine();
        report.AppendLine($"**Generated**: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        report.AppendLine($"**Generator**: WoWRollback CLI v{Assembly.GetExecutingAssembly().GetName().Version}");
        report.AppendLine();
        
        report.AppendLine("## Source Configuration");
        report.AppendLine();
        report.AppendLine($"- **Map**: {config.Map}");
        report.AppendLine($"- **Version**: {config.Version}");
        report.AppendLine($"- **Mode**: {config.Mode}");
        report.AppendLine($"- **Config Created**: {config.Metadata?.Created}");
        report.AppendLine($"- **Config Modified**: {config.Metadata?.Modified}");
        report.AppendLine($"- **Config Generator**: {config.Metadata?.Generator}");
        report.AppendLine();
        
        report.AppendLine("## Patching Statistics");
        report.AppendLine();
        report.AppendLine($"- **Objects Kept**: {stats.ObjectsKept}");
        report.AppendLine($"- **Objects Replaced**: {stats.ObjectsReplaced}");
        report.AppendLine($"  - M2 (Doodads): {stats.M2Replaced}");
        report.AppendLine($"  - WMO (Buildings): {stats.WmoReplaced}");
        report.AppendLine($"- **File Size**: {stats.OriginalFileSize} bytes → {stats.PatchedFileSize} bytes");
        report.AppendLine($"- **SHA256 (Original)**: `{stats.OriginalSha256}`");
        report.AppendLine($"- **SHA256 (Patched)**: `{stats.PatchedSha256}`");
        report.AppendLine();
        
        report.AppendLine("## Invisible Model Mappings");
        report.AppendLine();
        report.AppendLine("Replaced objects now reference:");
        report.AppendLine($"- **M2 (Doodads)**: `SPELLS\\Invisible.m2`");
        report.AppendLine($"- **WMO (Buildings)**: `world\\wmo\\dungeon\\test\\test.wmo`");
        report.AppendLine();
        
        if (config.Mode == "advanced" && config.Tiles != null)
        {
            report.AppendLine("## Per-Tile Details");
            report.AppendLine();
            report.AppendLine($"**Total Tiles Configured**: {config.Tiles.Count}");
            report.AppendLine();
            
            foreach (var tile in config.Tiles.OrderBy(t => t.Key))
            {
                report.AppendLine($"### Tile {tile.Key}");
                report.AppendLine();
                report.AppendLine($"**Selected Ranges**: {tile.Value.SelectedRanges.Count}");
                report.AppendLine();
                
                if (tile.Value.SelectedRanges.Any())
                {
                    report.AppendLine("| Min UniqueID | Max UniqueID | Count | Type |");
                    report.AppendLine("|--------------|--------------|-------|------|");
                    foreach (var range in tile.Value.SelectedRanges)
                    {
                        report.AppendLine($"| {range.Min} | {range.Max} | {range.Count} | {range.Type} |");
                    }
                    report.AppendLine();
                }
            }
        }
        
        report.AppendLine("## Reproducibility");
        report.AppendLine();
        report.AppendLine("To reproduce this patching operation:");
        report.AppendLine();
        report.AppendLine("```powershell");
        report.AppendLine($"dotnet run --project WoWRollback.Cli -- rollback patch-alpha-wdt \\");
        report.AppendLine($"  --wdt \"{stats.OriginalWdtPath}\" \\");
        report.AppendLine($"  --config \"{stats.ConfigPath}\" \\");
        report.AppendLine($"  --output \"{stats.OutputPath}\"");
        report.AppendLine("```");
        report.AppendLine();
        
        report.AppendLine("## Verification");
        report.AppendLine();
        report.AppendLine("To verify the patched file:");
        report.AppendLine();
        report.AppendLine("1. Check SHA256 hash matches: `" + stats.PatchedSha256 + "`");
        report.AppendLine("2. Verify file size: " + stats.PatchedFileSize + " bytes");
        report.AppendLine("3. Test in WoW Alpha " + config.Version + " client");
        report.AppendLine("4. Confirm removed objects are invisible/missing");
        report.AppendLine();
        
        report.AppendLine("---");
        report.AppendLine("*This report provides complete lineage tracking for rollback operations.*");
        
        await File.WriteAllTextAsync(reportPath, report.ToString());
    }
}

public class PatchingStatistics
{
    public int ObjectsKept { get; set; }
    public int ObjectsReplaced { get; set; }
    public int M2Replaced { get; set; }
    public int WmoReplaced { get; set; }
    public long OriginalFileSize { get; set; }
    public long PatchedFileSize { get; set; }
    public string OriginalSha256 { get; set; }
    public string PatchedSha256 { get; set; }
    public string OriginalWdtPath { get; set; }
    public string ConfigPath { get; set; }
    public string OutputPath { get; set; }
}
```

**Usage**:
```powershell
dotnet run --project WoWRollback.Cli -- rollback patch-alpha-wdt \
  --wdt test_data/0.5.3.3368/tree/World/Maps/Azeroth/Azeroth.wdt \
  --config rollback_config.json \
  --output patched/Azeroth.wdt
```

---

### Step 5: Testing & Validation
**Time estimate**: 2-3 hours

#### Unit Tests
```csharp
[Fact]
public void Parser_ParsesWdtCorrectly()
{
    var parser = new AlphaWdtParser();
    var data = parser.Parse("test_data/Azeroth.wdt");
    
    Assert.NotNull(data.MmdxChunk);
    Assert.NotNull(data.MddfChunk);
    Assert.True(data.M2Placements.Count > 0);
}

[Fact]
public void Patcher_PreservesFileSize()
{
    var originalSize = new FileInfo(originalWdt).Length;
    
    patcher.PatchWdt(originalWdt, config, patchedWdt);
    
    var patchedSize = new FileInfo(patchedWdt).Length;
    Assert.Equal(originalSize, patchedSize);
}
```

#### Manual Testing Checklist
1. **Backup original WDT**: `Azeroth.wdt.bak`
2. **Create simple config**: Keep only UniqueIDs 1000-2000
3. **Run patch command**
4. **Verify file size**: Original vs Patched (should match)
5. **Copy patched WDT** to Alpha client data folder
6. **Launch Alpha 0.5.3 client**
7. **Navigate to test area** (e.g., Elwynn Forest)
8. **Verify**:
   - [ ] World loads without errors
   - [ ] Selected objects (1000-2000) are visible
   - [ ] Unselected objects are invisible
   - [ ] No crashes or visual corruption

---

## 📊 Progress Tracking

### Sprint Status
- [x] Step 1: Verify debug models (5 min)
- [ ] Step 2: Build parser (2-3 hours)
- [ ] Step 3: Build patcher (3-4 hours)
- [ ] Step 4: Add CLI command (1 hour)
- [ ] Step 5: Testing (2-3 hours)

**Total Estimated Time**: 8-11 hours

### Current Blockers
- [ ] None yet (pending Step 1 verification)

### Questions for User
1. Do you have Alpha 0.5.3 client for testing?
2. Which map should we test first (Azeroth, Kalimdor, or smaller instance)?
3. Should we also patch WMO objects, or focus on M2 only for MVP?

---

## 🎉 MVP Delivery

**When MVP is complete, user can**:
1. Generate per-tile UniqueID CSVs
2. Create rollback config (manually for now)
3. Run CLI command to patch AlphaWDT
4. Test patched map in Alpha client
5. Verify selected objects preserved, others invisible

**Future enhancements** (post-MVP):
- Tile selection UI (tile.html)
- Config export/import from UI
- Batch processing for multiple maps
- LK ADT patching (Phase 5B)

---

**Ready to start Step 1!** 🦀
