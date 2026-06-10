# Phase 4 Investigation Checklist

**Goal**: Verify debug models exist and determine optimal padding strategy for model path replacement.

---

## ✅ Investigation Tasks

### Task 1: Verify Debug Models Exist in Alpha Data
**Status**: 🔍 TO DO

**Steps**:
1. Check if `SPELLS\Invisible.m2` exists in Alpha 0.5.3 data tree
2. Check if `SPELLS\ErrorCube.m2` exists as fallback
3. Document full path (e.g., `test_data/0.5.3.3368/tree/World/SPELLS/Invisible.m2`)
4. Optional: Extract M2 file and verify structure

**Expected Locations**:
```
test_data/
└── 0.5.3.3368/
    └── tree/
        └── World/
            └── SPELLS/
                ├── Invisible.m2
                └── ErrorCube.m2
```

**Findings**:
- [ ] `SPELLS\Invisible.m2` found: YES / NO
- [ ] `SPELLS\ErrorCube.m2` found: YES / NO
- [ ] File size: _____ bytes
- [ ] Alternative paths tried: _____

---

### Task 2: Analyze Model Path Lengths in Alpha Data
**Status**: 🔍 TO DO

**Goal**: Understand the distribution of model path lengths to validate padding strategy.

**Implementation**:
Create `WoWRollback.Core/Services/Analysis/ModelPathAnalyzer.cs`:

```csharp
public class ModelPathAnalyzer
{
    public Dictionary<int, List<string>> GroupPathsByLength(List<Placement> placements)
    {
        var groups = new Dictionary<int, List<string>>();
        
        foreach (var placement in placements)
        {
            var path = placement.ModelPath;
            var length = path.Length;
            
            if (!groups.ContainsKey(length))
                groups[length] = new List<string>();
            
            if (!groups[length].Contains(path))
                groups[length].Add(path);
        }
        
        return groups;
    }
    
    public void GenerateReport(string mapName, Dictionary<int, List<string>> groups)
    {
        Console.WriteLine($"\n=== Model Path Length Analysis: {mapName} ===\n");
        
        var sortedLengths = groups.Keys.OrderBy(k => k).ToList();
        
        Console.WriteLine($"Shortest path: {sortedLengths.First()} chars");
        Console.WriteLine($"Longest path: {sortedLengths.Last()} chars");
        Console.WriteLine($"Total unique lengths: {sortedLengths.Count}\n");
        
        Console.WriteLine("Length Distribution:");
        foreach (var length in sortedLengths)
        {
            Console.WriteLine($"  {length,3} chars: {groups[length].Count,5} unique paths");
        }
        
        Console.WriteLine("\nSample paths by length:");
        foreach (var length in sortedLengths.Take(5))
        {
            Console.WriteLine($"\n  [{length} chars]");
            foreach (var path in groups[length].Take(3))
            {
                Console.WriteLine($"    {path}");
            }
        }
    }
}
```

**Add CLI Command**:
```powershell
dotnet run --project WoWRollback.Cli -- analyze-model-paths \
  --map Azeroth \
  --version 0.5.3.3368 \
  --alpha-root test_data
```

**Expected Output**:
```
=== Model Path Length Analysis: Azeroth ===

Shortest path: 15 chars
Longest path: 78 chars
Total unique lengths: 24

Length Distribution:
   15 chars:    12 unique paths
   20 chars:   145 unique paths
   25 chars:   892 unique paths
   ...

Sample paths by length:
  [15 chars]
    SPELLS\Test.m2
    World\Tree.m2
    ...
```

**Findings**:
- [ ] Shortest path length: _____ chars
- [ ] Longest path length: _____ chars
- [ ] Most common length: _____ chars (_____ paths)
- [ ] `SPELLS\Invisible.m2` length: 19 chars (confirmed)
- [ ] Average padding needed: _____ chars

---

### Task 3: Manual Hex-Edit Test
**Status**: 🔍 TO DO

**Goal**: Verify that null-padded replacement works in WoW client.

**Steps**:
1. **Select Test ADT**:
   - Choose a small map for faster testing (e.g., `DeadminesInstance`)
   - Pick a tile with 1-2 visible objects
   - Backup original ADT: `Azeroth_39_37.adt.bak`

2. **Analyze ADT Structure**:
   - Open in hex editor (HxD, 010 Editor, etc.)
   - Locate MMDX chunk (M2 model names)
   - Find a model path to replace (e.g., `World\Azeroth\Elwynn\Building.m2`)
   - Note original path length and offset

3. **Replace with Invisible Model**:
   ```
   Original (33 chars): "World\Azeroth\Elwynn\Building.m2"
   
   Replace with (33 chars):
   "SPELLS\Invisible.m2" (19 chars) + 14 null bytes (0x00)
   
   Hex view:
   53 50 45 4C 4C 53 5C 49 6E 76 69 73 69 62 6C 65 2E 6D 32  S P E L L S \ I n v i s i b l e . m 2
   00 00 00 00 00 00 00 00 00 00 00 00 00 00                  (14 null bytes)
   ```

4. **Verify File Size**:
   - Before: _____ bytes
   - After: _____ bytes (should be IDENTICAL)

5. **Test in WoW Client**:
   - Launch Alpha 0.5.3 client
   - Navigate to test area (Elwynn, coordinates X,Y)
   - Check results:
     - [ ] ADT loads without error
     - [ ] Original object is invisible
     - [ ] No crashes or visual glitches
     - [ ] Other objects still visible

**Alternative Padding Tests** (if null-padding fails):
- Try space-padding: `SPELLS\Invisible.m2` + spaces
- Try underscore-padding: `SPELLS\Invisible.m2_____________`
- Try zero-then-space: `SPELLS\Invisible.m2\0` + spaces

**Findings**:
- [ ] Null-padding: WORKS / FAILS
- [ ] Space-padding: WORKS / FAILS (if tested)
- [ ] Underscore-padding: WORKS / FAILS (if tested)
- [ ] Recommended strategy: _____
- [ ] Notes: _____

---

### Task 4: WMO Path Investigation
**Status**: 🔍 TO DO

**Goal**: Determine if WMO objects need different handling than M2 models.

**Questions**:
1. Does `SPELLS\Invisible.wmo` exist?
2. Can we use `SPELLS\Invisible.m2` for WMO paths?
3. Are WMO paths structured differently?

**Steps**:
1. Check for `SPELLS\Invisible.wmo` in Alpha data
2. Analyze MWMO chunk structure (WMO model names)
3. Compare to MMDX chunk structure (M2 model names)
4. Test replacement on WMO object if needed

**Findings**:
- [ ] `SPELLS\Invisible.wmo` exists: YES / NO
- [ ] WMO paths use same structure as M2: YES / NO
- [ ] Can use same replacement strategy: YES / NO
- [ ] Notes: _____

---

## 📊 Summary Report Template

**Investigation Complete**: ✅ / ❌

### Debug Models
- **Invisible.m2 exists**: YES / NO
- **ErrorCube.m2 exists**: YES / NO
- **Preferred model**: _____

### Path Length Analysis
- **Shortest**: _____ chars
- **Longest**: _____ chars
- **Average**: _____ chars
- **Most common**: _____ chars

### Padding Strategy
- **Recommended**: Null-padding / Space-padding / Other
- **Tested**: YES / NO
- **Client accepts**: YES / NO
- **Concerns**: _____

### WMO Handling
- **Same as M2**: YES / NO
- **Special handling needed**: YES / NO

### Blockers
- [ ] No debug models found → _____
- [ ] Padding strategy fails → _____
- [ ] WMO requires different approach → _____

### Next Steps
✅ **Ready for Phase 5 (ADT Patching)**: YES / NO

If YES:
- Proceed to implement `AdtPatcher.cs`
- Use confirmed padding strategy
- Start with single-tile patching

If NO:
- Address blockers: _____
- Research alternatives: _____

---

**Investigation started**: _____  
**Investigation completed**: _____  
**Total time**: _____
