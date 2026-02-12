# AI Assistant Guidelines for GillijimProject

## General Approach

1. **Always understand before coding** - Read relevant source files first
2. **Preserve existing functionality** - Don't break working features
3. **Add debugging output** - Useful for troubleshooting file discovery
4. **Use consistent patterns** - Follow existing code style

## Common Issues & Solutions

### MPQ File Discovery
- **Symptom**: Only BLP/WMO files show, no MDX/M2
- **Cause**: MPQ internal files not added to `_fileSet`
- **Fix**: Call `_mpq.GetAllKnownFiles()` and add to set
- **Debug**: Check "[MpqDataSource] Added X MPQ internal files"

### Case Sensitivity
- **Symptom**: Files not found despite existing
- **Cause**: Alpha uses uppercase extensions (.MPQ)
- **Fix**: Use `StringComparer.OrdinalIgnoreCase`
- **Debug**: Check file extension handling

### WMO Nested Archives
- **Symptom**: WMO files won't load
- **Cause**: WMO data stored in `.wmo.MPQ` archives
- **Fix**: Use `ScanWmoMpqArchives()` for nested scanning
- **Debug**: Check "[MpqDataSource] Added WMO MPQ:" messages

## Development Workflow

1. **Read relevant code** - Understand current implementation
2. **Make minimal changes** - Fix only what's broken
3. **Build and test** - Run `dotnet build --no-restore`
4. **Verify with debug output** - Check file counts by extension

## Key Files to Understand

| File | Purpose |
|------|---------|
| `MpqDataSource.cs` | MPQ file access and file list building |
| `NativeMpqService.cs` | Low-level MPQ archive reading |
| `ViewerApp.cs` | Main application entry point |
| `ModelRenderer.cs` | 3D model rendering |

## Code Patterns

### File Extension Check
```csharp
var ext = Path.GetExtension(file).ToLowerInvariant();
if (ext is ".mdx" or ".wmo" or ".m2" or ".blp")
```

### Case-Insensitive Comparison
```csharp
var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
```

### Path Normalization
```csharp
var virtualPath = file.Replace('/', '\\');
```

## Debug Output Examples

### File Discovery
```
[MpqDataSource] Ready. 4554 known files:
  .blp: 3554 files
  .wmo: 1000 files
  .mdx: 500 files
  .m2: 200 files
```

### WMO Archive Scanning
```
[MpqDataSource] Added WMO MPQ: World\wmo\Dungeon\test.wmo
```

### Scanning Progress
```
[MpqDataSource] Scanning root: H:\053-client\Data
[MpqDataSource] Added 1234 MPQ internal files.
```

## Testing Checklist

- [ ] Build succeeds (`dotnet build --no-restore`)
- [ ] MDX files appear in file browser
- [ ] M2 files appear in file browser
- [ ] WMO files load correctly
- [ ] BLP textures display properly
- [ ] Animation playback works (if applicable)

## When Stuck

1. Check debug output for file counts
2. Verify MPQ archives are loaded
3. Confirm `_fileSet` contains expected files
4. Test with known good files
5. Check path normalization
