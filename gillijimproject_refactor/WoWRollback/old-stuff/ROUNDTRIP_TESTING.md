# Round-Trip Testing Guide

## Overview

The round-trip tester validates the Alpha→LK→Alpha conversion pipeline by:
1. Reading an Alpha 0.5.3 ADT file
2. Converting it to LK format using managed builders
3. (Future) Converting back to Alpha using existing converters
4. Comparing the result with the original

## Quick Start

### Run Round-Trip Test

```bash
cd WoWRollback
dotnet run --project WoWRollback.AdtConverter roundtrip-test --alpha-adt "path/to/alpha.adt" --out "roundtrip_output"
```

### Example

```bash
# Test a single Alpha ADT tile
dotnet run --project WoWRollback.AdtConverter roundtrip-test \
  --alpha-adt "C:\WoW_Alpha\World\Maps\Azeroth\Azeroth_32_49.adt" \
  --out "test_results"
```

## What It Does

### Step 1: Extract Alpha Data
- Reads the Alpha ADT file
- Extracts 256 MCNK chunks
- Parses raw chunk data:
  - MCVT (height data, 580 bytes)
  - MCNR (normal data, 448 bytes)
  - MCLY (layer table)
  - MCAL (alpha maps)
  - MCSH (shadow map)
  - MCRF (doodad/WMO references)

### Step 2: Build LK ADT
- Populates `LkAdtSource` with 256 `LkMcnkSource` objects
- Uses `LkAdtBuilder` to create a valid LK ADT
- Writes intermediate LK ADT to output directory

### Step 3: Validate (Future)
- Convert LK back to Alpha using `AlphaMcnkBuilder`
- Compare with original Alpha ADT
- Report differences

## Output

The test creates:
- `<original_name>_lk.adt` - Intermediate LK ADT file
- Console output with test results

### Success Output
```
Round-trip test: path/to/alpha.adt
Output: roundtrip_output
✓ Round-trip test PASSED
  Tiles processed: 256
  LK ADT created successfully at: roundtrip_output/alpha_lk.adt
```

### Failure Output
```
✗ Round-trip test FAILED
  Error: <error message>
  Bytes different: <count>
```

## Testing in WoW Client

1. Run the round-trip test to generate LK ADT
2. Copy the LK ADT to your LK client's map directory
3. Launch the client and navigate to the tile
4. Verify terrain looks correct

## Current Status

**Implemented:**
- ✅ Alpha data extraction (`AlphaDataExtractor`)
- ✅ LK ADT building (`LkAdtBuilder`, `LkMcnkBuilder`)
- ✅ CLI command (`roundtrip-test`)
- ✅ Validation framework (`RoundTripValidator`)

**TODO:**
- ⏸️ LK→Alpha conversion (use existing `AlphaMcnkBuilder`)
- ⏸️ Full round-trip comparison
- ⏸️ Detailed diff reporting

## Architecture

```
Alpha ADT File
      ↓
AlphaDataExtractor.ExtractFromAlphaAdt()
      ↓
LkAdtSource (256 LkMcnkSource)
      ↓
LkAdtBuilder.Build()
      ↓
LK ADT Bytes
      ↓
(Future: AlphaMcnkBuilder)
      ↓
Alpha ADT Bytes
      ↓
Compare with Original
```

## Components

### AlphaDataExtractor
**Location**: `WoWRollback.LkToAlphaModule/Services/AlphaDataExtractor.cs`

Extracts raw chunk data from Alpha ADT files:
- `ExtractFromAlphaAdt(string)` - Extracts all 256 MCNKs
- `ExtractFromAlphaMcnk(string, int, int)` - Extracts single MCNK

### RoundTripValidator
**Location**: `WoWRollback.LkToAlphaModule/Validators/RoundTripValidator.cs`

Orchestrates the round-trip test:
- `ValidateRoundTrip(string, string, options)` - Runs full test
- `CompareAlphaAdts(string, string)` - Byte-by-byte comparison

### CLI Command
**Location**: `WoWRollback.AdtConverter/Program.cs`

Command: `roundtrip-test`
- `--alpha-adt` (required) - Input Alpha ADT file
- `--out` (optional) - Output directory (default: "roundtrip_output")

## Known Limitations

1. **MCAL Parsing**: Currently assumes simple 4096-byte layers
   - Doesn't handle compressed alpha
   - Doesn't parse MCLY flags properly
   - May need refinement for complex texturing

2. **Incomplete Round-Trip**: Only goes Alpha→LK
   - Need to add LK→Alpha step
   - Need comparison logic

3. **No WDT Support**: Tests individual ADTs only
   - Can't test full map conversions yet

## Next Steps

1. Add LK→Alpha conversion step
2. Implement byte-by-byte comparison
3. Add detailed diff reporting (which chunks differ, by how much)
4. Add batch testing for multiple tiles
5. Add visual diff tools (height maps, texture maps)

## Troubleshooting

### "Alpha ADT not found"
- Check the file path is correct
- Use absolute paths to avoid confusion

### "Build failed"
- Ensure all projects build successfully
- Check for missing dependencies

### "Invalid data exception"
- Alpha ADT may be corrupted
- Check file format (should be 0.5.3 Alpha)
- Verify MCNK chunks are valid

## Testing Strategy

1. **Start Small**: Test a single simple tile (flat terrain, few textures)
2. **Validate Visually**: Load in client to verify terrain looks correct
3. **Test Complex**: Gradually test more complex tiles
4. **Automate**: Once working, batch test entire maps
