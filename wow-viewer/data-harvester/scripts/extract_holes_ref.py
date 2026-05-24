"""Minimal C# tool to extract hole masks from WDT in MPQ archives."""
# This is a reference - the actual extraction is done by the C# harvester.
# The Python patch script reads the output of this extraction.

# Build and run:
# dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -- extract-holes --client-root "path" --output "holes.json"

# The output JSON has format:
# {
#   "build": {
#     "map_tileX_tileY": [16x16 array of uint16 hole masks]
#   }
# }
