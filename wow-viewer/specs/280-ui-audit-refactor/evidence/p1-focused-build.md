# Focused viewer build
Command: dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore -v quiet
Exit status: 1
I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\Workbench\Services\FileOperations\TerrainFileOperationsService.cs(277,33): error CS0104: 'Image' is an ambiguous reference between 'System.Drawing.Image' and 'SixLabors.ImageSharp.Image' [I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\WoWViewer.csproj]
I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\Workbench\Services\FileOperations\TerrainFileOperationsService.cs(277,33): error CS0104: 'Image' is an ambiguous reference between 'System.Drawing.Image' and 'SixLabors.ImageSharp.Image' [I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\WoWViewer.csproj]
    91 Warning(s)
    1 Error(s)
Time Elapsed 00:00:07.77
# Rebuild after Image alias fix
Exit status: 1
I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\Workbench\Services\FileOperations\ModelWorldFileOperationsService.cs(95,42): error CS0103: The name 'WarcraftNetM2Adapter' does not exist in the current context [I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\WoWViewer.csproj]
I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\Workbench\Services\FileOperations\ModelWorldFileOperationsService.cs(95,42): error CS0103: The name 'WarcraftNetM2Adapter' does not exist in the current context [I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\WoWViewer.csproj]
    317 Warning(s)
    1 Error(s)
Time Elapsed 00:00:08.00
# Rebuild after adapter namespace fix
Exit status: 0
    317 Warning(s)
    0 Error(s)
Time Elapsed 00:00:08.76
# Final focused build
Exit status: 0
    317 Warning(s)
    0 Error(s)
Time Elapsed 00:00:06.76

Warning detail elided; compiler error lines, command, exit statuses and summaries retained. Final source build: exit 0, 317 warnings, 0 errors.
