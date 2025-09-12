using System;
using System.Collections.Generic;
using System.IO;

namespace GillijimProject.Next.Core.PM4;

public static class Pm4ObjectAssemblerService
{
    /// <summary>
    /// Exports per-object (buildings) using the ParpToolbox hierarchical assembler.
    /// Returns the count of exported objects with geometry.
    /// </summary>
    public static int ExportPerObject(string inputPath, bool includeAdjacent, string outputDirectory, string baseFileName)
    {
        Directory.CreateDirectory(outputDirectory);
        var parpScene = Pm4Loader.LoadParp(inputPath, includeAdjacent, applyMscnRemap: true);
        var assembler = new ParpToolbox.Services.PM4.Pm4HierarchicalAssembler();
        var summary = assembler.ExportBuildingsUsingHierarchicalAssembly(parpScene, outputDirectory, baseFileName);
        return summary.SuccessfulExports;
    }

    /// <summary>
    /// Assemble buildings and write per-object OBJs to <paramref name="objectsOutDir"/>.
    /// Returns the list of assembled buildings (geometry in world coords after PM4 transform).
    /// </summary>
    public static IReadOnlyList<ParpToolbox.Services.PM4.Pm4HierarchicalAssembler.CompleteBuilding> GetBuildings(
        string inputPath,
        bool includeAdjacent,
        string objectsOutDir,
        string baseFileName)
    {
        Directory.CreateDirectory(objectsOutDir);
        var parpScene = Pm4Loader.LoadParp(inputPath, includeAdjacent, applyMscnRemap: true);
        var assembler = new ParpToolbox.Services.PM4.Pm4HierarchicalAssembler();
        var summary = assembler.ExportBuildingsUsingHierarchicalAssembly(parpScene, objectsOutDir, baseFileName);
        return summary.Buildings;
    }
}
