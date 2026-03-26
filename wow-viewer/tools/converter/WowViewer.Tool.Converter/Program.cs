using WowViewer.Core.IO;
using WowViewer.Core.PM4;
using WowViewer.Tools.Shared;

Console.WriteLine("WowViewer.Tool.Converter skeleton");
Console.WriteLine($"Owns families: {string.Join(", ", IoBoundaries.OwnedFamilies)}");
Console.WriteLine($"PM4 source-of-truth: runtime={Pm4Boundary.RuntimeReference}, seed={Pm4Boundary.LibrarySeed}");
Console.WriteLine($"Planned hosts: {ToolHosts.Planned.Length}");
