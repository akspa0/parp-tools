using WowViewer.Core;
using WowViewer.Core.PM4;
using WowViewer.Core.Runtime;

Console.WriteLine("WowViewer.App skeleton");
Console.WriteLine($"Solution: {ProjectIdentity.SolutionName} {ProjectIdentity.PlannedVersion}");
Console.WriteLine($"PM4 runtime reference: {Pm4Boundary.RuntimeReference}");
Console.WriteLine($"Runtime service areas: {RuntimeBoundaries.All.Length}");
