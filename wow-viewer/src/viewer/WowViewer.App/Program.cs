using WowViewer.Core;
using WowViewer.Core.PM4;
using WowViewer.Core.Runtime;

Console.WriteLine("WowViewer.App skeleton");
Console.WriteLine($"Solution: {ProjectIdentity.SolutionName} {ProjectIdentity.PlannedVersion}");
Console.WriteLine($"PM4 canonical owner: {Pm4Boundary.CanonicalOwner}");
Console.WriteLine($"PM4 legacy reference: {Pm4Boundary.LegacyReference}");
Console.WriteLine($"Runtime service areas: {RuntimeBoundaries.All.Length}");
