
using System.Collections.Generic;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.Data;

namespace WoWToolbox.Core.v2.Services.PM4
{
public class Pm4ModelBuilder : IPm4ModelBuilder
{
    private readonly ICoordinateService _coordinateService;

    public Pm4ModelBuilder(ICoordinateService coordinateService)
    {
        _coordinateService = coordinateService;
    }

    public CompleteWMOModel Build(PM4File pm4File)
    {
        var combinedVertices = new List<System.Numerics.Vector3>();
        var combinedTriangleIndices = new List<int>();
        int totalVerticesOffset = 0;

        // 1. MSVT render vertices
        if (pm4File.MSVT != null && pm4File.MSVT.Vertices.Any())
        {
            var vertices = pm4File.MSVT.Vertices.Select(v => _coordinateService.FromMsvtVertexSimple(v)).ToList();
            combinedVertices.AddRange(vertices);

            if (pm4File.MSVI != null && pm4File.MSVI.Indices.Any())
            {
                combinedTriangleIndices.AddRange(pm4File.MSVI.Indices.Select(i => (int)(i + totalVerticesOffset)));
            }
            totalVerticesOffset = combinedVertices.Count;
        }

        // 2. MSPV geometry vertices
        if (pm4File.MSPV != null && pm4File.MSPV.Vertices.Any())
        {
            var vertices = pm4File.MSPV.Vertices.Select(v => _coordinateService.FromMspvVertex(v)).ToList();
            combinedVertices.AddRange(vertices);

            if (pm4File.MSPI != null && pm4File.MSPI.Indices.Any())
            {
                combinedTriangleIndices.AddRange(pm4File.MSPI.Indices.Select(i => (int)(i + totalVerticesOffset)));
            }
            totalVerticesOffset = combinedVertices.Count;
        }

        // 3. MSCN collision boundaries
        if (pm4File.MSCN != null && pm4File.MSCN.ExteriorVertices.Any())
        {
            var vertices = pm4File.MSCN.ExteriorVertices.Select(v => _coordinateService.FromMscnVertex(v)).ToList();
            combinedVertices.AddRange(vertices);
            // MSCN chunks do not have their own index buffers (MSCI), they define collision hulls.
        }

        var model = new CompleteWMOModel();
        model.Vertices.AddRange(combinedVertices);
        model.TriangleIndices.AddRange(combinedTriangleIndices);

        if (model.VertexCount > 0 && model.FaceCount > 0)
        {
            model.AddNormals(_coordinateService.ComputeVertexNormals(model.Vertices, model.TriangleIndices).ToArray());
        }

        return model;
    }
}
}

