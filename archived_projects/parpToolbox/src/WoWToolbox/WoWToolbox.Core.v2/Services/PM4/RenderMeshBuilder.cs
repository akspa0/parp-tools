using System.Collections.Generic;
using System.Numerics;
using WoWToolbox.Core.v2.Models.PM4;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public class RenderMeshBuilder : IRenderMeshBuilder
    {
        private readonly ICoordinateService _coordinateService;

        public RenderMeshBuilder(ICoordinateService coordinateService)
        {
            _coordinateService = coordinateService;
        }

        public RenderMesh BuildFromPm4(PM4File pm4File)
        {
            var mesh = new RenderMesh();

            if (pm4File?.MSVT?.Vertices == null || pm4File.MSVI?.Indices == null)
            {
                return mesh; // Return an empty mesh if essential data is missing.
            }

            // 1. Transform and add vertices
            foreach (var vertex in pm4File.MSVT.Vertices)
            {
                mesh.Vertices.Add(_coordinateService.FromMsvtVertexSimple(vertex));
            }

            // 2. Add faces
            mesh.Faces.AddRange(pm4File.MSVI.Indices.Select(i => (int)i));

            // 3. Compute normals
            mesh.Normals = _coordinateService.ComputeVertexNormals(mesh.Vertices, mesh.Faces);

            return mesh;
        }
    }
}
