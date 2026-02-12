# WoW Alpha 0.5.3 Detail Doodads

## Overview

Detail doodads are small decorative elements like grass, flowers, and rocks that are scattered across terrain chunks to add visual detail. They are rendered with reduced draw distance and use instanced rendering for efficiency.

## Detail Doodad Instance

**Address:** [`CDetailDoodadInst::CDetailDoodadInst`](0x006a2580) (0x006a2580)

### Key Fields

```c
struct CDetailDoodadInst {
    TSLink<CDetailDoodadGeom> lameAssLink;  // Link to geometry
    CDetailDoodadGeom* geom[2];              // 2 geometry pointers
    CGxBuf* gxBuf[2];                       // 2 graphics buffers
};
```

### Detail Doodad Geometry

```c
struct CDetailDoodadGeom {
    C3Vector position;      // Position in world space
    C3Vector rotation;      // Rotation (Euler angles)
    C3Vector scale;         // Scale
    uint modelId;           // Model ID
    uint textureId;         // Texture ID
};
```

## Detail Doodad Creation

**Address:** [`CMapChunk::CreateDetailDoodads`](0x006a6cf0) (0x006a6cf0)

### Purpose

Create detail doodads for a chunk.

### Algorithm

```c
void CreateDetailDoodads(CMapChunk* chunk) {
    // Check if chunk is within detail doodad distance
    float distance = Distance(chunk->position, camPos);
    if (distance > CWorld::detailDoodadDist) {
        return;
    }
    
    // Create detail doodad instance
    chunk->detailDoodadInst = new CDetailDoodadInst();
    
    // Generate random positions for doodads
    for (int i = 0; i < MAX_DETAIL_DOODADS; i++) {
        // Generate random position within chunk
        float x = RandomFloat(0.0f, 1.0f);
        float y = RandomFloat(0.0f, 1.0f);
        
        // Get terrain height at position
        float z = GetTerrainHeight(chunk->position.x + x * CHUNK_SIZE, 
                                   chunk->position.y + y * CHUNK_SIZE);
        
        // Create doodad geometry
        CDetailDoodadGeom* geom = new CDetailDoodadGeom();
        geom->position.x = chunk->position.x + x * CHUNK_SIZE;
        geom->position.y = chunk->position.y + y * CHUNK_SIZE;
        geom->position.z = z;
        geom->rotation.x = RandomFloat(0.0f, 360.0f);
        geom->rotation.y = RandomFloat(0.0f, 360.0f);
        geom->rotation.z = RandomFloat(0.0f, 360.0f);
        geom->scale.x = RandomFloat(0.5f, 1.5f);
        geom->scale.y = RandomFloat(0.5f, 1.5f);
        geom->scale.z = RandomFloat(0.5f, 1.5f);
        geom->modelId = RandomModelId();
        geom->textureId = RandomTextureId();
        
        // Add to detail doodad instance
        chunk->detailDoodadInst->geom[i] = geom;
    }
    
    // Create graphics buffers
    for (int i = 0; i < 2; i++) {
        chunk->detailDoodadInst->gxBuf[i] = CreateGraphicsBuffer(chunk->detailDoodadInst->geom[i]);
    }
}
```

## Detail Doodad Rendering

### Purpose

Render detail doodads for a chunk.

### Algorithm

```c
void RenderDetailDoodads(CMapChunk* chunk) {
    // Check if detail doodads exist
    if (chunk->detailDoodadInst == NULL) {
        return;
    }
    
    // Render each doodad
    for (int i = 0; i < 2; i++) {
        CDetailDoodadGeom* geom = chunk->detailDoodadInst->geom[i];
        if (geom == NULL) {
            continue;
        }
        
        // Set up world transform
        C44Matrix worldMatrix;
        worldMatrix = CreateTranslationMatrix(geom->position);
        worldMatrix = worldMatrix * CreateRotationMatrix(geom->rotation);
        worldMatrix = worldMatrix * CreateScaleMatrix(geom->scale);
        
        GxXformSet(GxXform_World, &worldMatrix);
        
        // Render doodad
        RenderModel(geom->modelId, geom->textureId);
    }
}
```

## Detail Doodad Density

### Console Commands

- [`ConsoleCommand_DetailDoodadAlpha`](0x00665ff0) (0x00665ff0) - Set detail doodad alpha
- [`ConsoleCommand_DetailDoodadTest`](0x00665fb0) (0x00665fb0) - Test detail doodads
- [`ConsoleCommand_ShowDetailDoodads`](0x00665770) (0x00665770) - Show/hide detail doodads

### Density Control

```c
// Detail doodad density
float detailDoodadDensity = 1.0f;  // 0.0f to 1.0f

// Detail doodad distance
float detailDoodadDist = 100.0f;  // Distance from camera to show detail doodads

// Maximum detail doodads per chunk
const int MAX_DETAIL_DOODADS = 64;
```

## Implementation Guidelines

### C# Detail Doodads

```csharp
public class DetailDoodadManager
{
    private class DetailDoodadInst
    {
        public DetailDoodadGeom[] Geoms { get; set; }  // 2 geometry pointers
        public GraphicsBuffer[] GxBufs { get; set; }  // 2 graphics buffers
    }
    
    private class DetailDoodadGeom
    {
        public C3Vector Position { get; set; }
        public C3Vector Rotation { get; set; }
        public C3Vector Scale { get; set; }
        public uint ModelId { get; set; }
        public uint TextureId { get; set; }
    }
    
    private const float DETAIL_DOODAD_DISTANCE = 100.0f;
    private const int MAX_DETAIL_DOODADS = 64;
    
    public void CreateDetailDoodads(TerrainChunk chunk, C3Vector cameraPosition)
    {
        // Check if chunk is within detail doodad distance
        float distance = Vector3.Distance(chunk.Position, cameraPosition);
        if (distance > DETAIL_DOODAD_DISTANCE)
        {
            return;
        }
        
        // Create detail doodad instance
        DetailDoodadInst inst = new DetailDoodadInst();
        inst.Geoms = new DetailDoodadGeom[MAX_DETAIL_DOODADS];
        inst.GxBufs = new GraphicsBuffer[MAX_DETAIL_DOODADS];
        
        // Generate random positions for doodads
        Random random = new Random(chunk.Seed);
        
        for (int i = 0; i < MAX_DETAIL_DOODADS; i++)
        {
            // Generate random position within chunk
            float x = (float)random.NextDouble();
            float y = (float)random.NextDouble();
            
            // Get terrain height at position
            float z = GetTerrainHeight(chunk.Position.X + x * CHUNK_SIZE, 
                                       chunk.Position.Y + y * CHUNK_SIZE);
            
            // Create doodad geometry
            inst.Geoms[i] = new DetailDoodadGeom
            {
                Position = new C3Vector(
                    chunk.Position.X + x * CHUNK_SIZE,
                    chunk.Position.Y + y * CHUNK_SIZE,
                    z),
                Rotation = new C3Vector(
                    (float)random.NextDouble() * 360.0f,
                    (float)random.NextDouble() * 360.0f,
                    (float)random.NextDouble() * 360.0f),
                Scale = new C3Vector(
                    0.5f + (float)random.NextDouble(),
                    0.5f + (float)random.NextDouble(),
                    0.5f + (float)random.NextDouble()),
                ModelId = RandomModelId(random),
                TextureId = RandomTextureId(random)
            };
            
            // Create graphics buffer
            inst.GxBufs[i] = CreateGraphicsBuffer(inst.Geoms[i]);
        }
        
        chunk.DetailDoodadInst = inst;
    }
    
    public void RenderDetailDoodads(TerrainChunk chunk)
    {
        // Check if detail doodads exist
        if (chunk.DetailDoodadInst == null)
        {
            return;
        }
        
        // Render each doodad
        for (int i = 0; i < MAX_DETAIL_DOODADS; i++)
        {
            DetailDoodadGeom geom = chunk.DetailDoodadInst.Geoms[i];
            if (geom == null)
            {
                continue;
            }
            
            // Set up world transform
            Matrix4x4 worldMatrix = Matrix4x4.CreateTranslation(geom.Position);
            worldMatrix *= Matrix4x4.CreateRotationX(geom.Rotation.X * (float)Math.PI / 180.0f);
            worldMatrix *= Matrix4x4.CreateRotationY(geom.Rotation.Y * (float)Math.PI / 180.0f);
            worldMatrix *= Matrix4x4.CreateRotationZ(geom.Rotation.Z * (float)Math.PI / 180.0f);
            worldMatrix *= Matrix4x4.CreateScale(geom.Scale);
            
            // Set world transform
            GL.UniformMatrix4(worldMatrixLocation, false, ref worldMatrix);
            
            // Render doodad
            RenderModel(geom.ModelId, geom.TextureId);
        }
    }
    
    private uint RandomModelId(Random random)
    {
        // Return random model ID
        return (uint)random.Next(0, 100);
    }
    
    private uint RandomTextureId(Random random)
    {
        // Return random texture ID
        return (uint)random.Next(0, 100);
    }
    
    private GraphicsBuffer CreateGraphicsBuffer(DetailDoodadGeom geom)
    {
        // Create graphics buffer for doodad
        return new GraphicsBuffer();
    }
    
    private void RenderModel(uint modelId, uint textureId)
    {
        // Render model with texture
        // This would involve binding the model and texture and drawing
    }
}
```

## References

- [`CDetailDoodadInst::CDetailDoodadInst`](0x006a2580) (0x006a2580) - Detail doodad instance constructor
- [`CMapChunk::CreateDetailDoodads`](0x006a6cf0) (0x006a6cf0) - Create detail doodads for chunk
- [`ConsoleCommand_DetailDoodadAlpha`](0x00665ff0) (0x00665ff0) - Set detail doodad alpha
- [`ConsoleCommand_DetailDoodadTest`](0x00665fb0) (0x00665fb0) - Test detail doodads
- [`ConsoleCommand_ShowDetailDoodads`](0x00665770) (0x00665770) - Show/hide detail doodads
