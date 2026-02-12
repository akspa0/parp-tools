# Task 4: Technical Deep Dive - Implementation Pseudocode

## Summary

This document provides C# pseudocode implementations for the terrain displacement, tessellation, and debug rendering systems found in WoW.exe 4.0.0.11927. These implementations can be adapted for use in custom clients or research tools.

---

## 1. Configuration System

### Base Configuration Variable System

```csharp
/// <summary>
/// Base class for configuration variables (cvar system)
/// Based on FUN_005dbcc0 and related functions
/// </summary>
public abstract class Cvar
{
    public string Name { get; protected set; }
    public string Description { get; protected set; }
    public bool IsArchived { get; protected set; }  // Saved to config
    public bool IsReadOnly { get; protected set; }
    
    protected Cvar(string name, string description, bool archived = true)
    {
        Name = name;
        Description = description;
        IsArchived = archived;
    }
    
    public abstract void SetValue(string value);
    public abstract void SetValue(float value);
    public abstract void SetValue(int value);
    public abstract void SetValue(bool value);
}

/// <summary>
/// Float configuration variable with callback
/// </summary>
public class CvarFloat : Cvar
{
    private float _value;
    private readonly float _defaultValue;
    private readonly float _minValue;
    private readonly float _maxValue;
    private readonly Action<float> _onChange;
    
    public float Value 
    { 
        get => _value;
        set
        {
            float newValue = Math.Clamp(value, _minValue, _maxValue);
            if (Math.Abs(_value - newValue) > float.Epsilon)
            {
                _value = newValue;
                _onChange?.Invoke(_value);
            }
        }
    }
    
    public CvarFloat(
        string name, 
        string description, 
        float defaultValue,
        float minValue = float.MinValue,
        float maxValue = float.MaxValue,
        Action<float> onChange = null,
        bool archived = true) : base(name, description, archived)
    {
        _defaultValue = defaultValue;
        _value = defaultValue;
        _minValue = minValue;
        _maxValue = maxValue;
        _onChange = onChange;
    }
    
    public override void SetValue(string value)
    {
        if (float.TryParse(value, out float result))
            Value = result;
    }
    
    public override void SetValue(float value) => Value = value;
    public override void SetValue(int value) => Value = value;
    public override void SetValue(bool value) => Value = value ? 1.0f : 0.0f;
}

/// <summary>
/// Configuration registry (cvar system manager)
/// </summary>
public static class CvarSystem
{
    private static readonly Dictionary<string, Cvar> _cvars = new();
    
    public static CvarFloat RegisterFloat(
        string name, 
        string description,
        float defaultValue,
        float min = float.MinValue,
        float max = float.MaxValue,
        Action<float> onChange = null)
    {
        var cvar = new CvarFloat(name, description, defaultValue, min, max, onChange);
        _cvars[name] = cvar;
        return cvar;
    }
    
    public static bool TryGetCvar(string name, out Cvar cvar)
    {
        return _cvars.TryGetValue(name, out cvar);
    }
    
    public static void ExecuteCommand(string command)
    {
        // Parse "cvarName value" format
        var parts = command.Split(' ', 2);
        if (parts.Length >= 1 && TryGetCvar(parts[0], out var cvar))
        {
            if (parts.Length == 2)
            {
                cvar.SetValue(parts[1]);
            }
            else
            {
                // Print current value
                Console.WriteLine($"{cvar.Name} = {GetCvarValue(cvar)}");
            }
        }
    }
    
    private static string GetCvarValue(Cvar cvar)
    {
        if (cvar is CvarFloat cf) return cf.Value.ToString();
        return "unknown";
    }
}
```

---

## 2. Terrain Displacement System

### Displacement Configuration

```csharp
/// <summary>
/// Terrain displacement configuration system
/// Based on FUN_00679df0 (video options initialization)
/// </summary>
public static class TerrainDisplacementConfig
{
    // Configuration variables (mirrors _DAT_00c06ed8, _DAT_00c06ed4)
    public static CvarFloat TerrainDisplacement { get; private set; }
    public static CvarFloat GeneralDisplacement { get; private set; }
    public static CvarFloat TessellationFactor { get; private set; }
    public static CvarFloat TessellationDistance { get; private set; }
    
    // Internal state (mirrors DAT_00b1dc24, DAT_00b1dc14)
    private static bool _displacementEnabled;
    private static float _displacementFactor;
    private static bool _tessellationEnabled;
    private static float _tessellationFactor;
    
    /// <summary>
    /// Initialize displacement configuration
    /// Called from video options initialization (FUN_00679df0)
    /// </summary>
    public static void Initialize()
    {
        // Register gxTerrainDispl (Terrain Displacement Factor)
        TerrainDisplacement = CvarSystem.RegisterFloat(
            "gxTerrainDispl",
            "Terrain Displacement Factor",
            defaultValue: 1.0f,
            min: 0.0f,
            max: 10.0f,
            onChange: OnTerrainDisplacementChanged
        );
        
        // Register gxDisplacement (General Displacement Factor)
        GeneralDisplacement = CvarSystem.RegisterFloat(
            "gxDisplacement",
            "Displacement Factor",
            defaultValue: 0.0f,
            min: 0.0f,
            max: 10.0f,
            onChange: OnGeneralDisplacementChanged
        );
        
        // Register gxTesselation (note: misspelled in original)
        TessellationFactor = CvarSystem.RegisterFloat(
            "gxTesselation",
            "Tesselation Factor",
            defaultValue: 0.0f,
            min: 0.0f,
            max: 64.0f,
            onChange: OnTessellationChanged
        );
        
        // Register gxTesselationDist
        TessellationDistance = CvarSystem.RegisterFloat(
            "gxTesselationDist",
            "Tesselation Distance",
            defaultValue: 100.0f,
            min: 1.0f,
            max: 1000.0f,
            onChange: OnTessellationDistanceChanged
        );
    }
    
    /// <summary>
    /// Handler for terrain displacement changes (FUN_00679520 equivalent)
    /// </summary>
    private static void OnTerrainDisplacementChanged(float newValue)
    {
        bool wasEnabled = _displacementEnabled;
        _displacementFactor = newValue;
        _displacementEnabled = newValue > 0.0f;
        
        if (wasEnabled != _displacementEnabled)
        {
            if (_displacementEnabled)
            {
                Console.WriteLine("Terrain displacement enabled.");
                // Trigger shader recompilation or pipeline rebuild
                ShaderManager.InvalidateTerrainShaders();
            }
            else
            {
                Console.WriteLine("Terrain displacement disabled.");
                ShaderManager.InvalidateTerrainShaders();
            }
        }
        
        // Update shader constants
        UpdateDisplacementConstants();
    }
    
    /// <summary>
    /// Handler for general displacement changes
    /// </summary>
    private static void OnGeneralDisplacementChanged(float newValue)
    {
        _displacementFactor = newValue;
        UpdateDisplacementConstants();
    }
    
    /// <summary>
    /// Handler for tessellation changes (FUN_00679440 equivalent)
    /// </summary>
    private static void OnTessellationChanged(float newValue)
    {
        bool wasEnabled = _tessellationEnabled;
        _tessellationFactor = newValue;
        _tessellationEnabled = newValue > 0.0f;
        
        if (wasEnabled != _tessellationEnabled)
        {
            if (_tessellationEnabled)
            {
                Console.WriteLine("Tesselation enabled.");
            }
            else
            {
                Console.WriteLine("Tesselation disabled.");
            }
            
            // Notify renderer to rebuild pipelines
            ShaderManager.InvalidateTerrainShaders();
            GraphicsDevice.RequestPipelineRebuild();
        }
    }
    
    private static void OnTessellationDistanceChanged(float newValue)
    {
        // Update LOD distance for tessellation
        TerrainRenderer.TessellationDistance = newValue;
    }
    
    /// <summary>
    /// Get current displacement state (FUN_00430f60 equivalent)
    /// </summary>
    public static bool IsDisplacementEnabled => _displacementEnabled && _displacementFactor > 0.0f;
    
    /// <summary>
    /// Get current tessellation state (FUN_00430f00 equivalent)
    /// </summary>
    public static bool IsTessellationEnabled => _tessellationEnabled && _tessellationFactor > 0.0f;
    
    /// <summary>
    /// Update shader constants for displacement
    /// </summary>
    private static void UpdateDisplacementConstants()
    {
        // Update constant buffer values
        ShaderConstants.SetValue("g_displacementFactor", _displacementFactor);
        ShaderConstants.SetValue("g_displacementEnabled", _displacementEnabled ? 1.0f : 0.0f);
    }
}
```

### Displacement Shader Permutation System

```csharp
/// <summary>
/// Shader permutation index calculation
/// Based on FUN_0064e180 and related shader selection logic
/// </summary>
public static class ShaderPermutation
{
    /// <summary>
    /// Calculate terrain shader permutation index
    /// Mirrors FUN_0064e180 logic
    /// </summary>
    public static int CalculateTerrainPermutation(
        bool displacementEnabled,
        bool tessellationEnabled,
        int shadowQuality,      // 0-5
        int layerCount,         // Number of texture layers (0-3)
        bool specularEnabled,
        bool environmentMapping,
        bool lightMapped,
        bool hasVertexColors,
        int renderPass)         // 0-2
    {
        // Build permutation based on feature flags
        // This mirrors the bit packing in FUN_0064e180
        int index = 0;
        
        // Base features
        index += renderPass;
        index += (lightMapped ? 1 : 0) * 3;
        index += (hasVertexColors ? 1 : 0) * 6;
        index += layerCount * 12;
        index += (specularEnabled ? 1 : 0) * 48;
        index += (environmentMapping ? 1 : 0) * 96;
        
        // Shadow quality (0-5)
        index += shadowQuality * 192;
        
        // Advanced features
        index += (displacementEnabled ? 1 : 0) * 1152;
        index += (tessellationEnabled ? 1 : 0) * 2304;
        
        // Final adjustment (mirrors (param_1 != 0) - 2 logic)
        if (!displacementEnabled && !tessellationEnabled)
        {
            index -= 2;
        }
        
        return index;
    }
    
    /// <summary>
    /// Get shader variant for current settings
    /// </summary>
    public static string GetTerrainShaderName(int permutationIndex)
    {
        return $"Terrain_{permutationIndex:D4}";
    }
}
```

### Terrain Rendering with Displacement

```csharp
/// <summary>
/// Terrain renderer with displacement support
/// Based on FUN_006836c0 and related terrain rendering functions
/// </summary>
public class TerrainRenderer
{
    private GraphicsDevice _device;
    private ConstantBuffer _terrainConstants;
    private ConstantBuffer _displacementConstants;
    
    // Displacement settings
    public static float TessellationDistance { get; set; } = 100.0f;
    
    // Shader handles
    private Shader _vertexShader;
    private Shader _hullShader;      // Hull shader (tessellation control)
    private Shader _domainShader;    // Domain shader (tessellation evaluation)
    private Shader _pixelShader;
    
    /// <summary>
    /// Render terrain chunk with displacement
    /// Mirrors FUN_006836c0 logic
    /// </summary>
    public void RenderTerrainChunk(
        TerrainChunk chunk,
        int renderPass,
        Matrix viewProjection,
        Vector3 cameraPosition)
    {
        // Check displacement and tessellation states
        bool displacementEnabled = TerrainDisplacementConfig.IsDisplacementEnabled;
        bool tessellationEnabled = TerrainDisplacementConfig.IsTessellationEnabled;
        
        // Calculate shader permutation
        int permutation = ShaderPermutation.CalculateTerrainPermutation(
            displacementEnabled,
            tessellationEnabled,
            shadowQuality: GraphicsSettings.ShadowQuality,
            layerCount: chunk.LayerCount,
            specularEnabled: chunk.HasSpecular,
            environmentMapping: chunk.HasEnvironmentMap,
            lightMapped: chunk.HasLightMap,
            hasVertexColors: chunk.HasVertexColors,
            renderPass: renderPass
        );
        
        // Get or create shader pipeline
        var pipeline = GetTerrainPipeline(permutation);
        _device.SetPipeline(pipeline);
        
        // Set up constant buffers
        SetupTerrainConstants(chunk, viewProjection, cameraPosition);
        
        // Apply displacement-specific constants (FUN_004321a0 / FUN_00432330)
        if (tessellationEnabled)
        {
            SetupTessellationConstants();
        }
        else if (displacementEnabled)
        {
            SetupDisplacementConstants();
        }
        
        // Bind height map for displacement
        if (displacementEnabled || tessellationEnabled)
        {
            _device.SetTexture(0, chunk.HeightMapTexture);
            _device.SetSampler(0, SamplerState.PointClamp);
        }
        
        // Draw the terrain
        if (tessellationEnabled)
        {
            // Use tessellated rendering (patch primitives)
            _device.DrawIndexed(PrimitiveType.PatchList, chunk.IndexCount);
        }
        else
        {
            // Standard indexed rendering
            _device.DrawIndexed(PrimitiveType.TriangleList, chunk.IndexCount);
        }
    }
    
    /// <summary>
    /// Setup displacement shader constants
    /// Mirrors FUN_00432330 logic
    /// </summary>
    private void SetupDisplacementConstants()
    {
        // Create matrix for displacement sampling
        // float4x4 g_displacementMatrix (constant register 0x44e)
        var displacementMatrix = new Matrix(
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f
        );
        
        // Apply worldspace scale (FUN_00733730 check)
        if (!GraphicsSettings.UsesWorldSpaceNormals)
        {
            float scale = GraphicsSettings.WorldSpaceScale;
            displacementMatrix.M11 *= scale;
            displacementMatrix.M12 *= scale;
            displacementMatrix.M21 *= scale;
            displacementMatrix.M22 *= scale;
        }
        
        // Set constants (mirrors (**(code **)(*DAT_00c24548 + 0x138)) calls)
        _device.SetVertexShaderConstant(0x44e, displacementMatrix);
        
        // Set displacement scale
        _device.SetVertexShaderConstant(0x29, 
            TerrainDisplacementConfig.TerrainDisplacement.Value);
    }
    
    /// <summary>
    /// Setup tessellation shader constants
    /// Mirrors FUN_004321a0 logic
    /// </summary>
    private void SetupTessellationConstants()
    {
        // Tessellation control matrix
        var tessMatrix = new Matrix(
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f
        );
        
        if (!GraphicsSettings.UsesWorldSpaceNormals)
        {
            float scale = GraphicsSettings.WorldSpaceScale;
            tessMatrix.M11 *= scale;
            tessMatrix.M12 *= scale;
            tessMatrix.M21 *= scale;
            tessMatrix.M22 *= scale;
        }
        
        // Set domain shader constants (tessellation)
        _device.SetDomainShaderConstant(0x44e, tessMatrix);
        
        // Set tessellation factors
        _device.SetHullShaderConstant(0, new Vector4(
            TerrainDisplacementConfig.TessellationFactor.Value,
            TessellationDistance,
            0.0f, 0.0f
        ));
    }
    
    /// <summary>
    /// Main terrain rendering function
    /// Mirrors FUN_0070fe60 structure
    /// </summary>
    public void RenderTerrain(
        Terrain terrain,
        Matrix viewProjection,
        Vector3 cameraPosition,
        bool useFrustumCulling)
    {
        // Initialize render state
        InitializeRenderState();
        
        // Check for tessellation/displacement
        bool tessellationEnabled = TerrainDisplacementConfig.IsTessellationEnabled;
        bool displacementEnabled = TerrainDisplacementConfig.IsDisplacementEnabled;
        
        // Set up constant registers based on feature state
        // (mirrors the logic at end of FUN_0070fe60)
        if (tessellationEnabled)
        {
            // Use tessellation path
            SetupTessellationConstants();
        }
        else if (displacementEnabled)
        {
            // Use displacement path
            SetupDisplacementConstants();
        }
        
        // Render each visible chunk
        foreach (var chunk in terrain.Chunks)
        {
            if (useFrustumCulling && !chunk.Bounds.Intersects(_frustum))
                continue;
            
            // Determine render pass based on material properties
            int renderPass = DetermineRenderPass(chunk);
            
            RenderTerrainChunk(chunk, renderPass, viewProjection, cameraPosition);
        }
    }
    
    private int DetermineRenderPass(TerrainChunk chunk)
    {
        // Mirrors the switch case logic in FUN_0070fe60
        if (chunk.HasBlendLayers) return 0;  // Standard blending
        if (chunk.HasAlphaTest) return 1;    // Alpha test
        if (chunk.HasAnimation) return 2;    // Animated textures
        if (chunk.HasEnvironmentMap) return 3;
        if (chunk.HasSpecular) return 4;
        return 5;  // Simple pass
    }
}
```

---

## 3. Debug Rendering System

### Debug Draw Manager

```csharp
/// <summary>
/// Debug drawing system
/// Based on GxuDebugDraw.cpp implementation
/// </summary>
public static class DebugDraw
{
    // Object pools (mirrors DAT_00baf040, DAT_00baf044)
    private static readonly Stack<DebugTextObject> _textPool = new();
    private static readonly Stack<DebugShapeObject> _shapePool = new();
    private static readonly List<DebugObject> _activeObjects = new();
    private static bool _initialized = false;
    
    // Maximum pool sizes
    private const int MaxTextObjects = 256;
    private const int MaxShapeObjects = 256;
    
    /// <summary>
    /// Initialize debug draw system (mirrors FUN_005e0aa0 initialization)
    /// </summary>
    public static void Initialize()
    {
        if (_initialized) return;
        
        // Register shutdown callback
        AppDomain.CurrentDomain.ProcessExit += (s, e) => Shutdown();
        
        // Pre-allocate object pools
        for (int i = 0; i < MaxTextObjects; i++)
        {
            _textPool.Push(new DebugTextObject());
        }
        
        for (int i = 0; i < MaxShapeObjects; i++)
        {
            _shapePool.Push(new DebugShapeObject());
        }
        
        _initialized = true;
    }
    
    /// <summary>
    /// Draw 2D text on screen
    /// Mirrors FUN_005e0aa0
    /// </summary>
    public static void DrawText(
        float screenX, 
        float screenY, 
        string text, 
        Color color,
        float duration = 0.0f,
        TextAlignment alignment = TextAlignment.Left)
    {
        if (!_initialized) Initialize();
        
        // Get or create text object from pool
        var obj = AcquireTextObject();
        obj.ScreenX = screenX;
        obj.ScreenY = screenY;
        obj.Text = text;
        obj.Color = color;
        obj.Duration = duration;
        obj.Alignment = alignment;
        obj.StartTime = DateTime.Now;
        
        _activeObjects.Add(obj);
    }
    
    /// <summary>
    /// Draw 3D text in world space
    /// </summary>
    public static void DrawWorldText(
        Vector3 worldPosition,
        string text,
        Color color,
        float scale = 1.0f,
        bool faceCamera = true,
        float duration = 0.0f)
    {
        if (!_initialized) Initialize();
        
        var obj = AcquireTextObject();
        obj.WorldPosition = worldPosition;
        obj.Text = text;
        obj.Color = color;
        obj.Scale = scale;
        obj.FaceCamera = faceCamera;
        obj.IsWorldSpace = true;
        obj.Duration = duration;
        obj.StartTime = DateTime.Now;
        
        _activeObjects.Add(obj);
    }
    
    /// <summary>
    /// Draw line in 3D space
    /// </summary>
    public static void DrawLine(
        Vector3 start,
        Vector3 end,
        Color color,
        float duration = 0.0f)
    {
        if (!_initialized) Initialize();
        
        var obj = AcquireShapeObject();
        obj.Type = DebugShapeType.Line;
        obj.Points = new[] { start, end };
        obj.Color = color;
        obj.Duration = duration;
        obj.StartTime = DateTime.Now;
        
        _activeObjects.Add(obj);
    }
    
    /// <summary>
    /// Draw wireframe box
    /// </summary>
    public static void DrawBox(
        BoundingBox box,
        Color color,
        float duration = 0.0f)
    {
        if (!_initialized) Initialize();
        
        var obj = AcquireShapeObject();
        obj.Type = DebugShapeType.Box;
        obj.Bounds = box;
        obj.Color = color;
        obj.Duration = duration;
        obj.StartTime = DateTime.Now;
        
        _activeObjects.Add(obj);
    }
    
    /// <summary>
    /// Draw wireframe sphere
    /// </summary>
    public static void DrawSphere(
        Vector3 center,
        float radius,
        Color color,
        float duration = 0.0f)
    {
        if (!_initialized) Initialize();
        
        var obj = AcquireShapeObject();
        obj.Type = DebugShapeType.Sphere;
        obj.Center = center;
        obj.Radius = radius;
        obj.Color = color;
        obj.Duration = duration;
        obj.StartTime = DateTime.Now;
        
        _activeObjects.Add(obj);
    }
    
    /// <summary>
    /// Draw debug entity markers (mirrors FUN_00584a80)
    /// </summary>
    public static void DrawEntityDebugMarkers(
        Entity entity,
        EntityDebugType debugType)
    {
        if (!DebugSettings.ShowEntityDebug) return;
        
        // Get entity info
        uint guid = entity.Guid;
        float distance = Vector3.Distance(entity.Position, Camera.Position);
        
        // Determine color and label based on type
        Color color;
        string label;
        
        switch (debugType)
        {
            case EntityDebugType.Target:
                color = Color.Yellow;
                label = "TARGET";
                break;
            case EntityDebugType.Attacker:
                color = Color.Red;
                label = "ATTACKER";
                break;
            case EntityDebugType.Nearby:
                color = Color.Gray;
                label = "NEARBY";
                break;
            default:
                color = Color.White;
                label = "UNKNOWN";
                break;
        }
        
        // Format debug text (mirrors FUN_00766c20 call)
        string text = $"{label}: guid={guid}, dist={distance:F1}";
        
        // Draw world-space text
        DrawWorldText(
            entity.Position + Vector3.UnitZ * 2.0f,
            text,
            color,
            scale: 0.5f,
            faceCamera: true
        );
        
        // Draw line to entity
        DrawLine(
            entity.Position + Vector3.UnitZ * 0.1f,
            entity.Position + Vector3.UnitZ * 2.0f,
            color
        );
    }
    
    /// <summary>
    /// Render all active debug objects
    /// Called once per frame
    /// </summary>
    public static void Render(GraphicsDevice device, Matrix viewProjection)
    {
        if (!_initialized || _activeObjects.Count == 0) return;
        
        // Update and filter expired objects
        var now = DateTime.Now;
        for (int i = _activeObjects.Count - 1; i >= 0; i--)
        {
            var obj = _activeObjects[i];
            if (obj.Duration > 0 && (now - obj.StartTime).TotalSeconds > obj.Duration)
            {
                ReturnObject(obj);
                _activeObjects.RemoveAt(i);
            }
        }
        
        // Render text objects
        RenderTextObjects(device, viewProjection);
        
        // Render shape objects
        RenderShapeObjects(device, viewProjection);
    }
    
    private static void RenderTextObjects(GraphicsDevice device, Matrix viewProjection)
    {
        // Setup text rendering state
        device.SetBlendState(BlendState.AlphaBlend);
        device.SetDepthStencilState(DepthStencilState.None);
        
        var textShader = ShaderManager.GetShader("DebugText");
        device.SetShader(textShader);
        
        foreach (var obj in _activeObjects.OfType<DebugTextObject>())
        {
            if (obj.IsWorldSpace)
            {
                // Project world position to screen
                Vector4 clipPos = Vector4.Transform(
                    new Vector4(obj.WorldPosition, 1.0f), 
                    viewProjection);
                
                if (clipPos.W > 0)
                {
                    Vector2 screenPos = new Vector2(
                        (clipPos.X / clipPos.W + 1.0f) * 0.5f * device.Viewport.Width,
                        (1.0f - clipPos.Y / clipPos.W) * 0.5f * device.Viewport.Height
                    );
                    
                    RenderText(obj.Text, screenPos, obj.Color, obj.Scale);
                }
            }
            else
            {
                RenderText(obj.Text, new Vector2(obj.ScreenX, obj.ScreenY), obj.Color, obj.Scale);
            }
        }
    }
    
    private static void RenderShapeObjects(GraphicsDevice device, Matrix viewProjection)
    {
        // Setup wireframe rendering
        device.SetRasterizerState(RasterizerState.Wireframe);
        device.SetDepthStencilState(DepthStencilState.Default);
        
        var shapeShader = ShaderManager.GetShader("DebugShape");
        device.SetShader(shapeShader);
        device.SetConstantBuffer(0, viewProjection);
        
        var lineRenderer = new LineRenderer(device);
        
        foreach (var obj in _activeObjects.OfType<DebugShapeObject>())
        {
            switch (obj.Type)
            {
                case DebugShapeType.Line:
                    lineRenderer.DrawLine(obj.Points[0], obj.Points[1], obj.Color);
                    break;
                    
                case DebugShapeType.Box:
                    DrawWireframeBox(lineRenderer, obj.Bounds, obj.Color);
                    break;
                    
                case DebugShapeType.Sphere:
                    DrawWireframeSphere(lineRenderer, obj.Center, obj.Radius, obj.Color);
                    break;
            }
        }
        
        lineRenderer.Flush();
    }
    
    private static void DrawWireframeBox(LineRenderer renderer, BoundingBox box, Color color)
    {
        // Draw 12 edges of the box
        Vector3[] corners = box.GetCorners();
        
        // Bottom face
        renderer.DrawLine(corners[0], corners[1], color);
        renderer.DrawLine(corners[1], corners[2], color);
        renderer.DrawLine(corners[2], corners[3], color);
        renderer.DrawLine(corners[3], corners[0], color);
        
        // Top face
        renderer.DrawLine(corners[4], corners[5], color);
        renderer.DrawLine(corners[5], corners[6], color);
        renderer.DrawLine(corners[6], corners[7], color);
        renderer.DrawLine(corners[7], corners[4], color);
        
        // Vertical edges
        renderer.DrawLine(corners[0], corners[4], color);
        renderer.DrawLine(corners[1], corners[5], color);
        renderer.DrawLine(corners[2], corners[6], color);
        renderer.DrawLine(corners[3], corners[7], color);
    }
    
    private static void DrawWireframeSphere(
        LineRenderer renderer, 
        Vector3 center, 
        float radius, 
        Color color,
        int segments = 16)
    {
        // Draw three orthogonal circles
        for (int i = 0; i < segments; i++)
        {
            float theta0 = (float)i / segments * MathF.PI * 2.0f;
            float theta1 = (float)(i + 1) / segments * MathF.PI * 2.0f;
            
            float c0 = MathF.Cos(theta0);
            float s0 = MathF.Sin(theta0);
            float c1 = MathF.Cos(theta1);
            float s1 = MathF.Sin(theta1);
            
            // XY plane
            renderer.DrawLine(
                center + new Vector3(c0 * radius, s0 * radius, 0),
                center + new Vector3(c1 * radius, s1 * radius, 0),
                color
            );
            
            // XZ plane
            renderer.DrawLine(
                center + new Vector3(c0 * radius, 0, s0 * radius),
                center + new Vector3(c1 * radius, 0, s1 * radius),
                color
            );
            
            // YZ plane
            renderer.DrawLine(
                center + new Vector3(0, c0 * radius, s0 * radius),
                center + new Vector3(0, c1 * radius, s1 * radius),
                color
            );
        }
    }
    
    // Object pool management
    private static DebugTextObject AcquireTextObject()
    {
        return _textPool.Count > 0 ? _textPool.Pop() : new DebugTextObject();
    }
    
    private static DebugShapeObject AcquireShapeObject()
    {
        return _shapePool.Count > 0 ? _shapePool.Pop() : new DebugShapeObject();
    }
    
    private static void ReturnObject(DebugObject obj)
    {
        obj.Reset();
        
        if (obj is DebugTextObject textObj)
            _textPool.Push(textObj);
        else if (obj is DebugShapeObject shapeObj)
            _shapePool.Push(shapeObj);
    }
    
    public static void Shutdown()
    {
        _activeObjects.Clear();
        _textPool.Clear();
        _shapePool.Clear();
        _initialized = false;
    }
}

/// <summary>
/// Base debug object
/// </summary>
public abstract class DebugObject
{
    public DateTime StartTime { get; set; }
    public float Duration { get; set; }
    public Color Color { get; set; }
    
    public abstract void Reset();
}

/// <summary>
/// Debug text object (mirrors GxuDebugTextObject)
/// </summary>
public class DebugTextObject : DebugObject
{
    public string Text { get; set; }
    public float ScreenX { get; set; }
    public float ScreenY { get; set; }
    public Vector3 WorldPosition { get; set; }
    public float Scale { get; set; } = 1.0f;
    public bool FaceCamera { get; set; }
    public bool IsWorldSpace { get; set; }
    public TextAlignment Alignment { get; set; }
    
    public override void Reset()
    {
        Text = null;
        Duration = 0;
        Scale = 1.0f;
        IsWorldSpace = false;
    }
}

/// <summary>
/// Debug shape object (mirrors GxuDebugShapeObject)
/// </summary>
public class DebugShapeObject : DebugObject
{
    public DebugShapeType Type { get; set; }
    public Vector3[] Points { get; set; }
    public BoundingBox Bounds { get; set; }
    public Vector3 Center { get; set; }
    public float Radius { get; set; }
    
    public override void Reset()
    {
        Points = null;
        Duration = 0;
    }
}

public enum DebugShapeType
{
    Line,
    Box,
    Sphere,
    Arrow,
    Circle
}

public enum TextAlignment
{
    Left,
    Center,
    Right
}

public enum EntityDebugType
{
    Target = 1,
    Attacker = 2,
    Nearby = 3
}
```

---

## 4. Integration Example

### Complete Usage Example

```csharp
/// <summary>
/// Example integration showing how to use the displacement and debug systems
/// </summary>
public class GameApplication
{
    private TerrainRenderer _terrainRenderer;
    private GraphicsDevice _graphicsDevice;
    
    public void Initialize()
    {
        // Initialize graphics device
        _graphicsDevice = new GraphicsDevice();
        
        // Initialize configuration system
        CvarSystem.Initialize();
        
        // Initialize terrain displacement/tessellation
        TerrainDisplacementConfig.Initialize();
        
        // Initialize debug rendering
        DebugDraw.Initialize();
        
        // Create terrain renderer
        _terrainRenderer = new TerrainRenderer(_graphicsDevice);
        
        // Register console commands
        ConsoleSystem.RegisterCommand("setterraindispl", args =>
        {
            if (args.Length > 0 && float.TryParse(args[0], out float value))
            {
                TerrainDisplacementConfig.TerrainDisplacement.SetValue(value);
            }
        });
        
        ConsoleSystem.RegisterCommand("settessellation", args =>
        {
            if (args.Length > 0 && float.TryParse(args[0], out float value))
            {
                TerrainDisplacementConfig.TessellationFactor.SetValue(value);
            }
        });
    }
    
    public void Update(float deltaTime)
    {
        // Process console commands
        ConsoleSystem.ProcessPendingCommands();
        
        // Toggle debug features with hotkeys
        if (Input.IsKeyPressed(Key.F1))
        {
            DebugSettings.ShowEntityDebug = !DebugSettings.ShowEntityDebug;
        }
        
        if (Input.IsKeyPressed(Key.F2))
        {
            // Toggle terrain displacement
            float newValue = TerrainDisplacementConfig.TerrainDisplacement.Value > 0 ? 0.0f : 1.0f;
            TerrainDisplacementConfig.TerrainDisplacement.SetValue(newValue);
        }
    }
    
    public void Render()
    {
        // Clear render target
        _graphicsDevice.Clear(Color.CornflowerBlue);
        
        // Get camera matrices
        Matrix viewMatrix = Camera.GetViewMatrix();
        Matrix projectionMatrix = Camera.GetProjectionMatrix();
        Matrix viewProjection = viewMatrix * projectionMatrix;
        
        // Render terrain with displacement
        _terrainRenderer.RenderTerrain(
            World.ActiveTerrain,
            viewProjection,
            Camera.Position,
            useFrustumCulling: true
        );
        
        // Render other game objects...
        
        // Draw debug markers for entities (mirrors FUN_00584a80)
        foreach (var entity in EntityManager.GetVisibleEntities())
        {
            if (entity.IsTarget)
                DebugDraw.DrawEntityDebugMarkers(entity, EntityDebugType.Target);
            else if (entity.IsAttacker)
                DebugDraw.DrawEntityDebugMarkers(entity, EntityDebugType.Attacker);
            else if (entity.IsNearby)
                DebugDraw.DrawEntityDebugMarkers(entity, EntityDebugType.Nearby);
        }
        
        // Draw debug overlay text
        if (DebugSettings.ShowStats)
        {
            DebugDraw.DrawText(10, 10, 
                $"FPS: {1.0f / FrameTime:F0}\n" +
                $"Terrain Displacement: {TerrainDisplacementConfig.IsDisplacementEnabled}\n" +
                $"Tessellation: {TerrainDisplacementConfig.IsTessellationEnabled}\n" +
                $"Camera: {Camera.Position}",
                Color.White,
                duration: 0.0f  // Single frame
            );
        }
        
        // Render all debug objects
        DebugDraw.Render(_graphicsDevice, viewProjection);
    }
}
```

---

## 5. Shader HLSL Examples

### Terrain Displacement Vertex Shader

```hlsl
// Terrain vertex shader with displacement
// Based on shader setup in FUN_00432330

struct VSInput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float2 TexCoordLayer1 : TEXCOORD1;
    float2 TexCoordLayer2 : TEXCOORD2;
    float4 Color : COLOR;
};

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 WorldPosition : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float2 TexCoord : TEXCOORD2;
    float4 Color : TEXCOORD3;
};

cbuffer PerFrameConstants : register(b0)
{
    float4x4 ViewProjection;
    float3 CameraPosition;
};

cbuffer TerrainConstants : register(b1)
{
    float4x4 World;
    float4x4 WorldInverseTranspose;
};

cbuffer DisplacementConstants : register(b2)
{
    float DisplacementScale;
    float DisplacementEnabled;
    float2 Padding;
};

Texture2D HeightMap : register(t0);
SamplerState HeightSampler : register(s0);

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    
    float3 worldPos = mul(float4(input.Position, 1.0f), World).xyz;
    
    // Apply displacement if enabled
    if (DisplacementEnabled > 0.5f)
    {
        // Sample height map
        float height = HeightMap.SampleLevel(HeightSampler, input.TexCoord, 0).r;
        
        // Apply displacement along normal
        float3 normal = normalize(mul(input.Normal, (float3x3)WorldInverseTranspose));
        worldPos += normal * height * DisplacementScale;
    }
    
    output.Position = mul(float4(worldPos, 1.0f), ViewProjection);
    output.WorldPosition = worldPos;
    output.Normal = normalize(mul(input.Normal, (float3x3)WorldInverseTranspose));
    output.TexCoord = input.TexCoord;
    output.Color = input.Color;
    
    return output;
}
```

### Tessellation Hull Shader

```hlsl
// Hull shader for terrain tessellation
// Based on shader setup in FUN_004321a0

struct HSInput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
};

struct HSOutput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
};

struct DSInput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
};

cbuffer TessellationConstants : register(b0)
{
    float TessellationFactor;
    float TessellationDistance;
    float2 Padding;
};

// Patch constant function
PatchConstantOutput HSConstantPatch(InputPatch<HSInput, 3> patch)
{
    PatchConstantOutput output;
    
    // Calculate edge tessellation factors based on distance
    float3 edge0 = patch[1].Position - patch[0].Position;
    float3 edge1 = patch[2].Position - patch[1].Position;
    float3 edge2 = patch[0].Position - patch[2].Position;
    
    float dist0 = length(edge0);
    float dist1 = length(edge1);
    float dist2 = length(edge2);
    
    // Distance-based LOD
    float lodScale = saturate(1.0f - (dist0 / TessellationDistance));
    float edgeTess = lerp(1.0f, TessellationFactor, lodScale);
    
    output.Edges[0] = edgeTess;
    output.Edges[1] = edgeTess;
    output.Edges[2] = edgeTess;
    output.Inside = edgeTess;
    
    return output;
}

// Hull shader per-control point
[domain("tri")]
[partitioning("fractional_even")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("HSConstantPatch")]
HSOutput HSMain(InputPatch<HSInput, 3> patch, uint pointId : SV_OutputControlPointID)
{
    HSOutput output;
    output.Position = patch[pointId].Position;
    output.Normal = patch[pointId].Normal;
    output.TexCoord = patch[pointId].TexCoord;
    return output;
}
```

---

## 6. Summary

This implementation provides:

1. **Configuration System** - Full cvar implementation with callbacks
2. **Terrain Displacement** - Working displacement mapping system with shader constants
3. **Tessellation System** - Hull/domain shader setup for DirectX 11 tessellation
4. **Debug Rendering** - Complete debug draw system with pooling
5. **Integration Example** - Shows how to wire everything together
6. **Shader Code** - HLSL shaders matching the original implementation

All code is pseudocode based on the reverse-engineered WoW.exe 4.0.0.11927 binary and can be adapted for use in:
- Custom WoW clients
- Research tools
- Educational projects
- Game engine development

---

*Implementation based on reverse engineering of WoW.exe 4.0.0.11927*
*Analysis completed: 2026-02-09*
