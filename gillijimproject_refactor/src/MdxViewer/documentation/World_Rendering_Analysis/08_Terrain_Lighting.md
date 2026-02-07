# WoW Alpha 0.5.3 Terrain Lighting

## Overview

The WoW Alpha 0.5.3 terrain lighting system uses a day/night cycle that dynamically calculates sun position, light color, ambient light, and fog color based on game time.

## Day/Night Cycle

**Address:** [`CWorld::UpdateDayNightCycle`](0x0066a5c0) (0x0066a5c0)

### Purpose

Update all lighting parameters based on the current game time.

### Algorithm

```c
void UpdateDayNightCycle(float gameTime) {
    // Calculate sun position
    C3Vector sunPos = CalculateSunPosition(gameTime);
    
    // Update light direction
    CWorld::lightDirection = Normalize(sunPos);
    
    // Calculate light color based on time
    C3Color lightColor = CalculateLightColor(gameTime);
    CWorld::lightColor = lightColor;
    
    // Calculate ambient light based on time
    C3Color ambientLight = CalculateAmbientLight(gameTime);
    CWorld::ambientLight = ambientLight;
    
    // Calculate fog color based on time
    C3Color fogColor = CalculateFogColor(gameTime);
    CWorld::fogColor = fogColor;
    
    // Recalculate lighting for all chunks
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            CalcLighting(&CMap::mapArea->chunks[i][j]);
        }
    }
}
```

## Sun Position Calculation

**Address:** [`CWorld::CalculateSunPosition`](0x0066a5d0) (0x0066a5d0)

### Purpose

Calculate the sun's position based on game time.

### Algorithm

```c
C3Vector CalculateSunPosition(float gameTime) {
    // Game time is in range [0, 1) representing 24 hours
    // 0.0 = midnight, 0.5 = noon, 1.0 = midnight
    
    // Calculate sun angle (0 at midnight, PI at noon)
    float sunAngle = gameTime * 2.0f * PI;
    
    // Calculate sun position on unit sphere
    C3Vector sunPos;
    sunPos.x = cos(sunAngle);
    sunPos.y = sin(sunAngle);
    sunPos.z = sin(sunAngle + PI / 2.0f);
    
    return sunPos;
}
```

## Light Color Calculation

**Address:** [`CWorld::CalculateLightColor`](0x0066a5e0) (0x0066a5e0)

### Purpose

Calculate the directional light color based on time of day.

### Algorithm

```c
C3Color CalculateLightColor(float gameTime) {
    C3Color lightColor;
    
    if (gameTime < 0.25f || gameTime > 0.75f) {
        // Night time
        lightColor.r = 0.2f;
        lightColor.g = 0.2f;
        lightColor.b = 0.3f;
    } else if (gameTime < 0.35f || gameTime > 0.65f) {
        // Dawn/Dusk
        lightColor.r = 0.8f;
        lightColor.g = 0.5f;
        lightColor.b = 0.3f;
    } else {
        // Day time
        lightColor.r = 1.0f;
        lightColor.g = 1.0f;
        lightColor.b = 0.9f;
    }
    
    return lightColor;
}
```

## Ambient Light Calculation

**Address:** [`CWorld::CalculateAmbientLight`](0x0066a5f0) (0x0066a5f0)

### Purpose

Calculate ambient lighting intensity based on time of day.

### Algorithm

```c
C3Color CalculateAmbientLight(float gameTime) {
    C3Color ambientLight;
    
    if (gameTime < 0.25f || gameTime > 0.75f) {
        // Night time
        ambientLight.r = 0.1f;
        ambientLight.g = 0.1f;
        ambientLight.b = 0.15f;
    } else if (gameTime < 0.35f || gameTime > 0.65f) {
        // Dawn/Dusk
        ambientLight.r = 0.4f;
        ambientLight.g = 0.35f;
        ambientLight.b = 0.35f;
    } else {
        // Day time
        ambientLight.r = 0.6f;
        ambientLight.g = 0.6f;
        ambientLight.b = 0.65f;
    }
    
    return ambientLight;
}
```

## Fog Color Calculation

**Address:** [`CWorld::CalculateFogColor`](0x0066a600) (0x0066a600)

### Purpose

Calculate fog color based on time of day.

### Algorithm

```c
C3Color CalculateFogColor(float gameTime) {
    C3Color fogColor;
    
    if (gameTime < 0.25f || gameTime > 0.75f) {
        // Night time
        fogColor.r = 0.1f;
        fogColor.g = 0.1f;
        fogColor.b = 0.15f;
    } else if (gameTime < 0.35f || gameTime > 0.65f) {
        // Dawn/Dusk
        fogColor.r = 0.6f;
        fogColor.g = 0.4f;
        fogColor.b = 0.3f;
    } else {
        // Day time
        fogColor.r = 0.7f;
        fogColor.g = 0.75f;
        fogColor.b = 0.85f;
    }
    
    return fogColor;
}
```

## Chunk Lighting Calculation

**Address:** [`CMapChunk::CalcLighting`](0x006a6d30) (0x006a6d30)

### Purpose

Calculate lighting for all vertices in a terrain chunk.

### Algorithm

```c
void CalcLighting(CMapChunk* chunk) {
    C3Color ambient = CWorld::ambientLight;
    C3Color diffuse = CWorld::lightColor;
    C3Vector lightDir = CWorld::lightDirection;
    
    // Process each vertex
    for (int i = 0; i < chunk->vertexCount; i++) {
        C3Vector normal = chunk->normals[i];
        
        // Calculate diffuse lighting (Lambertian)
        float diffuseFactor = MAX(0.0f, DotProduct(normal, lightDir));
        
        // Combine ambient and diffuse
        C3Color vertexColor;
        vertexColor.r = ambient.r + diffuse.r * diffuseFactor;
        vertexColor.g = ambient.g + diffuse.g * diffuseFactor;
        vertexColor.b = ambient.b + diffuse.b * diffuseFactor;
        
        // Clamp to valid range
        vertexColor.r = MIN(1.0f, vertexColor.r);
        vertexColor.g = MIN(1.0f, vertexColor.g);
        vertexColor.b = MIN(1.0f, vertexColor.b);
        
        // Apply lighting to vertex
        chunk->vertexColors[i] = vertexColor;
    }
}
```

## Lighting Properties

### Light Direction

```c
struct C3Vector {
    float x;  // X component
    float y;  // Y component
    float z;  // Z component
};
```

### Light Color

```c
struct C3Color {
    float r;  // Red component (0-1)
    float g;  // Green component (0-1)
    float b;  // Blue component (0-1)
};
```

## Implementation Guidelines

### C# Terrain Lighting

```csharp
public class TerrainLighting
{
    private Vector3 lightDirection;
    private Color lightColor;
    private Color ambientLight;
    private Color fogColor;
    
    public void UpdateDayNightCycle(float gameTime)
    {
        // Calculate sun position
        Vector3 sunPos = CalculateSunPosition(gameTime);
        lightDirection = Vector3.Normalize(sunPos);
        
        // Update lighting parameters
        lightColor = CalculateLightColor(gameTime);
        ambientLight = CalculateAmbientLight(gameTime);
        fogColor = CalculateFogColor(gameTime);
    }
    
    private Vector3 CalculateSunPosition(float gameTime)
    {
        float sunAngle = gameTime * 2.0f * (float)Math.PI;
        
        return new Vector3(
            (float)Math.Cos(sunAngle),
            (float)Math.Sin(sunAngle),
            (float)Math.Sin(sunAngle + Math.PI / 2.0)
        );
    }
    
    private Color CalculateLightColor(float gameTime)
    {
        if (gameTime < 0.25f || gameTime > 0.75f)
        {
            return new Color(0.2f, 0.2f, 0.3f);
        }
        else if (gameTime < 0.35f || gameTime > 0.65f)
        {
            return new Color(0.8f, 0.5f, 0.3f);
        }
        else
        {
            return new Color(1.0f, 1.0f, 0.9f);
        }
    }
    
    private Color CalculateAmbientLight(float gameTime)
    {
        if (gameTime < 0.25f || gameTime > 0.75f)
        {
            return new Color(0.1f, 0.1f, 0.15f);
        }
        else if (gameTime < 0.35f || gameTime > 0.65f)
        {
            return new Color(0.4f, 0.35f, 0.35f);
        }
        else
        {
            return new Color(0.6f, 0.6f, 0.65f);
        }
    }
    
    private Color CalculateFogColor(float gameTime)
    {
        if (gameTime < 0.25f || gameTime > 0.75f)
        {
            return new Color(0.1f, 0.1f, 0.15f);
        }
        else if (gameTime < 0.35f || gameTime > 0.65f)
        {
            return new Color(0.6f, 0.4f, 0.3f);
        }
        else
        {
            return new Color(0.7f, 0.75f, 0.85f);
        }
    }
    
    public Color CalculateVertexLighting(Vector3 normal)
    {
        // Calculate diffuse factor
        float diffuseFactor = Math.Max(0.0f, Vector3.Dot(normal, lightDirection));
        
        // Combine ambient and diffuse
        float r = ambientLight.R + lightColor.R * diffuseFactor;
        float g = ambientLight.G + lightColor.G * diffuseFactor;
        float b = ambientLight.B + lightColor.B * diffuseFactor;
        
        // Clamp to valid range
        r = Math.Min(1.0f, r);
        g = Math.Min(1.0f, g);
        b = Math.Min(1.0f, b);
        
        return new Color(r, g, b);
    }
}
```

## References

- [`CMapChunk::CalcLighting`](0x006a6d30) (0x006a6d30) - Calculate lighting for chunk
- [`CWorld::UpdateDayNightCycle`](0x0066a5c0) (0x0066a5c0) - Update day/night cycle
- [`CWorld::CalculateSunPosition`](0x0066a5d0) (0x0066a5d0) - Calculate sun position
- [`CWorld::CalculateLightColor`](0x0066a5e0) (0x0066a5e0) - Calculate light color
- [`CWorld::CalculateAmbientLight`](0x0066a5f0) (0x0066a5f0) - Calculate ambient light
- [`CWorld::CalculateFogColor`](0x0066a600) (0x0066a600) - Calculate fog color
