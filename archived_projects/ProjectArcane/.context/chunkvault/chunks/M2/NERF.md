# NERF Chunk (Normal-based Edge Rendering Factor)

## Overview
The NERF (Normal-based Edge Rendering Factor) chunk was introduced in Shadowlands (9.0.1.33978) and contains data for calculating alpha attenuation based on distance. The chunk provides coefficients used in a ratio calculation with the squared length of the instanced model radius, affecting the overall alpha transparency of the model instance.

## Structure
```cpp
struct NERF_Chunk {
  C2Vector coefs;  // Two coefficients used in alpha attenuation calculation
}
```

## Fields
- **coefs**: A C2Vector containing two coefficients (x and y) used to calculate distance-based alpha attenuation

## Dependencies
- Requires the MD21 chunk for basic model data
- Interacts with the model instance's radius for distance calculations

## Usage
The NERF chunk is used to:
- Calculate alpha attenuation for model instances based on distance
- Create smooth fade-out effects as models move away from the viewer
- Implement more sophisticated level-of-detail transitions
- Control visibility of model instances in a distance-dependent manner

## Legacy Support
- Not present in pre-Shadowlands M2 files
- Earlier versions likely used simpler alpha fading techniques or shader-based approaches

## Implementation Notes
- The alpha value is calculated using the formula: `(coefs.x - squaredRadius) / (coefs.x - coefs.y)`
- This value is used as a multiplier for the model instance's alpha
- The calculation provides a smooth transition as the model moves toward or away from the view position
- Implementation requires tracking the squared distance of the model instance from the viewer
- The effect is applied globally to the model instance rather than to specific components

## Version History
- Introduced in Shadowlands (9.0.1.33978)
- Represents an enhancement to the M2 format's distance-based rendering controls
- Part of ongoing improvements to model rendering efficiency and visual quality in WoW 