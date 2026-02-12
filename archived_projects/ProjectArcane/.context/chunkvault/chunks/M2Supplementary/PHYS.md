# M2.PHYS File Format

## Overview
The .phys files are chunked supplementary files for M2 models that were introduced in Mists of Pandaria (expansion level 5) and are used by Blizzard's Domino physics engine. The M2 model requests a .phys file to be loaded by having GlobalModelFlags & 0x20 set. These files define physics properties and behavior for M2 models, including bodies, shapes, and joints.

## File Structure
The .phys file consists of a main PHYS chunk followed by an unordered sequence of unique chunks. The structure includes bodies (representing physical objects), shapes (defining collision geometry), and joints (connecting bodies together).

## Versions
The .phys format has gone through several versions:
- Version 0: Since MoP (5.0.1.15464)
- Version 1: Since Legion (7.0.1.20773)
- Version 2: Since Legion (7.0.1.20979)
- Version 2*: Since Legion (7.0.1.21063) - Changed semantics and chunk names
- Version 3: Since Legion (7.0.3.21287)
- Version 4: Since Legion (7.0.3.21846)
- Version 5: Since Legion (7.3.0.24500)
- Version 6: Since Shadowlands

Each version maintains backward compatibility by filling missing fields with default values.

## Chunks

### PHYS (Physics Definition)
```cpp
struct PHYS_Chunk {
    short version;    // Version number from 0 to 6
};
```

#### Fields
- **version**: Indicates the version of the .phys file format

### PHYT (Physics Type, version 1+)
```cpp
struct PHYT_Chunk {
    uint32_t phyt;    // Default: 0
};
```

#### Fields
- **phyt**: Physics type identifier

### BODY/BDY2/BDY3/BDY4 (Bodies)
Different versions of body chunks with evolving structure:

#### Version 0-1 (BODY)
```cpp
struct BODY_Chunk {
    struct {
        unsigned short type;           // 0 -> 1, 1 -> 0 = dm_dynamicBody, * -> 2
        char padding_a[2];
        vec3 position;
        unsigned short modelBoneIndex;
        char padding_b[2];
        int shapes_base;               // Starting at shapes[shapes_base]
        int shapes_count;              // Number of shapes in this body
    } bodies[];
};
```

#### Version 2 (BDY2)
Adds an additional float field with default value 1.0.

#### Version 3-4 (BDY3/BDY4)
More comprehensive versions with additional parameters for physics behavior:
```cpp
struct BDY3_BDY4_Chunk {
    struct {
        unsigned short type;           // Body type
        unsigned short boneIndex;      // Model bone index
        vec3 position;                 // Position vector
        unsigned short shapeIndex;     // Shape index
        char padding_b[2];
        int shapesCount;               // Number of shapes
        float unk0;                    // Unknown float
        float mass;                    // Default 1.0
        float drag;                    // Default 0
        float weight;                  // Default 0 (affects kinematic behavior)
        float damping;                 // Default 0.89999998
        // Version 4 adds additional padding
    } bodies[];
};
```

### SHAP/SHP2 (Shapes)
Defines collision shapes for physics bodies:

```cpp
struct SHAP_Chunk {
    struct {
        short shapeType;              // 0=box, 1=capsule, 2=sphere, 3=polytope(v3+)
        short shapeIndex;             // Index into corresponding shape chunk
        char unk[4];
        float friction;               // Friction coefficient
        float restitution;            // Bounce factor
        float density;                // Mass per volume unit
        // Version 2+ adds additional fields
    } shapes[];
};
```

### Shape Types
#### BOXS (Box Shapes)
```cpp
struct BOXS_Chunk {
    struct {
        mat3x4 a;                    // Orientation matrix
        vec3 c;                      // Center point
    } boxShapes[];
};
```

#### CAPS (Capsule Shapes)
```cpp
struct CAPS_Chunk {
    struct {
        vec3 localPosition1;         // First end point
        vec3 localPosition2;         // Second end point
        float radius;                // Capsule radius
    } capsuleShapes[];
};
```

#### SPHS (Sphere Shapes)
```cpp
struct SPHS_Chunk {
    struct {
        vec3 localPosition;          // Center position
        float radius;                // Sphere radius
    } sphereShapes[];
};
```

#### PLYT (Polytope Shapes, version 3+)
Complex mesh-based collision shapes with vertices and node structure:
```cpp
struct PLYT_Chunk {
    uint32_t count;
    
    struct {
        uint32_t vertexCount;           // Number of vertices (mostly 8)
        char unk_04[0x4];
        uint64_t RUNTIME_08_ptr_data_0; // Runtime pointer
        uint32_t count_10;              // Mostly 6
        char unk_14[0x4];
        uint64_t RUNTIME_18_ptr_data_1; // Runtime pointer
        uint64_t RUNTIME_20_ptr_data_2; // Runtime pointer
        uint32_t nodeCount;             // Number of nodes (mostly 24)
        char unk_2C[0x4];
        uint64_t RUNTIME_30_ptr_data_3; // Runtime pointer
        float unk_38[6];                // Unknown float array
    } header[count];
    
    // Variable-sized data for each polytope
    struct {
        vec3 vertices[header[i].vertexCount]; // Vertices forming the convex hull
        // Additional node and connection data
    } data[count];
};
```

### JOIN (Joints)
Connects bodies together and defines their relationship:
```cpp
struct JOIN_Chunk {
    struct {
        unsigned int bodyAIdx;         // First body index
        unsigned int bodyBIdx;         // Second body index
        char unk[4];
        short jointType;               // 0=spherical, 1=shoulder, 2=weld,
                                       // 3=revolute(v2+), 4=prismatic(v2+), 5=distance(v2+)
        short jointId;                 // Index into corresponding joint chunk
    } joints[];
};
```

### Joint Types
#### WELJ/WLJ2/WLJ3 (Weld Joints)
Rigidly connects two bodies:
```cpp
struct WELJ_Chunk {
    struct {
        mat3x4 frameA;                // Frame for body A
        mat3x4 frameB;                // Frame for body B
        float angularFrequencyHz;      // Oscillation frequency
        float angularDampingRatio;     // Dampening factor
        // Version 2+ adds additional fields
    } weldJoints[];
};
```

#### SPHJ (Spherical Joints)
Ball-and-socket type joint:
```cpp
struct SPHJ_Chunk {
    struct {
        vec3 anchorA;                 // Anchor point on first body
        vec3 anchorB;                 // Anchor point on second body
        float frictionTorque;         // Rotational friction
    } sphericalJointEntries[];
};
```

#### SHOJ/SHJ2 (Shoulder Joints)
Specialized joint for shoulder-like movement:
```cpp
struct SHOJ_Chunk {
    struct {
        mat3x4 frameA;                // Frame for body A
        mat3x4 frameB;                // Frame for body B
        float lowerTwistAngle;        // Minimum twist angle
        float upperTwistAngle;        // Maximum twist angle
        float coneAngle;              // Movement cone angle
        // Version 2+ adds additional fields
    } shoulderJoints[];
};
```

#### PRSJ/PRS2 (Prismatic Joints, version 2+)
Sliding joints with linear motion:
```cpp
struct PRSJ_Chunk {
    struct {
        mat3x4 frameA;                // Frame for body A
        mat3x4 frameB;                // Frame for body B
        float lowerLimit;             // Lower translation limit
        float upperLimit;             // Upper translation limit
        float unknownValue;           // Unknown parameter
        float maxMotorForce;          // Maximum force for motor
        float unknownValue2;          // Unknown parameter
        uint32_t motorMode;           // Motor behavior mode
        // Version PRS2 adds additional fields
    } prismaticJoints[];
};
```

#### REVJ/REV2 (Revolute Joints, version 2+)
Hinge-like joints with rotational motion:
```cpp
struct REVJ_Chunk {
    struct {
        mat3x4 frameA;                // Frame for body A
        mat3x4 frameB;                // Frame for body B
        float lowerAngle;             // Lower angle limit
        float upperAngle;             // Upper angle limit
        float maxMotorTorque;         // Maximum torque for motor
        uint32_t motorMode;           // 1=position, 2=velocity
        // Version REV2 adds additional fields
    } revoluteJoints[];
};
```

#### DSTJ (Distance Joints, version 2+)
Maintains a distance between bodies:
```cpp
struct DSTJ_Chunk {
    struct {
        vec3 localAnchorA;            // Anchor point on first body
        vec3 localAnchorB;            // Anchor point on second body
        float some_distance_factor;    // Distance parameter
    } distanceJoints[];
};
```

### PHYV (Physics Variables, version 1+)
Defines tuning parameters for physics behavior:
```cpp
struct PHYV_Chunk {
    struct {
        float parameters[6];          // Six float parameters
    } phyv[];
};
```

When this chunk is present, it sets default values for physics parameters, which are then overwritten with the values from the chunk.

## Dependencies
- Requires the parent M2 model
- References bones in the M2 model via indices
- Interacts with the Domino physics engine

## Usage
The .phys files are used to:
- Define physical properties of models
- Enable collision detection between models
- Create articulated physics simulations
- Define how model parts interact with each other and the environment
- Support cloth, soft-body, and rigid-body physics

## Implementation Notes
- The physics system uses a hierarchical approach with bodies, shapes, and joints
- Bodies are connected to bones in the M2 model
- Shapes define the collision geometry for bodies
- Joints define how bodies connect and interact with each other
- Implementation should handle the version differences and provide appropriate defaults
- Backward compatibility is maintained through versions with increasing complexity

## Version History
- Introduced in Mists of Pandaria (5.0.1.15464)
- Expanded significantly in Legion (7.0.x) with multiple version updates
- Further refined in Battle for Azeroth and Shadowlands
- Each version adds more sophisticated physics capabilities 