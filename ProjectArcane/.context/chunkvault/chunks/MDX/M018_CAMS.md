# CAMS - MDX Cameras Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The CAMS (Cameras) chunk defines camera definitions used in cinematics, game cutscenes, and model previews. Cameras define viewpoints, movement paths, and field of view settings. They provide directed viewing angles for showcasing the model and are particularly important for in-game cinematics and model viewers. Cameras can be animated to create smooth transitions between different viewpoints.

## Structure

```csharp
public struct CAMS
{
    /// <summary>
    /// Array of camera definitions
    /// </summary>
    // MDLCAMERA cameras[numCameras] follows
}

public struct MDLCAMERA
{
    /// <summary>
    /// Inclusively specifies the type of camera
    /// </summary>
    public uint inclusiveType;
    
    /// <summary>
    /// Position of the camera
    /// </summary>
    public Vector3 position;
    
    /// <summary>
    /// Field of view in radians
    /// </summary>
    public float fieldOfView;
    
    /// <summary>
    /// Far clip distance
    /// </summary>
    public float farClippingPlane;
    
    /// <summary>
    /// Near clip distance
    /// </summary>
    public float nearClippingPlane;
    
    /// <summary>
    /// Target position the camera points at
    /// </summary>
    public Vector3 targetPosition;
    
    /// <summary>
    /// Name of the camera
    /// </summary>
    public string name;
    
    /// <summary>
    /// Animation data for the camera properties
    /// </summary>
    // MDLKEYTRACK animations follow
}
```

## Properties

### MDLCAMERA Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | inclusiveType | uint | Type of camera (see Camera Types) |
| 0x04 | position | Vector3 | Position of the camera in 3D space |
| 0x10 | fieldOfView | float | Field of view angle in radians |
| 0x14 | farClippingPlane | float | Far clipping plane distance |
| 0x18 | nearClippingPlane | float | Near clipping plane distance |
| 0x1C | targetPosition | Vector3 | Position the camera points at |
| 0x28 | name | string | Camera name, null-terminated |
| varies | ... | ... | Animation tracks follow |

## Camera Types

| Value | Type | Description |
|-------|------|-------------|
| 0 | Target | Camera points at a specific target position |
| 1 | Portrait | Close-up camera for character portraits |
| 2 | Cinematic | Camera for cinematics and cutscenes |

## Animation Tracks
After the base properties, several animation tracks may follow:

- Position track (Vector3)
- Target position track (Vector3)
- Rotation track (Quaternion)
- Field of view track (float)

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Added support for quaternion rotation |

## Dependencies
- MDLKEYTRACK - Used for animation tracks within the structure
- SEQS - Camera animations are typically tied to specific sequences

## Implementation Notes
- Cameras define viewpoints for cinematic playback and model viewers
- The inclusiveType field determines how the camera behaves
- The target position is the point the camera looks at (for target cameras)
- Field of view controls the camera lens angle (wider = more visible)
- Clipping planes define the minimum and maximum visible distances
- Camera animations are typically used for cutscenes and special effects
- For proper rendering, the aspect ratio must be considered alongside the field of view
- Camera transitions should be smoothly interpolated for visual quality
- Position and target position tracks allow for camera movement and panning
- The coordinate system uses Y-up with Z forward, matching the model space
- Cameras are not rendered but used to determine the viewing frustum
- Multiple cameras can be defined and switched between during animations
- For correct rendering, both the camera position and orientation must be applied to the view matrix
- The field of view is vertical in radians; horizontal FOV depends on the aspect ratio

## Usage Context
Cameras in MDX models are used for:
- In-game cinematics and cutscenes
- Model viewer applications
- Character portrait rendering
- Spell and ability showcase views
- Location establishing shots
- Director-controlled viewing angles
- Smooth transitions between viewpoints
- Special camera effects (shake, zoom, pan)
- Showcase animations from specific angles
- Guided tours of complex models or scenes

## Implementation Example

```csharp
public class CAMSChunk : IMdxChunk
{
    public string ChunkId => "CAMS";
    
    public List<MdxCamera> Cameras { get; private set; } = new List<MdxCamera>();
    
    public void Parse(BinaryReader reader, long totalSize)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + totalSize;
        
        // Clear existing cameras
        Cameras.Clear();
        
        // Read cameras until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var camera = new MdxCamera();
            
            // Read basic camera properties
            camera.InclusiveType = reader.ReadUInt32();
            
            // Read position
            float x = reader.ReadSingle();
            float y = reader.ReadSingle();
            float z = reader.ReadSingle();
            camera.Position = new Vector3(x, y, z);
            
            // Read camera settings
            camera.FieldOfView = reader.ReadSingle();
            camera.FarClippingPlane = reader.ReadSingle();
            camera.NearClippingPlane = reader.ReadSingle();
            
            // Read target position
            x = reader.ReadSingle();
            y = reader.ReadSingle();
            z = reader.ReadSingle();
            camera.TargetPosition = new Vector3(x, y, z);
            
            // Read name
            camera.Name = reader.ReadCString();
            
            // Read animation tracks
            camera.PositionTrack = new MdxKeyTrack<Vector3>();
            camera.PositionTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            camera.TargetPositionTrack = new MdxKeyTrack<Vector3>();
            camera.TargetPositionTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            camera.RotationTrack = new MdxKeyTrack<Quaternion>();
            camera.RotationTrack.Parse(reader, r => new Quaternion(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            camera.FieldOfViewTrack = new MdxKeyTrack<float>();
            camera.FieldOfViewTrack.Parse(reader, r => r.ReadSingle());
            
            Cameras.Add(camera);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var camera in Cameras)
        {
            // Write basic camera properties
            writer.Write(camera.InclusiveType);
            
            // Write position
            writer.Write(camera.Position.X);
            writer.Write(camera.Position.Y);
            writer.Write(camera.Position.Z);
            
            // Write camera settings
            writer.Write(camera.FieldOfView);
            writer.Write(camera.FarClippingPlane);
            writer.Write(camera.NearClippingPlane);
            
            // Write target position
            writer.Write(camera.TargetPosition.X);
            writer.Write(camera.TargetPosition.Y);
            writer.Write(camera.TargetPosition.Z);
            
            // Write name
            writer.WriteCString(camera.Name);
            
            // Write animation tracks
            camera.PositionTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            camera.TargetPositionTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            camera.RotationTrack.Write(writer, (w, q) => { w.Write(q.X); w.Write(q.Y); w.Write(q.Z); w.Write(q.W); });
            camera.FieldOfViewTrack.Write(writer, (w, f) => w.Write(f));
        }
    }
    
    /// <summary>
    /// Finds a camera by name
    /// </summary>
    /// <param name="name">Name to search for</param>
    /// <returns>The camera with the given name, or null if not found</returns>
    public MdxCamera FindCameraByName(string name)
    {
        return Cameras.FirstOrDefault(c => string.Equals(c.Name, name, StringComparison.OrdinalIgnoreCase));
    }
    
    /// <summary>
    /// Gets the current camera parameters at a specific time
    /// </summary>
    /// <param name="cameraIndex">Index of the camera</param>
    /// <param name="time">Current animation time in milliseconds</param>
    /// <param name="sequenceDuration">Duration of the current sequence</param>
    /// <param name="globalSequences">Dictionary of global sequence durations</param>
    /// <returns>The camera parameters at the specified time</returns>
    public MdxCameraParams GetCameraParams(int cameraIndex, uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        if (cameraIndex < 0 || cameraIndex >= Cameras.Count)
        {
            return null;
        }
        
        var camera = Cameras[cameraIndex];
        var result = new MdxCameraParams();
        
        // Copy static properties
        result.InclusiveType = camera.InclusiveType;
        result.NearClippingPlane = camera.NearClippingPlane;
        result.FarClippingPlane = camera.FarClippingPlane;
        
        // Get position from track or static value
        result.Position = camera.Position;
        if (camera.PositionTrack.NumKeys > 0)
        {
            result.Position = camera.PositionTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        // Get target position from track or static value
        result.TargetPosition = camera.TargetPosition;
        if (camera.TargetPositionTrack.NumKeys > 0)
        {
            result.TargetPosition = camera.TargetPositionTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        // Get rotation from track if available
        result.Rotation = Quaternion.Identity;
        if (camera.RotationTrack.NumKeys > 0)
        {
            result.Rotation = camera.RotationTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        // Get field of view from track or static value
        result.FieldOfView = camera.FieldOfView;
        if (camera.FieldOfViewTrack.NumKeys > 0)
        {
            result.FieldOfView = camera.FieldOfViewTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        return result;
    }
    
    /// <summary>
    /// Creates a view matrix from the camera parameters
    /// </summary>
    /// <param name="parameters">Camera parameters</param>
    /// <returns>View matrix for rendering</returns>
    public static Matrix4x4 CreateViewMatrix(MdxCameraParams parameters)
    {
        // For target cameras, create a look-at matrix
        if (parameters.InclusiveType == 0) // Target camera
        {
            Vector3 position = parameters.Position;
            Vector3 target = parameters.TargetPosition;
            Vector3 up = Vector3.UnitY; // Y is up in MDX
            
            return Matrix4x4.CreateLookAt(position, target, up);
        }
        // For other camera types, use rotation
        else
        {
            Vector3 position = parameters.Position;
            Matrix4x4 rotationMatrix = Matrix4x4.CreateFromQuaternion(parameters.Rotation);
            
            // Combine translation and rotation
            Matrix4x4 viewMatrix = rotationMatrix;
            viewMatrix.M41 = -Vector3.Dot(position, new Vector3(rotationMatrix.M11, rotationMatrix.M21, rotationMatrix.M31));
            viewMatrix.M42 = -Vector3.Dot(position, new Vector3(rotationMatrix.M12, rotationMatrix.M22, rotationMatrix.M32));
            viewMatrix.M43 = -Vector3.Dot(position, new Vector3(rotationMatrix.M13, rotationMatrix.M23, rotationMatrix.M33));
            
            return viewMatrix;
        }
    }
    
    /// <summary>
    /// Creates a projection matrix from the camera parameters
    /// </summary>
    /// <param name="parameters">Camera parameters</param>
    /// <param name="aspectRatio">Screen aspect ratio (width/height)</param>
    /// <returns>Projection matrix for rendering</returns>
    public static Matrix4x4 CreateProjectionMatrix(MdxCameraParams parameters, float aspectRatio)
    {
        return Matrix4x4.CreatePerspectiveFieldOfView(
            parameters.FieldOfView,
            aspectRatio,
            parameters.NearClippingPlane,
            parameters.FarClippingPlane
        );
    }
}

public class MdxCamera
{
    public uint InclusiveType { get; set; }
    public Vector3 Position { get; set; }
    public float FieldOfView { get; set; }
    public float FarClippingPlane { get; set; }
    public float NearClippingPlane { get; set; }
    public Vector3 TargetPosition { get; set; }
    public string Name { get; set; }
    
    public MdxKeyTrack<Vector3> PositionTrack { get; set; }
    public MdxKeyTrack<Vector3> TargetPositionTrack { get; set; }
    public MdxKeyTrack<Quaternion> RotationTrack { get; set; }
    public MdxKeyTrack<float> FieldOfViewTrack { get; set; }
    
    /// <summary>
    /// Gets a friendly name for the camera type
    /// </summary>
    public string TypeName
    {
        get
        {
            switch (InclusiveType)
            {
                case 0: return "Target";
                case 1: return "Portrait";
                case 2: return "Cinematic";
                default: return $"Unknown ({InclusiveType})";
            }
        }
    }
    
    /// <summary>
    /// Computes the direction vector from position to target
    /// </summary>
    public Vector3 Direction
    {
        get
        {
            return Vector3.Normalize(TargetPosition - Position);
        }
    }
    
    /// <summary>
    /// Computes the horizontal field of view for a given aspect ratio
    /// </summary>
    /// <param name="aspectRatio">Screen aspect ratio (width/height)</param>
    /// <returns>Horizontal field of view in radians</returns>
    public float GetHorizontalFieldOfView(float aspectRatio)
    {
        return 2 * MathF.Atan(MathF.Tan(FieldOfView / 2) * aspectRatio);
    }
    
    /// <summary>
    /// Checks if this is a target-tracking camera
    /// </summary>
    public bool IsTargetCamera => InclusiveType == 0;
    
    /// <summary>
    /// Checks if this is a portrait camera
    /// </summary>
    public bool IsPortraitCamera => InclusiveType == 1;
    
    /// <summary>
    /// Checks if this is a cinematic camera
    /// </summary>
    public bool IsCinematicCamera => InclusiveType == 2;
}

public class MdxCameraParams
{
    public uint InclusiveType { get; set; }
    public Vector3 Position { get; set; }
    public Vector3 TargetPosition { get; set; }
    public Quaternion Rotation { get; set; }
    public float FieldOfView { get; set; }
    public float FarClippingPlane { get; set; }
    public float NearClippingPlane { get; set; }
    
    /// <summary>
    /// Computes the direction vector from position to target
    /// </summary>
    public Vector3 Direction
    {
        get
        {
            if (InclusiveType == 0) // Target camera
            {
                return Vector3.Normalize(TargetPosition - Position);
            }
            else
            {
                // For rotation-based cameras, forward is the negative Z axis
                Matrix4x4 rotationMatrix = Matrix4x4.CreateFromQuaternion(Rotation);
                return -new Vector3(rotationMatrix.M13, rotationMatrix.M23, rotationMatrix.M33);
            }
        }
    }
    
    /// <summary>
    /// Computes the up vector for the camera
    /// </summary>
    public Vector3 Up
    {
        get
        {
            if (InclusiveType == 0) // Target camera
            {
                // For target cameras, up is generally world up unless looking straight up/down
                Vector3 forward = Direction;
                Vector3 worldUp = Vector3.UnitY;
                
                // If looking straight up or down, use a fallback
                if (MathF.Abs(Vector3.Dot(forward, worldUp)) > 0.999f)
                {
                    Vector3 fallback = new Vector3(0, 0, 1);
                    return Vector3.Normalize(Vector3.Cross(Vector3.Cross(forward, fallback), forward));
                }
                
                // Normal case
                return Vector3.Normalize(Vector3.Cross(Vector3.Cross(forward, worldUp), forward));
            }
            else
            {
                // For rotation-based cameras, up is the positive Y axis
                Matrix4x4 rotationMatrix = Matrix4x4.CreateFromQuaternion(Rotation);
                return new Vector3(rotationMatrix.M12, rotationMatrix.M22, rotationMatrix.M32);
            }
        }
    }
}
``` 