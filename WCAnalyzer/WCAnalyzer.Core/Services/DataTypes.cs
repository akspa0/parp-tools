using System;
using System.Collections.Generic;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Represents a 3D vector with X, Y, and Z components.
    /// </summary>
    public class Vector3
    {
        /// <summary>
        /// Gets or sets the X component.
        /// </summary>
        public float X { get; set; }

        /// <summary>
        /// Gets or sets the Y component.
        /// </summary>
        public float Y { get; set; }

        /// <summary>
        /// Gets or sets the Z component.
        /// </summary>
        public float Z { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Vector3"/> class.
        /// </summary>
        public Vector3()
        {
            X = 0;
            Y = 0;
            Z = 0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Vector3"/> class with the specified components.
        /// </summary>
        /// <param name="x">The X component.</param>
        /// <param name="y">The Y component.</param>
        /// <param name="z">The Z component.</param>
        public Vector3(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            return $"({X}, {Y}, {Z})";
        }
    }

    /// <summary>
    /// Represents a 2D vector with X and Y components.
    /// </summary>
    public class Vector2
    {
        /// <summary>
        /// Gets or sets the X component.
        /// </summary>
        public float X { get; set; }

        /// <summary>
        /// Gets or sets the Y component.
        /// </summary>
        public float Y { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Vector2"/> class.
        /// </summary>
        public Vector2()
        {
            X = 0;
            Y = 0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Vector2"/> class with the specified components.
        /// </summary>
        /// <param name="x">The X component.</param>
        /// <param name="y">The Y component.</param>
        public Vector2(float x, float y)
        {
            X = x;
            Y = y;
        }

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            return $"({X}, {Y})";
        }
    }

    /// <summary>
    /// Represents link data from a PM4 file.
    /// </summary>
    public class LinkData
    {
        /// <summary>
        /// Gets or sets the first value.
        /// </summary>
        public int Value0x00 { get; set; }

        /// <summary>
        /// Gets or sets the second value.
        /// </summary>
        public int Value0x04 { get; set; }

        /// <summary>
        /// Gets or sets the third value.
        /// </summary>
        public int Value0x08 { get; set; }

        /// <summary>
        /// Gets or sets the fourth value.
        /// </summary>
        public int Value0x0C { get; set; }
        
        /// <summary>
        /// Gets or sets the source index.
        /// </summary>
        public int SourceIndex { get; set; }
        
        /// <summary>
        /// Gets or sets the target index.
        /// </summary>
        public int TargetIndex { get; set; }
    }

    /// <summary>
    /// Represents position data from a PM4 file.
    /// </summary>
    public class PositionData
    {
        /// <summary>
        /// Gets or sets the first value.
        /// </summary>
        public int Value0x00 { get; set; }

        /// <summary>
        /// Gets or sets the second value.
        /// </summary>
        public int Value0x04 { get; set; }

        /// <summary>
        /// Gets or sets the third value.
        /// </summary>
        public float Value0x08 { get; set; }

        /// <summary>
        /// Gets or sets the fourth value.
        /// </summary>
        public float Value0x0C { get; set; }

        /// <summary>
        /// Gets or sets the fifth value.
        /// </summary>
        public float Value0x10 { get; set; }

        /// <summary>
        /// Gets or sets the sixth value.
        /// </summary>
        public float Value0x14 { get; set; }

        /// <summary>
        /// Gets or sets the seventh value.
        /// </summary>
        public float Value0x18 { get; set; }

        /// <summary>
        /// Gets or sets the eighth value.
        /// </summary>
        public float Value0x1C { get; set; }

        /// <summary>
        /// Determines whether this position data entry is a special entry.
        /// </summary>
        /// <returns>True if this is a special entry, false otherwise.</returns>
        public bool IsSpecialEntry()
        {
            return Value0x00 == -1;
        }

        /// <summary>
        /// Gets the special value for this position data entry.
        /// </summary>
        /// <returns>The special value.</returns>
        public int SpecialValue()
        {
            return Value0x04;
        }
    }

    /// <summary>
    /// Represents a position reference from a PM4 file.
    /// </summary>
    public class PositionReference
    {
        /// <summary>
        /// Gets or sets the first value.
        /// </summary>
        public int Value0x00 { get; set; }

        /// <summary>
        /// Gets or sets the second value.
        /// </summary>
        public int Value0x04 { get; set; }

        /// <summary>
        /// Gets or sets the third value.
        /// </summary>
        public int Value0x08 { get; set; }

        /// <summary>
        /// Gets or sets the fourth value.
        /// </summary>
        public int Value0x0C { get; set; }
        
        /// <summary>
        /// Gets or sets the first simplified value.
        /// </summary>
        public int Value1 { get; set; }
        
        /// <summary>
        /// Gets or sets the second simplified value.
        /// </summary>
        public int Value2 { get; set; }
    }

    /// <summary>
    /// Represents material data from a PD4 file.
    /// </summary>
    public class MaterialData
    {
        /// <summary>
        /// Gets or sets the texture index.
        /// </summary>
        public int TextureIndex { get; set; }

        /// <summary>
        /// Gets or sets the flags.
        /// </summary>
        public int Flags { get; set; }
        
        /// <summary>
        /// Gets or sets the first additional value.
        /// </summary>
        public int Value1 { get; set; }
        
        /// <summary>
        /// Gets or sets the second additional value.
        /// </summary>
        public int Value2 { get; set; }
    }
} 