using System;
using System.Collections.Generic;
using System.IO;

namespace MdxViewer.Formats.Mdx
{
    /// <summary>
    /// Geoset animation data for WoW Alpha 0.5.3 MDX files.
    /// Contains alpha (opacity) and color animation tracks for geosets.
    /// </summary>
    public class GeosetAnimation
    {
        /// <summary>
        /// The geoset ID this animation applies to.
        /// </summary>
        public uint GeosetId { get; set; }

        /// <summary>
        /// Default alpha (opacity) value (0.0 to 1.0).
        /// </summary>
        public float DefaultAlpha { get; set; } = 1.0f;

        /// <summary>
        /// Default color values (RGB).
        /// </summary>
        public C3Color DefaultColor { get; set; } = new C3Color(1.0f, 1.0f, 1.0f);

        /// <summary>
        /// Alpha animation keyframes.
        /// </summary>
        public KeyframeTrack<float> AlphaKeys { get; set; } = new KeyframeTrack<float>();

        /// <summary>
        /// Color animation keyframes.
        /// </summary>
        public KeyframeTrack<C3Color> ColorKeys { get; set; } = new KeyframeTrack<C3Color>();

        /// <summary>
        /// Unknown field - possibly flags or padding.
        /// </summary>
        public uint Unknown { get; set; }

        /// <summary>
        /// Evaluates the alpha value at the given time.
        /// </summary>
        /// <param name="time">Animation time in milliseconds.</param>
        /// <param name="globalSequenceTime">Optional global sequence time override.</param>
        /// <returns>The interpolated alpha value.</returns>
        public float EvaluateAlpha(int time, int? globalSequenceTime = null)
        {
            if (AlphaKeys.Keyframes.Count == 0)
                return DefaultAlpha;

            int evalTime = globalSequenceTime ?? time;
            return AlphaKeys.Evaluate(evalTime);
        }

        /// <summary>
        /// Evaluates the color value at the given time.
        /// </summary>
        /// <param name="time">Animation time in milliseconds.</param>
        /// <param name="globalSequenceTime">Optional global sequence time override.</param>
        /// <returns>The interpolated color value.</returns>
        public C3Color EvaluateColor(int time, int? globalSequenceTime = null)
        {
            if (ColorKeys.Keyframes.Count == 0)
                return DefaultColor;

            int evalTime = globalSequenceTime ?? time;
            return ColorKeys.Evaluate(evalTime);
        }
    }

    /// <summary>
    /// Represents a keyframe track with interpolation support.
    /// </summary>
    /// <typeparam name="T">The value type (float for alpha, C3Color for color).</typeparam>
    public class KeyframeTrack<T>
    {
        /// <summary>
        /// The keyframes in this track.
        /// </summary>
        public List<Keyframe<T>> Keyframes { get; set; } = new List<Keyframe<T>>();

        /// <summary>
        /// Interpolation type for this track.
        /// </summary>
        public InterpolationType InterpolationType { get; set; } = InterpolationType.Linear;

        /// <summary>
        /// Global sequence ID, or -1 if not used.
        /// </summary>
        public int GlobalSequenceId { get; set; } = -1;

        /// <summary>
        /// Evaluates the track value at the given time.
        /// </summary>
        /// <param name="time">Animation time in milliseconds.</param>
        /// <returns>The interpolated value.</returns>
        public T Evaluate(int time)
        {
            if (Keyframes.Count == 0)
                return default(T);

            if (Keyframes.Count == 1)
                return Keyframes[0].Value;

            // Find the two keyframes surrounding the time
            int startIdx = 0;
            for (int i = 0; i < Keyframes.Count - 1; i++)
            {
                if (time >= Keyframes[i].Time && time < Keyframes[i + 1].Time)
                {
                    startIdx = i;
                    break;
                }
                startIdx = i;
            }

            var start = Keyframes[startIdx];
            var end = Keyframes[Math.Min(startIdx + 1, Keyframes.Count - 1)];

            // Clamp time to keyframe range
            if (time <= start.Time)
                return start.Value;
            if (time >= end.Time)
                return end.Value;

            // Calculate interpolation factor (0.0 to 1.0)
            float t = (float)(time - start.Time) / (end.Time - start.Time);

            // Interpolate based on type
            return Interpolate(start, end, t);
        }

        private T Interpolate(Keyframe<T> start, Keyframe<T> end, float t)
        {
            if (typeof(T) == typeof(float))
            {
                float startVal = (float)(object)start.Value;
                float endVal = (float)(object)end.Value;

                switch (InterpolationType)
                {
                    case InterpolationType.Linear:
                        return (T)(object)(startVal + (endVal - startVal) * t);

                    case InterpolationType.Hermite:
                    case InterpolationType.Bezier:
                    case InterpolationType.Bezier2:
                        // For simplicity, fall back to linear for now
                        // A full implementation would use the tangent values
                        return (T)(object)(startVal + (endVal - startVal) * t);

                    default:
                        return (T)(object)startVal;
                }
            }
            else if (typeof(T) == typeof(C3Color))
            {
                C3Color startVal = (C3Color)(object)start.Value;
                C3Color endVal = (C3Color)(object)end.Value;

                switch (InterpolationType)
                {
                    case InterpolationType.Linear:
                        return (T)(object)new C3Color(
                            startVal.R + (endVal.R - startVal.R) * t,
                            startVal.G + (endVal.G - startVal.G) * t,
                            startVal.B + (endVal.B - startVal.B) * t
                        );

                    case InterpolationType.Hermite:
                    case InterpolationType.Bezier:
                    case InterpolationType.Bezier2:
                        // For simplicity, fall back to linear for now
                        return (T)(object)new C3Color(
                            startVal.R + (endVal.R - startVal.R) * t,
                            startVal.G + (endVal.G - startVal.G) * t,
                            startVal.B + (endVal.B - startVal.B) * t
                        );

                    default:
                        return (T)(object)startVal;
                }
            }

            return default(T);
        }
    }

    /// <summary>
    /// Represents a single keyframe in an animation track.
    /// </summary>
    /// <typeparam name="T">The value type.</typeparam>
    public class Keyframe<T>
    {
        /// <summary>
        /// The time of this keyframe in milliseconds.
        /// </summary>
        public int Time { get; set; }

        /// <summary>
        /// The value at this keyframe.
        /// </summary>
        public T Value { get; set; }

        /// <summary>
        /// Tangent in value (for Hermite/Bezier interpolation).
        /// Only used for float values.
        /// </summary>
        public float TangentIn { get; set; }

        /// <summary>
        /// Tangent out value (for Hermite/Bezier interpolation).
        /// Only used for float values.
        /// </summary>
        public float TangentOut { get; set; }

        /// <summary>
        /// Tangent in values for color (RGB).
        /// Only used for C3Color values.
        /// </summary>
        public C3Color ColorTangentIn { get; set; }

        /// <summary>
        /// Tangent out values for color (RGB).
        /// Only used for C3Color values.
        /// </summary>
        public C3Color ColorTangentOut { get; set; }
    }

    /// <summary>
    /// Interpolation types for animation keyframes.
    /// </summary>
    public enum InterpolationType
    {
        /// <summary>
        /// Linear interpolation between keyframes.
        /// </summary>
        Linear = 0,

        /// <summary>
        /// Hermite interpolation with tangents.
        /// </summary>
        Hermite = 1,

        /// <summary>
        /// Bezier interpolation.
        /// </summary>
        Bezier = 2,

        /// <summary>
        /// Bezier interpolation variant.
        /// </summary>
        Bezier2 = 3
    }

    /// <summary>
    /// 3-component color (RGB).
    /// </summary>
    public class C3Color
    {
        public float R { get; set; }
        public float G { get; set; }
        public float B { get; set; }

        public C3Color() { }

        public C3Color(float r, float g, float b)
        {
            R = r;
            G = g;
            B = b;
        }

        public static C3Color operator +(C3Color a, C3Color b)
        {
            return new C3Color(a.R + b.R, a.G + b.G, a.B + b.B);
        }

        public static C3Color operator -(C3Color a, C3Color b)
        {
            return new C3Color(a.R - b.R, a.G - b.G, a.B - b.B);
        }

        public static C3Color operator *(C3Color a, float scalar)
        {
            return new C3Color(a.R * scalar, a.G * scalar, a.B * scalar);
        }
    }

    /// <summary>
    /// Reader for ATSQ (Geoset Animation) chunks in WoW Alpha 0.5.3 MDX files.
    /// </summary>
    public static class AtsqReader
    {
        private const uint KGAO = 0x4f41474b; // "KGAO" - Keyframe Group Alpha Opacity
        private const uint KGAC = 0x4341474b; // "KGAC" - Keyframe Group Alpha Color

        /// <summary>
        /// Reads an ATSQ chunk from the given binary reader.
        /// </summary>
        /// <param name="br">The binary reader positioned at the ATSQ chunk.</param>
        /// <param name="chunkSize">The size of the ATSQ chunk.</param>
        /// <returns>The parsed geoset animation data.</returns>
        public static GeosetAnimation Read(BinaryReader br, uint chunkSize)
        {
            long startPos = br.BaseStream.Position;
            long endPos = startPos + chunkSize;

            var anim = new GeosetAnimation();

            // Read header
            anim.GeosetId = br.ReadUInt32();
            anim.DefaultAlpha = br.ReadSingle();
            anim.DefaultColor = new C3Color(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            anim.Unknown = br.ReadUInt32();

            // Read sub-chunks
            while (br.BaseStream.Position < endPos - 8)
            {
                uint tag = br.ReadUInt32();
                uint subChunkSize = br.ReadUInt32();

                long subChunkEnd = br.BaseStream.Position + subChunkSize;

                switch (tag)
                {
                    case KGAO:
                        ReadAlphaKeys(br, anim.AlphaKeys);
                        break;

                    case KGAC:
                        ReadColorKeys(br, anim.ColorKeys);
                        break;

                    default:
                        // Skip unknown sub-chunks
                        br.BaseStream.Position = subChunkEnd;
                        break;
                }

                // Ensure we're at the correct position
                if (br.BaseStream.Position != subChunkEnd)
                {
                    br.BaseStream.Position = subChunkEnd;
                }
            }

            return anim;
        }

        /// <summary>
        /// Reads alpha keyframes from a KGAO sub-chunk.
        /// </summary>
        private static void ReadAlphaKeys(BinaryReader br, KeyframeTrack<float> track)
        {
            uint keyframeCount = br.ReadUInt32();
            track.InterpolationType = (InterpolationType)br.ReadUInt32();
            track.GlobalSequenceId = br.ReadInt32();

            for (uint i = 0; i < keyframeCount; i++)
            {
                var keyframe = new Keyframe<float>
                {
                    Time = br.ReadInt32(),
                    Value = br.ReadSingle()
                };

                // Read tangents for Hermite/Bezier interpolation
                if (track.InterpolationType >= InterpolationType.Hermite)
                {
                    keyframe.TangentIn = br.ReadSingle();
                    keyframe.TangentOut = br.ReadSingle();
                }

                track.Keyframes.Add(keyframe);
            }
        }

        /// <summary>
        /// Reads color keyframes from a KGAC sub-chunk.
        /// </summary>
        private static void ReadColorKeys(BinaryReader br, KeyframeTrack<C3Color> track)
        {
            uint keyframeCount = br.ReadUInt32();
            track.InterpolationType = (InterpolationType)br.ReadUInt32();
            track.GlobalSequenceId = br.ReadInt32();

            for (uint i = 0; i < keyframeCount; i++)
            {
                var keyframe = new Keyframe<C3Color>
                {
                    Time = br.ReadInt32(),
                    Value = new C3Color(br.ReadSingle(), br.ReadSingle(), br.ReadSingle())
                };

                // Read tangents for Hermite/Bezier interpolation
                if (track.InterpolationType >= InterpolationType.Hermite)
                {
                    keyframe.ColorTangentIn = new C3Color(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                    keyframe.ColorTangentOut = new C3Color(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                }

                track.Keyframes.Add(keyframe);
            }
        }
    }
}
