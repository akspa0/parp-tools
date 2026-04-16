using System.Buffers.Binary;
using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2SoftwareVisualSnapshot
{
    public M2SoftwareVisualSnapshot(
        int width,
        int height,
        int litPixelCount,
        string visualHash,
        byte[] rgbPixels)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 16);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 16);
        ArgumentOutOfRangeException.ThrowIfNegative(litPixelCount);
        ArgumentException.ThrowIfNullOrWhiteSpace(visualHash);
        ArgumentNullException.ThrowIfNull(rgbPixels);

        Width = width;
        Height = height;
        LitPixelCount = litPixelCount;
        VisualHash = visualHash;
        RgbPixels = rgbPixels;
    }

    public int Width { get; }

    public int Height { get; }

    public int LitPixelCount { get; }

    public string VisualHash { get; }

    public byte[] RgbPixels { get; }
}

public static class M2SoftwareVisualSnapshotBuilder
{
    public static M2SoftwareVisualSnapshot Build(M2RenderFrame frame, int width = 256, int height = 256)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 16);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 16);

        byte[] pixels = new byte[checked(width * height * 3)];
        Fill(pixels, 7, 9, 11);
        float[] depthBuffer = new float[checked(width * height)];
        Array.Fill(depthBuffer, float.NegativeInfinity);

        M2RenderBackendVertex[] allVertices = frame.DrawCommands
            .SelectMany(static command => command.Vertices)
            .ToArray();
        if (allVertices.Length == 0)
            return CreateSnapshot(width, height, pixels, litPixelCount: 0);

        const int ViewGap = 6;
        int cellWidth = Math.Max(32, (width - (ViewGap * 3)) / 2);
        int cellHeight = Math.Max(32, (height - (ViewGap * 3)) / 2);

        SnapshotView[] views =
        [
            new("iso", Matrix4x4.CreateFromYawPitchRoll(-MathF.PI / 4.0f, 0.60f, 0.0f), ViewGap, ViewGap),
            new("side", Matrix4x4.CreateFromYawPitchRoll(MathF.PI / 2.0f, 0.0f, 0.0f), (ViewGap * 2) + cellWidth, ViewGap),
            new("front", Matrix4x4.Identity, ViewGap, (ViewGap * 2) + cellHeight),
            new("top", Matrix4x4.CreateFromYawPitchRoll(0.0f, -MathF.PI / 2.0f, 0.0f), (ViewGap * 2) + cellWidth, (ViewGap * 2) + cellHeight),
        ];

        foreach (SnapshotView view in views)
        {
            SnapshotProjection projection = CreateProjection(allVertices, cellWidth, cellHeight, view.Rotation);
            RenderView(frame, pixels, depthBuffer, width, height, cellWidth, cellHeight, view, projection);
        }

        int litPixelCount = CountLitPixels(pixels);
        return CreateSnapshot(width, height, pixels, litPixelCount);
    }

    private static void RenderView(
        M2RenderFrame frame,
        byte[] pixels,
        float[] depthBuffer,
        int canvasWidth,
        int canvasHeight,
        int cellWidth,
        int cellHeight,
        SnapshotView view,
        SnapshotProjection projection)
    {
        int commandIndex = 0;
        foreach (M2RenderDrawCommand command in frame.DrawCommands)
        {
            if (command.Vertices.Count == 0)
            {
                commandIndex++;
                continue;
            }

            (byte r, byte g, byte b) = ResolveColor(command, commandIndex);
            ProjectedVertex[] projectedVertices = command.Vertices
                .Select(vertex => Project(vertex, projection, cellWidth, cellHeight, view.OffsetX, view.OffsetY))
                .ToArray();

            if (command.Indices.Count >= 3)
            {
                for (int index = 0; index + 2 < command.Indices.Count; index += 3)
                {
                    uint ia = command.Indices[index];
                    uint ib = command.Indices[index + 1];
                    uint ic = command.Indices[index + 2];
                    if (ia >= command.Vertices.Count || ib >= command.Vertices.Count || ic >= command.Vertices.Count)
                        continue;

                    RasterizeTriangle(
                        pixels,
                        depthBuffer,
                        canvasWidth,
                        canvasHeight,
                        projectedVertices[(int)ia],
                        projectedVertices[(int)ib],
                        projectedVertices[(int)ic],
                        r,
                        g,
                        b);
                }
            }
            else
            {
                foreach (ProjectedVertex vertex in projectedVertices)
                    DrawPoint(pixels, canvasWidth, canvasHeight, vertex.X, vertex.Y, r, g, b);
            }

            commandIndex++;
        }
    }

    public static void WriteBmp(Stream output, M2SoftwareVisualSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(snapshot);

        int rowStride = ((snapshot.Width * 3) + 3) & ~3;
        int pixelDataSize = checked(rowStride * snapshot.Height);
        int fileSize = 14 + 40 + pixelDataSize;
        Span<byte> header = stackalloc byte[54];
        header[0] = (byte)'B';
        header[1] = (byte)'M';
        BinaryPrimitives.WriteInt32LittleEndian(header[2..6], fileSize);
        BinaryPrimitives.WriteInt32LittleEndian(header[10..14], 54);
        BinaryPrimitives.WriteInt32LittleEndian(header[14..18], 40);
        BinaryPrimitives.WriteInt32LittleEndian(header[18..22], snapshot.Width);
        BinaryPrimitives.WriteInt32LittleEndian(header[22..26], snapshot.Height);
        BinaryPrimitives.WriteInt16LittleEndian(header[26..28], 1);
        BinaryPrimitives.WriteInt16LittleEndian(header[28..30], 24);
        BinaryPrimitives.WriteInt32LittleEndian(header[34..38], pixelDataSize);
        output.Write(header);

        byte[] padding = new byte[rowStride - (snapshot.Width * 3)];
        for (int y = snapshot.Height - 1; y >= 0; y--)
        {
            int rowOffset = y * snapshot.Width * 3;
            for (int x = 0; x < snapshot.Width; x++)
            {
                int offset = rowOffset + (x * 3);
                output.WriteByte(snapshot.RgbPixels[offset + 2]);
                output.WriteByte(snapshot.RgbPixels[offset + 1]);
                output.WriteByte(snapshot.RgbPixels[offset + 0]);
            }

            if (padding.Length > 0)
                output.Write(padding);
        }
    }

    private static M2SoftwareVisualSnapshot CreateSnapshot(int width, int height, byte[] pixels, int litPixelCount)
    {
        return new M2SoftwareVisualSnapshot(width, height, litPixelCount, ComputeHash(width, height, pixels), pixels);
    }

    private static SnapshotProjection CreateProjection(IReadOnlyList<M2RenderBackendVertex> vertices, int width, int height, Matrix4x4 rotation)
    {
        const int Margin = 12;
        Vector3 min = new(float.PositiveInfinity);
        Vector3 max = new(float.NegativeInfinity);
        foreach (M2RenderBackendVertex vertex in vertices)
        {
            min = Vector3.Min(min, vertex.Position);
            max = Vector3.Max(max, vertex.Position);
        }

        Vector3 center = (min + max) * 0.5f;
        Vector3 rotatedMin = new(float.PositiveInfinity);
        Vector3 rotatedMax = new(float.NegativeInfinity);
        foreach (M2RenderBackendVertex vertex in vertices)
        {
            Vector3 rotated = Vector3.Transform(vertex.Position - center, rotation);
            rotatedMin = Vector3.Min(rotatedMin, rotated);
            rotatedMax = Vector3.Max(rotatedMax, rotated);
        }

        float extentX = Math.Max(rotatedMax.X - rotatedMin.X, 0.0001f);
        float extentY = Math.Max(rotatedMax.Y - rotatedMin.Y, 0.0001f);
        float scaleX = (width - (Margin * 2) - 1) / extentX;
        float scaleY = (height - (Margin * 2) - 1) / extentY;
        float scale = MathF.Min(scaleX, scaleY);
        return new SnapshotProjection(rotation, center, rotatedMin, rotatedMax, scale, Margin);
    }

    private static ProjectedVertex Project(M2RenderBackendVertex vertex, SnapshotProjection projection, int width, int height, int offsetX, int offsetY)
    {
        Vector3 rotatedPosition = Vector3.Transform(vertex.Position - projection.Center, projection.Rotation);
        float relativeX = rotatedPosition.X - projection.Min.X;
        int x = offsetX + projection.Margin + (int)MathF.Round(relativeX * projection.Scale);
        int y = offsetY + projection.Margin + (int)MathF.Round((projection.Max.Y - rotatedPosition.Y) * projection.Scale);
        x = Math.Clamp(x, offsetX, offsetX + width - 1);
        y = Math.Clamp(y, offsetY, offsetY + height - 1);
        return new ProjectedVertex(x, y, rotatedPosition.Z);
    }

    private static (byte R, byte G, byte B) ResolveColor(M2RenderDrawCommand command, int commandIndex)
    {
        (byte r, byte g, byte b) = command.Family switch
        {
            M2RenderEntryFamily.Projected => ((byte)255, (byte)186, (byte)76),
            M2RenderEntryFamily.Ribbon => ((byte)104, (byte)226, (byte)151),
            M2RenderEntryFamily.Particle => ((byte)255, (byte)116, (byte)72),
            M2RenderEntryFamily.Doodad => ((byte)158, (byte)128, (byte)240),
            _ => ((byte)88, (byte)183, (byte)255),
        };

        int shade = 220 + ((commandIndex * 17) % 36);
        return ((byte)(r * shade / 255), (byte)(g * shade / 255), (byte)(b * shade / 255));
    }

    private static void RasterizeTriangle(
        byte[] pixels,
        float[] depthBuffer,
        int width,
        int height,
        ProjectedVertex a,
        ProjectedVertex b,
        ProjectedVertex c,
        byte r,
        byte g,
        byte bColor)
    {
        float area = Edge(a.X, a.Y, b.X, b.Y, c.X, c.Y);
        if (Math.Abs(area) < 0.0001f)
            return;

        int minX = Math.Clamp(Math.Min(a.X, Math.Min(b.X, c.X)), 0, width - 1);
        int maxX = Math.Clamp(Math.Max(a.X, Math.Max(b.X, c.X)), 0, width - 1);
        int minY = Math.Clamp(Math.Min(a.Y, Math.Min(b.Y, c.Y)), 0, height - 1);
        int maxY = Math.Clamp(Math.Max(a.Y, Math.Max(b.Y, c.Y)), 0, height - 1);

        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                float px = x + 0.5f;
                float py = y + 0.5f;
                float w0 = Edge(b.X, b.Y, c.X, c.Y, px, py);
                float w1 = Edge(c.X, c.Y, a.X, a.Y, px, py);
                float w2 = Edge(a.X, a.Y, b.X, b.Y, px, py);
                bool inside = area > 0
                    ? w0 >= 0 && w1 >= 0 && w2 >= 0
                    : w0 <= 0 && w1 <= 0 && w2 <= 0;
                if (!inside)
                    continue;

                float invArea = 1.0f / area;
                float bary0 = w0 * invArea;
                float bary1 = w1 * invArea;
                float bary2 = w2 * invArea;
                float depth = (a.Depth * bary0) + (b.Depth * bary1) + (c.Depth * bary2);
                int pixelIndex = (y * width) + x;
                if (depth < depthBuffer[pixelIndex])
                    continue;

                depthBuffer[pixelIndex] = depth;
                SetPixel(pixels, width, height, x, y, r, g, bColor);
            }
        }

    }

    private static float Edge(float ax, float ay, float bx, float by, float px, float py)
    {
        return ((px - ax) * (by - ay)) - ((py - ay) * (bx - ax));
    }

    private static void Fill(byte[] pixels, byte r, byte g, byte b)
    {
        for (int offset = 0; offset < pixels.Length; offset += 3)
        {
            pixels[offset + 0] = r;
            pixels[offset + 1] = g;
            pixels[offset + 2] = b;
        }
    }

    private static void DrawLine(byte[] pixels, int width, int height, int x0, int y0, int x1, int y1, byte r, byte g, byte b)
    {
        int dx = Math.Abs(x1 - x0);
        int sx = x0 < x1 ? 1 : -1;
        int dy = -Math.Abs(y1 - y0);
        int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;

        while (true)
        {
            DrawPoint(pixels, width, height, x0, y0, r, g, b);
            if (x0 == x1 && y0 == y1)
                break;

            int e2 = error * 2;
            if (e2 >= dy)
            {
                error += dy;
                x0 += sx;
            }

            if (e2 <= dx)
            {
                error += dx;
                y0 += sy;
            }
        }
    }

    private static void DrawPoint(byte[] pixels, int width, int height, int x, int y, byte r, byte g, byte b)
    {
        SetPixel(pixels, width, height, x, y, r, g, b);
    }

    private static void SetPixel(byte[] pixels, int width, int height, int x, int y, byte r, byte g, byte b)
    {
        if ((uint)x >= (uint)width || (uint)y >= (uint)height)
            return;

        int offset = ((y * width) + x) * 3;
        pixels[offset + 0] = r;
        pixels[offset + 1] = g;
        pixels[offset + 2] = b;
    }

    private static int CountLitPixels(byte[] pixels)
    {
        int count = 0;
        for (int offset = 0; offset < pixels.Length; offset += 3)
        {
            if (pixels[offset + 0] != 7 || pixels[offset + 1] != 9 || pixels[offset + 2] != 11)
                count++;
        }

        return count;
    }

    private static string ComputeHash(int width, int height, byte[] pixels)
    {
        StringBuilder builder = new();
        builder.Append(width.ToString(CultureInfo.InvariantCulture)).Append('x').Append(height.ToString(CultureInfo.InvariantCulture));
        byte[] headerBytes = Encoding.UTF8.GetBytes(builder.ToString());
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hash.AppendData(headerBytes);
        hash.AppendData(pixels);
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }

    private readonly record struct SnapshotProjection(
        Matrix4x4 Rotation,
        Vector3 Center,
        Vector3 Min,
        Vector3 Max,
        float Scale,
        int Margin);

    private readonly record struct SnapshotView(
        string Name,
        Matrix4x4 Rotation,
        int OffsetX,
        int OffsetY);

    private readonly record struct ProjectedVertex(
        int X,
        int Y,
        float Depth);
}
