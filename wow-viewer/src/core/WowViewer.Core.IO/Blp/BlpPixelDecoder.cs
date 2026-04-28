using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.Core.IO.Blp;

public static class BlpPixelDecoder
{
	public static byte[] DecodeRgba(string blpPath, int mipLevel = 0)
	{
		using FileStream stream = File.OpenRead(blpPath);
		return DecodeRgba(stream, mipLevel);
	}

	public static byte[] DecodeRgba(Stream stream, int mipLevel = 0)
	{
		using BlpFile blp = new(stream);
		using Bitmap bitmap = blp.GetBitmap(mipLevel);
		int width = bitmap.Width;
		int height = bitmap.Height;
		byte[] rgba = new byte[width * height * 4];

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				System.Drawing.Color pixel = bitmap.GetPixel(x, y);
				int idx = (y * width + x) * 4;
				rgba[idx] = pixel.R;
				rgba[idx + 1] = pixel.G;
				rgba[idx + 2] = pixel.B;
				rgba[idx + 3] = pixel.A;
			}
		}

		return rgba;
	}

	public static (byte[] Rgba, int Width, int Height) DecodeRgbaWithDimensions(string blpPath, int mipLevel = 0)
	{
		using FileStream stream = File.OpenRead(blpPath);
		using BlpFile blp = new(stream);
		using Bitmap bitmap = blp.GetBitmap(mipLevel);
		int width = bitmap.Width;
		int height = bitmap.Height;
		byte[] rgba = new byte[width * height * 4];

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				System.Drawing.Color pixel = bitmap.GetPixel(x, y);
				int idx = (y * width + x) * 4;
				rgba[idx] = pixel.R;
				rgba[idx + 1] = pixel.G;
				rgba[idx + 2] = pixel.B;
				rgba[idx + 3] = pixel.A;
			}
		}

		return (rgba, width, height);
	}

	public static void SaveAsPng(string blpPath, string outputPath, int mipLevel = 0)
	{
		(byte[] rgba, int width, int height) = DecodeRgbaWithDimensions(blpPath, mipLevel);
		Image<Rgba32> image = Image.LoadPixelData<Rgba32>(rgba, width, height);
		image.SaveAsPng(outputPath);
	}

	public static byte[] DecodeGrayscale(string blpPath, int mipLevel = 0)
	{
		(byte[] rgba, int width, int height) = DecodeRgbaWithDimensions(blpPath, mipLevel);
		byte[] gray = new byte[width * height];

		for (int i = 0; i < width * height; i++)
		{
			int idx = i * 4;
			gray[i] = (byte)((rgba[idx] * 77 + rgba[idx + 1] * 151 + rgba[idx + 2] * 28) >> 8);
		}

		return gray;
	}
}
