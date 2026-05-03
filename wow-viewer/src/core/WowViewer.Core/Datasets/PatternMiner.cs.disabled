using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using System.Security.Cryptography;
using System.Numerics;

namespace WowViewer.Core.Datasets;

public sealed record PatternStamp(
	string TextureName,
	int Width,
	int Height,
	int TileSizeX,
	int TileSizeY,
	double PeriodicityScore,
	double[] DominantFrequencies,
	double[][] Autocorrelation,
	double[][] MagnitudeSpectrum,
	string PatternScaleHint,
	string EdgeBehavior,
	byte[] MipSignature,
	double MeanLuminance,
	double LuminanceStdDev,
	double[] MeanRgb,
	double[] RgbStdDev,
	double MeanHueDegrees,
	double MeanSaturation,
	double Colorfulness,
	string MeanColorHex,
	byte[] ColorMipSignature,
	byte[] ChromaMipSignature,
	byte[] ChromaDetailSignature,
	double ChromaDetailEnergy,
	string[] DominantColorsHex,
	string PatternSignatureHash,
	string ColorSignatureHash,
	string ChromaSignatureHash,
	string ChromaDetailSignatureHash);

public sealed record DominantFrequency(
	double FrequencyX,
	double FrequencyY,
	double Magnitude);

public static class PatternMiner
{
	public static PatternStamp AnalyzeTexture(byte[] rgba, int width, int height, string textureName)
	{
		byte[] gray = RgbaToGrayscale(rgba, width, height);
		double[,] autoCorr2d = ComputeAutocorrelation2D(gray, width, height);
		double[][] autoCorr = ToJaggedArray(autoCorr2d);
		DominantFrequency[] freqs = ExtractDominantFrequencies(gray, width, height);
		double[,] magSpec2d = ComputeMagnitudeSpectrum(gray, width, height);
		double[][] magSpec = ToJaggedArray(magSpec2d);
		(int tileX, int tileY, double periodicity) = DetectTileSize(autoCorr2d, width, height);
		string patternScale = ClassifyPatternScale(tileX, tileY, periodicity, freqs);
		string edgeBehavior = ClassifyEdgeBehavior(autoCorr2d, width, height);

		double mean = 0, std = 0;
		ComputeLuminanceStats(gray, out mean, out std);

		byte[] mipSig = ComputeMipSignature(gray, width, height, 8);
		byte[] colorMipSig = ComputeColorMipSignature(rgba, width, height, 8);
		byte[] chromaMipSig = ComputeChromaMipSignature(rgba, width, height, 8);
		byte[] chromaDetailSig = ComputeChromaDetailSignature(rgba, width, height, 16);
		double chromaDetailEnergy = ComputeDetailEnergy(chromaDetailSig);
		string[] dominantColors = ExtractDominantColors(rgba, width, height, 6);
		ComputeColorStats(
			rgba,
			width,
			height,
			out double[] meanRgb,
			out double[] rgbStdDev,
			out double meanHueDegrees,
			out double meanSaturation,
			out double colorfulness,
			out string meanColorHex);

		return new PatternStamp(
			TextureName: textureName,
			Width: width,
			Height: height,
			TileSizeX: tileX,
			TileSizeY: tileY,
			PeriodicityScore: periodicity,
			DominantFrequencies: freqs.Select(f => f.Magnitude).Take(5).ToArray(),
			Autocorrelation: autoCorr,
			MagnitudeSpectrum: magSpec,
			PatternScaleHint: patternScale,
			EdgeBehavior: edgeBehavior,
			MipSignature: mipSig,
			MeanLuminance: mean,
			LuminanceStdDev: std,
			MeanRgb: meanRgb,
			RgbStdDev: rgbStdDev,
			MeanHueDegrees: meanHueDegrees,
			MeanSaturation: meanSaturation,
			Colorfulness: colorfulness,
			MeanColorHex: meanColorHex,
			ColorMipSignature: colorMipSig,
			ChromaMipSignature: chromaMipSig,
			ChromaDetailSignature: chromaDetailSig,
			ChromaDetailEnergy: chromaDetailEnergy,
			DominantColorsHex: dominantColors,
			PatternSignatureHash: HashBytes(mipSig),
			ColorSignatureHash: HashBytes(colorMipSig),
			ChromaSignatureHash: HashBytes(chromaMipSig),
			ChromaDetailSignatureHash: HashBytes(chromaDetailSig));
	}

	private static double[][] ToJaggedArray(double[,] array)
	{
		int rows = array.GetLength(0);
		int cols = array.GetLength(1);
		double[][] result = new double[rows][];
		for (int i = 0; i < rows; i++)
		{
			result[i] = new double[cols];
			for (int j = 0; j < cols; j++)
				result[i][j] = array[i, j];
		}
		return result;
	}

	public static byte[] RgbaToGrayscale(byte[] rgba, int width, int height)
	{
		byte[] gray = new byte[width * height];
		for (int i = 0; i < width * height; i++)
		{
			int idx = i * 4;
			gray[i] = (byte)((rgba[idx] * 77 + rgba[idx + 1] * 151 + rgba[idx + 2] * 28) >> 8);
		}
		return gray;
	}

	public static void ComputeLuminanceStats(byte[] gray, out double mean, out double stdDev)
	{
		long sum = 0;
		for (int i = 0; i < gray.Length; i++)
			sum += gray[i];
		mean = (double)sum / gray.Length;

		double variance = 0;
		for (int i = 0; i < gray.Length; i++)
		{
			double d = gray[i] - mean;
			variance += d * d;
		}
		variance /= gray.Length;
		stdDev = Math.Sqrt(variance);
	}

	public static byte[] ComputeMipSignature(byte[] gray, int width, int height, int targetSize)
	{
		double scaleX = (double)width / targetSize;
		double scaleY = (double)height / targetSize;
		byte[] sig = new byte[targetSize * targetSize];

		for (int sy = 0; sy < targetSize; sy++)
		{
			for (int sx = 0; sx < targetSize; sx++)
			{
				int srcX = (int)(sx * scaleX);
				int srcY = (int)(sy * scaleY);
				srcX = Math.Min(srcX, width - 1);
				srcY = Math.Min(srcY, height - 1);
				sig[sy * targetSize + sx] = gray[srcY * width + srcX];
			}
		}

		return sig;
	}

	public static byte[] ComputeColorMipSignature(byte[] rgba, int width, int height, int targetSize)
	{
		return ComputeRgbCellSignature(rgba, width, height, targetSize, normalizeChroma: false);
	}

	public static byte[] ComputeChromaMipSignature(byte[] rgba, int width, int height, int targetSize)
	{
		return ComputeRgbCellSignature(rgba, width, height, targetSize, normalizeChroma: true);
	}

	public static byte[] ComputeChromaDetailSignature(byte[] rgba, int width, int height, int targetSize)
	{
		byte[] chroma = ComputeChromaMipSignature(rgba, width, height, targetSize);
		byte[] detail = new byte[chroma.Length];

		for (int y = 0; y < targetSize; y++)
		{
			for (int x = 0; x < targetSize; x++)
			{
				double[] localMean = [0.0, 0.0, 0.0];
				int count = 0;
				for (int ky = -1; ky <= 1; ky++)
				{
					int sy = Math.Clamp(y + ky, 0, targetSize - 1);
					for (int kx = -1; kx <= 1; kx++)
					{
						int sx = Math.Clamp(x + kx, 0, targetSize - 1);
						int sourceIdx = ((sy * targetSize) + sx) * 3;
						localMean[0] += chroma[sourceIdx + 0];
						localMean[1] += chroma[sourceIdx + 1];
						localMean[2] += chroma[sourceIdx + 2];
						count++;
					}
				}

				localMean[0] /= count;
				localMean[1] /= count;
				localMean[2] /= count;

				int idx = ((y * targetSize) + x) * 3;
				for (int channel = 0; channel < 3; channel++)
				{
					double residual = chroma[idx + channel] - localMean[channel];
					detail[idx + channel] = ToByte(128.0 + (residual * 2.0));
				}
			}
		}

		return detail;
	}

	private static byte[] ComputeRgbCellSignature(byte[] rgba, int width, int height, int targetSize, bool normalizeChroma)
	{
		byte[] sig = new byte[targetSize * targetSize * 3];
		double scaleX = (double)width / targetSize;
		double scaleY = (double)height / targetSize;

		for (int sy = 0; sy < targetSize; sy++)
		{
			int y0 = Math.Clamp((int)Math.Floor(sy * scaleY), 0, height - 1);
			int y1 = Math.Min(height, Math.Max(y0 + 1, (int)Math.Ceiling((sy + 1) * scaleY)));
			for (int sx = 0; sx < targetSize; sx++)
			{
				int x0 = Math.Clamp((int)Math.Floor(sx * scaleX), 0, width - 1);
				int x1 = Math.Min(width, Math.Max(x0 + 1, (int)Math.Ceiling((sx + 1) * scaleX)));
				long rSum = 0;
				long gSum = 0;
				long bSum = 0;
				long count = 0;

				for (int y = y0; y < y1; y++)
				{
					for (int x = x0; x < x1; x++)
					{
						int idx = ((y * width) + x) * 4;
						rSum += rgba[idx + 0];
						gSum += rgba[idx + 1];
						bSum += rgba[idx + 2];
						count++;
					}
				}

				double r = count > 0 ? rSum / (double)count : 0.0;
				double g = count > 0 ? gSum / (double)count : 0.0;
				double b = count > 0 ? bSum / (double)count : 0.0;
				if (normalizeChroma)
				{
					double total = r + g + b;
					if (total > 1.0e-6)
					{
						r = 255.0 * r / total;
						g = 255.0 * g / total;
						b = 255.0 * b / total;
					}
					else
					{
						r = 85.0;
						g = 85.0;
						b = 85.0;
					}
				}

				int outIdx = ((sy * targetSize) + sx) * 3;
				sig[outIdx + 0] = ToByte(r);
				sig[outIdx + 1] = ToByte(g);
				sig[outIdx + 2] = ToByte(b);
			}
		}

		return sig;
	}

	public static void ComputeColorStats(
		byte[] rgba,
		int width,
		int height,
		out double[] meanRgb,
		out double[] rgbStdDev,
		out double meanHueDegrees,
		out double meanSaturation,
		out double colorfulness,
		out string meanColorHex)
	{
		int pixelCount = Math.Max(1, width * height);
		double rSum = 0;
		double gSum = 0;
		double bSum = 0;
		double rgSum = 0;
		double ybSum = 0;
		double rgSqSum = 0;
		double ybSqSum = 0;
		double hueSinSum = 0;
		double hueCosSum = 0;
		double saturationSum = 0;

		for (int i = 0; i < pixelCount; i++)
		{
			int idx = i * 4;
			double r = rgba[idx + 0];
			double g = rgba[idx + 1];
			double b = rgba[idx + 2];
			rSum += r;
			gSum += g;
			bSum += b;

			double rg = r - g;
			double yb = 0.5 * (r + g) - b;
			rgSum += rg;
			ybSum += yb;
			rgSqSum += rg * rg;
			ybSqSum += yb * yb;

			(double hueDegrees, double saturation) = RgbToHueSaturation(r, g, b);
			double radians = hueDegrees * Math.PI / 180.0;
			hueSinSum += Math.Sin(radians) * saturation;
			hueCosSum += Math.Cos(radians) * saturation;
			saturationSum += saturation;
		}

		double rMean = rSum / pixelCount;
		double gMean = gSum / pixelCount;
		double bMean = bSum / pixelCount;
		meanRgb = [rMean, gMean, bMean];
		meanColorHex = $"#{ToByte(rMean):X2}{ToByte(gMean):X2}{ToByte(bMean):X2}";

		double rVar = 0;
		double gVar = 0;
		double bVar = 0;
		for (int i = 0; i < pixelCount; i++)
		{
			int idx = i * 4;
			rVar += Math.Pow(rgba[idx + 0] - rMean, 2);
			gVar += Math.Pow(rgba[idx + 1] - gMean, 2);
			bVar += Math.Pow(rgba[idx + 2] - bMean, 2);
		}
		rgbStdDev = [Math.Sqrt(rVar / pixelCount), Math.Sqrt(gVar / pixelCount), Math.Sqrt(bVar / pixelCount)];

		double hueRadians = Math.Atan2(hueSinSum, hueCosSum);
		meanHueDegrees = hueRadians * 180.0 / Math.PI;
		if (meanHueDegrees < 0)
			meanHueDegrees += 360.0;
		meanSaturation = saturationSum / pixelCount;

		double rgMean = rgSum / pixelCount;
		double ybMean = ybSum / pixelCount;
		double rgStd = Math.Sqrt(Math.Max(0.0, (rgSqSum / pixelCount) - (rgMean * rgMean)));
		double ybStd = Math.Sqrt(Math.Max(0.0, (ybSqSum / pixelCount) - (ybMean * ybMean)));
		colorfulness = Math.Sqrt((rgStd * rgStd) + (ybStd * ybStd)) + (0.3 * Math.Sqrt((rgMean * rgMean) + (ybMean * ybMean)));
	}

	public static string[] ExtractDominantColors(byte[] rgba, int width, int height, int maxColors)
	{
		Dictionary<int, (long R, long G, long B, int Count)> buckets = [];
		int pixelCount = Math.Max(1, width * height);
		for (int i = 0; i < pixelCount; i++)
		{
			int idx = i * 4;
			byte r = rgba[idx + 0];
			byte g = rgba[idx + 1];
			byte b = rgba[idx + 2];
			int key = ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4);
			buckets.TryGetValue(key, out (long R, long G, long B, int Count) bucket);
			buckets[key] = (bucket.R + r, bucket.G + g, bucket.B + b, bucket.Count + 1);
		}

		return buckets.Values
			.OrderByDescending(bucket => bucket.Count)
			.ThenByDescending(bucket => bucket.R + bucket.G + bucket.B)
			.Take(Math.Max(0, maxColors))
			.Select(bucket =>
			{
				double count = Math.Max(1, bucket.Count);
				return $"#{ToByte(bucket.R / count):X2}{ToByte(bucket.G / count):X2}{ToByte(bucket.B / count):X2}";
			})
			.ToArray();
	}

	private static double ComputeDetailEnergy(byte[] detailSignature)
	{
		if (detailSignature.Length == 0)
			return 0.0;

		double sum = 0.0;
		for (int i = 0; i < detailSignature.Length; i++)
		{
			double centered = detailSignature[i] - 128.0;
			sum += Math.Abs(centered);
		}

		return sum / detailSignature.Length;
	}

	private static (double HueDegrees, double Saturation) RgbToHueSaturation(double rByte, double gByte, double bByte)
	{
		double r = rByte / 255.0;
		double g = gByte / 255.0;
		double b = bByte / 255.0;
		double max = Math.Max(r, Math.Max(g, b));
		double min = Math.Min(r, Math.Min(g, b));
		double delta = max - min;
		double hue = 0.0;

		if (delta > 1.0e-9)
		{
			if (Math.Abs(max - r) < 1.0e-9)
				hue = 60.0 * (((g - b) / delta) % 6.0);
			else if (Math.Abs(max - g) < 1.0e-9)
				hue = 60.0 * (((b - r) / delta) + 2.0);
			else
				hue = 60.0 * (((r - g) / delta) + 4.0);
		}

		if (hue < 0.0)
			hue += 360.0;

		double saturation = max <= 1.0e-9 ? 0.0 : delta / max;
		return (hue, saturation);
	}

	private static string HashBytes(byte[] bytes)
	{
		return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
	}

	private static byte ToByte(double value)
	{
		return (byte)Math.Clamp((int)Math.Round(value), 0, 255);
	}

	public static string ClassifyPatternScale(int tileX, int tileY, double periodicity, DominantFrequency[] freqs)
	{
		if (periodicity < 0.15 || freqs.Length == 0)
			return "micro";

		int maxDim = Math.Max(tileX, tileY);
		if (maxDim <= 16)
			return "micro";
		if (maxDim <= 48)
			return "meso";
		return "macro";
	}

	public static string ClassifyEdgeBehavior(double[,] autocorr, int width, int height)
	{
		int maxY = autocorr.GetLength(0);
		int maxX = autocorr.GetLength(1);
		double centerSum = 0, edgeSum = 0;
		int centerCount = 0, edgeCount = 0;

		for (int y = 0; y < maxY; y++)
		{
			for (int x = 0; x < maxX; x++)
			{
				bool isCenter = x >= maxX / 3 && x < 2 * maxX / 3 && y >= maxY / 3 && y < 2 * maxY / 3;
				if (isCenter) { centerSum += autocorr[y, x]; centerCount++; }
				else { edgeSum += autocorr[y, x]; edgeCount++; }
			}
		}

		if (centerCount == 0 || edgeCount == 0)
			return "uniform";

		double centerAvg = centerSum / centerCount;
		double edgeAvg = edgeSum / edgeCount;

		if (centerAvg > edgeAvg * 1.2)
			return "center_highlight";
		if (edgeAvg > centerAvg * 1.2)
			return "edge_darkening";
		return "uniform";
	}

	public static double[,] ComputeAutocorrelation2D(byte[] gray, int width, int height)
	{
		double mean = gray.Average(b => (double)b);
		double[] centered = gray.Select(b => (double)b - mean).ToArray();

		int maxShiftX = Math.Min(width / 2, 64);
		int maxShiftY = Math.Min(height / 2, 64);
		double[,] result = new double[maxShiftY, maxShiftX];

		double variance = centered.Sum(v => v * v);

		for (int dy = 0; dy < maxShiftY; dy++)
		{
			for (int dx = 0; dx < maxShiftX; dx++)
			{
				double sum = 0;
				int count = 0;
				for (int y = 0; y < height - dy; y++)
				{
					for (int x = 0; x < width - dx; x++)
					{
						sum += centered[y * width + x] * centered[(y + dy) * width + (x + dx)];
						count++;
					}
				}
				result[dy, dx] = variance > 0 ? sum / (variance * count / (width * height)) : 0;
			}
		}

		return result;
	}

	public static DominantFrequency[] ExtractDominantFrequencies(byte[] gray, int width, int height)
	{
		Complex[] spectrum = ComputeFFT2D(gray, width, height);
		double[,] magnitude = new double[height, width];

		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
				magnitude[y, x] = spectrum[y * width + x].Magnitude;

		List<DominantFrequency> peaks = [];
		int cy = height / 2;
		int cx = width / 2;

		for (int y = 1; y < height - 1; y++)
		{
			for (int x = 1; x < width - 1; x++)
			{
				if (x == cx && y == cy) continue;
				double val = magnitude[y, x];
				if (val > magnitude[y - 1, x] && val > magnitude[y + 1, x] &&
					val > magnitude[y, x - 1] && val > magnitude[y, x + 1] &&
					val > magnitude[y - 1, x - 1] && val > magnitude[y + 1, x + 1] &&
					val > magnitude[y - 1, x + 1] && val > magnitude[y + 1, x - 1])
				{
					double fx = (double)(x - cx) / width;
					double fy = (double)(y - cy) / height;
					peaks.Add(new DominantFrequency(fx, fy, val));
				}
			}
		}

		return peaks.OrderByDescending(p => p.Magnitude).Take(10).ToArray();
	}

	public static Complex[] ComputeFFT2D(byte[] gray, int width, int height)
	{
		Complex[] buffer = new Complex[width * height];

		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
				buffer[y * width + x] = new Complex(gray[y * width + x] * (1 - 2 * ((x + y) % 2)), 0);

		for (int y = 0; y < height; y++)
		{
			Complex[] row = new Complex[width];
			Array.Copy(buffer, y * width, row, 0, width);
			Fourier.Forward(row, FourierOptions.Default);
			Array.Copy(row, 0, buffer, y * width, width);
		}

		for (int x = 0; x < width; x++)
		{
			Complex[] col = new Complex[height];
			for (int y = 0; y < height; y++)
				col[y] = buffer[y * width + x];
			Fourier.Forward(col, FourierOptions.Default);
			for (int y = 0; y < height; y++)
				buffer[y * width + x] = col[y];
		}

		return buffer;
	}

	public static double[,] ComputeMagnitudeSpectrum(byte[] gray, int width, int height)
	{
		Complex[] spectrum = ComputeFFT2D(gray, width, height);
		int mw = width / 2;
		int mh = height / 2;
		double[,] mag = new double[mh, mw];

		for (int y = 0; y < mh; y++)
			for (int x = 0; x < mw; x++)
				mag[y, x] = spectrum[(y + height / 2) % height * width + (x + width / 2) % width].Magnitude;

		return mag;
	}

	public static (int TileSizeX, int TileSizeY, double PeriodicityScore) DetectTileSize(double[,] autocorr, int width, int height)
	{
		int maxY = autocorr.GetLength(0);
		int maxX = autocorr.GetLength(1);

		int bestX = 1, bestY = 1;
		double bestScore = 0;

		for (int dy = 4; dy < maxY; dy++)
		{
			for (int dx = 4; dx < maxX; dx++)
			{
				double peakVal = autocorr[dy, dx];
				if (peakVal > 0.3)
				{
					double score = peakVal * (1 - Math.Abs(dx - dy) / (double)Math.Max(dx, dy));
					if (score > bestScore)
					{
						bestScore = score;
						bestX = dx;
						bestY = dy;
					}
				}
			}
		}

		return (bestX, bestY, bestScore);
	}

	public static (int StartX, int StartY, int StampWidth, int StampHeight) ExtractPatternBounds(double[,] autocorr, int width, int height)
	{
		int maxY = autocorr.GetLength(0);
		int maxX = autocorr.GetLength(1);
		int firstPeakX = -1, firstPeakY = -1;

		for (int dy = 4; dy < maxY && firstPeakY < 0; dy++)
			for (int dx = 4; dx < maxX && firstPeakX < 0; dx++)
				if (autocorr[dy, dx] > 0.4)
				{
					firstPeakX = dx;
					firstPeakY = dy;
				}

		if (firstPeakX < 0 || firstPeakY < 0)
			return (0, 0, width, height);

		double halfX = firstPeakX / 2.0;
		double halfY = firstPeakY / 2.0;
		int startX = Math.Max(0, (int)(halfX - firstPeakX / 4));
		int startY = Math.Max(0, (int)(halfY - firstPeakY / 4));
		int stampW = Math.Min(firstPeakX, width - startX);
		int stampH = Math.Min(firstPeakY, height - startY);

		return (startX, startY, stampW, stampH);
	}
}
