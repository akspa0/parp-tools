using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
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
	double LuminanceStdDev);

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
			LuminanceStdDev: std);
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
