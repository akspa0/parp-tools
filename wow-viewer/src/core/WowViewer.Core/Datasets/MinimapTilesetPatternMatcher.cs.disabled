namespace WowViewer.Core.Datasets;

public sealed record TilesetPatternCandidate(
	string Id,
	string TextureName,
	string DesignKit,
	string EraTag,
	PatternStamp Stamp);

public sealed record MinimapCellSignature(
	string MeanColorHex,
	string[] DominantColorsHex,
	string ChromaSignatureHash,
	string ChromaDetailSignatureHash,
	double ChromaDetailEnergy,
	double[] MeanRgb);

public sealed record MinimapTilesetMatch(
	string CandidateId,
	string TextureName,
	string DesignKit,
	string EraTag,
	double Score,
	double MeanColorDistance,
	double ChromaDistance,
	double ChromaDetailDistance,
	double PaletteDistance,
	double DetailEnergyDistance,
	string MeanColorHex,
	string ChromaSignatureHash,
	string ChromaDetailSignatureHash);

public sealed record MinimapTilesetCell(
	int CellX,
	int CellY,
	int PixelX,
	int PixelY,
	int Width,
	int Height,
	MinimapCellSignature Observed,
	IReadOnlyList<MinimapTilesetMatch> Candidates);

public sealed record MinimapTilesetDecomposition(
	int Width,
	int Height,
	int GridSizeX,
	int GridSizeY,
	IReadOnlyList<MinimapTilesetCell> Cells);

public static class MinimapTilesetPatternMatcher
{
	public static MinimapTilesetDecomposition Decompose(
		byte[] rgba,
		int width,
		int height,
		IReadOnlyList<TilesetPatternCandidate> patterns,
		int gridSize = 16,
		int maxCandidates = 3)
	{
		if (rgba.Length < width * height * 4)
			throw new ArgumentException("RGBA buffer is smaller than width * height * 4.", nameof(rgba));
		if (width <= 0 || height <= 0)
			throw new ArgumentOutOfRangeException(nameof(width), "Image dimensions must be positive.");
		if (gridSize <= 0)
			throw new ArgumentOutOfRangeException(nameof(gridSize), "Grid size must be positive.");
		if (maxCandidates <= 0)
			throw new ArgumentOutOfRangeException(nameof(maxCandidates), "Candidate count must be positive.");
		if (patterns.Count == 0)
			throw new ArgumentException("At least one tileset pattern candidate is required.", nameof(patterns));

		List<MinimapTilesetCell> cells = new(gridSize * gridSize);
		for (int gy = 0; gy < gridSize; gy++)
		{
			int y0 = (int)Math.Floor(gy * height / (double)gridSize);
			int y1 = (int)Math.Floor((gy + 1) * height / (double)gridSize);
			y1 = Math.Max(y0 + 1, Math.Min(height, y1));

			for (int gx = 0; gx < gridSize; gx++)
			{
				int x0 = (int)Math.Floor(gx * width / (double)gridSize);
				int x1 = (int)Math.Floor((gx + 1) * width / (double)gridSize);
				x1 = Math.Max(x0 + 1, Math.Min(width, x1));

				int cellWidth = x1 - x0;
				int cellHeight = y1 - y0;
				byte[] cellRgba = ExtractCell(rgba, width, x0, y0, cellWidth, cellHeight);
				PatternStamp observed = PatternMiner.AnalyzeTexture(cellRgba, cellWidth, cellHeight, $"cell_{gx}_{gy}");
				MinimapCellSignature signature = new(
					MeanColorHex: observed.MeanColorHex,
					DominantColorsHex: observed.DominantColorsHex,
					ChromaSignatureHash: observed.ChromaSignatureHash,
					ChromaDetailSignatureHash: observed.ChromaDetailSignatureHash,
					ChromaDetailEnergy: observed.ChromaDetailEnergy,
					MeanRgb: observed.MeanRgb);

				List<MinimapTilesetMatch> matches = patterns
					.Select(candidate => ScoreCandidate(observed, candidate))
					.OrderByDescending(match => match.Score)
					.ThenBy(match => match.TextureName, StringComparer.OrdinalIgnoreCase)
					.Take(maxCandidates)
					.ToList();

				cells.Add(new MinimapTilesetCell(
					CellX: gx,
					CellY: gy,
					PixelX: x0,
					PixelY: y0,
					Width: cellWidth,
					Height: cellHeight,
					Observed: signature,
					Candidates: matches));
			}
		}

		return new MinimapTilesetDecomposition(width, height, gridSize, gridSize, cells);
	}

	private static MinimapTilesetMatch ScoreCandidate(PatternStamp observed, TilesetPatternCandidate candidate)
	{
		PatternStamp stamp = candidate.Stamp;
		double colorDistance = RgbDistance(observed.MeanRgb, stamp.MeanRgb);
		double chromaDistance = ByteDistance(observed.ChromaMipSignature, stamp.ChromaMipSignature);
		double detailDistance = ByteDistance(observed.ChromaDetailSignature, stamp.ChromaDetailSignature);
		double paletteDistance = PaletteDistance(observed.DominantColorsHex, stamp.DominantColorsHex);
		double detailEnergyDistance = Math.Min(1.0, Math.Abs(observed.ChromaDetailEnergy - stamp.ChromaDetailEnergy) / 64.0);

		double combined =
			(0.24 * colorDistance) +
			(0.30 * chromaDistance) +
			(0.30 * detailDistance) +
			(0.10 * paletteDistance) +
			(0.06 * detailEnergyDistance);
		double score = Math.Clamp(1.0 - combined, 0.0, 1.0);

		return new MinimapTilesetMatch(
			CandidateId: candidate.Id,
			TextureName: candidate.TextureName,
			DesignKit: candidate.DesignKit,
			EraTag: candidate.EraTag,
			Score: score,
			MeanColorDistance: colorDistance,
			ChromaDistance: chromaDistance,
			ChromaDetailDistance: detailDistance,
			PaletteDistance: paletteDistance,
			DetailEnergyDistance: detailEnergyDistance,
			MeanColorHex: stamp.MeanColorHex,
			ChromaSignatureHash: stamp.ChromaSignatureHash,
			ChromaDetailSignatureHash: stamp.ChromaDetailSignatureHash);
	}

	private static byte[] ExtractCell(byte[] rgba, int sourceWidth, int x0, int y0, int width, int height)
	{
		byte[] cell = new byte[width * height * 4];
		for (int y = 0; y < height; y++)
		{
			int sourceOffset = (((y0 + y) * sourceWidth) + x0) * 4;
			int targetOffset = y * width * 4;
			Array.Copy(rgba, sourceOffset, cell, targetOffset, width * 4);
		}

		return cell;
	}

	private static double ByteDistance(byte[] a, byte[] b)
	{
		if (a.Length == 0 || b.Length == 0)
			return 1.0;

		int count = Math.Min(a.Length, b.Length);
		double total = 0.0;
		for (int i = 0; i < count; i++)
			total += Math.Abs(a[i] - b[i]) / 255.0;

		double mismatchPenalty = Math.Abs(a.Length - b.Length) / (double)Math.Max(a.Length, b.Length);
		return Math.Clamp((total / count) + (0.25 * mismatchPenalty), 0.0, 1.0);
	}

	private static double RgbDistance(double[] a, double[] b)
	{
		if (a.Length < 3 || b.Length < 3)
			return 1.0;

		double dr = a[0] - b[0];
		double dg = a[1] - b[1];
		double db = a[2] - b[2];
		return Math.Clamp(Math.Sqrt((dr * dr) + (dg * dg) + (db * db)) / 441.67295593, 0.0, 1.0);
	}

	private static double PaletteDistance(string[] observed, string[] candidate)
	{
		if (observed.Length == 0 || candidate.Length == 0)
			return 1.0;

		double total = 0.0;
		foreach (string color in observed.Take(4))
		{
			(byte r, byte g, byte b) = ParseHexColor(color);
			double best = 1.0;
			foreach (string other in candidate.Take(4))
			{
				(byte or, byte og, byte ob) = ParseHexColor(other);
				double dr = r - or;
				double dg = g - og;
				double db = b - ob;
				double distance = Math.Sqrt((dr * dr) + (dg * dg) + (db * db)) / 441.67295593;
				best = Math.Min(best, distance);
			}

			total += best;
		}

		return Math.Clamp(total / Math.Min(4, observed.Length), 0.0, 1.0);
	}

	private static (byte R, byte G, byte B) ParseHexColor(string hex)
	{
		if (hex.Length != 7 || hex[0] != '#')
			return (0, 0, 0);

		return (
			Convert.ToByte(hex.Substring(1, 2), 16),
			Convert.ToByte(hex.Substring(3, 2), 16),
			Convert.ToByte(hex.Substring(5, 2), 16));
	}
}
