using WowViewer.Core.Datasets;

namespace WowViewer.Core.Tests;

public sealed class PatternMinerTests
{
	[Fact]
	public void AnalyzeTexture_SeparatesSameLuminancePatternByChromaSignature()
	{
		byte[] warm = BuildCheckerTexture(16, (255, 40, 0), (255, 140, 0));
		byte[] cool = BuildCheckerTexture(16, (0, 170, 0), (0, 254, 84));

		PatternStamp warmStamp = PatternMiner.AnalyzeTexture(warm, 16, 16, "warm");
		PatternStamp coolStamp = PatternMiner.AnalyzeTexture(cool, 16, 16, "cool");

		Assert.Equal(warmStamp.PatternSignatureHash, coolStamp.PatternSignatureHash);
		Assert.NotEqual(warmStamp.ChromaSignatureHash, coolStamp.ChromaSignatureHash);
		Assert.NotEqual(warmStamp.ChromaDetailSignatureHash, coolStamp.ChromaDetailSignatureHash);
		Assert.NotEqual(warmStamp.MeanColorHex, coolStamp.MeanColorHex);
		Assert.Equal(8 * 8 * 3, warmStamp.ColorMipSignature.Length);
		Assert.Equal(8 * 8 * 3, warmStamp.ChromaMipSignature.Length);
		Assert.Equal(16 * 16 * 3, warmStamp.ChromaDetailSignature.Length);
	}

	[Fact]
	public void ComputeChromaMipSignature_NormalizesAwayBrightness()
	{
		byte[] darkGreen = BuildSolidTexture(8, 20, 80, 20);
		byte[] brightGreen = BuildSolidTexture(8, 60, 240, 60);

		byte[] darkChroma = PatternMiner.ComputeChromaMipSignature(darkGreen, 8, 8, 4);
		byte[] brightChroma = PatternMiner.ComputeChromaMipSignature(brightGreen, 8, 8, 4);

		Assert.Equal(darkChroma, brightChroma);
	}

	[Fact]
	public void AnalyzeTexture_CapturesBakedColorDetailIndependentOfMeanTint()
	{
		byte[] checker = BuildCheckerTexture(16, (120, 80, 40), (40, 120, 80));
		byte[] stripes = BuildStripeTexture(16, (120, 80, 40), (40, 120, 80));

		PatternStamp checkerStamp = PatternMiner.AnalyzeTexture(checker, 16, 16, "checker");
		PatternStamp stripeStamp = PatternMiner.AnalyzeTexture(stripes, 16, 16, "stripes");

		Assert.Equal(checkerStamp.MeanColorHex, stripeStamp.MeanColorHex);
		Assert.NotEqual(checkerStamp.ChromaDetailSignatureHash, stripeStamp.ChromaDetailSignatureHash);
		Assert.True(checkerStamp.ChromaDetailEnergy > 0.0);
		Assert.True(stripeStamp.DominantColorsHex.Length >= 2);
	}

	[Fact]
	public void DecomposeMinimap_RanksColorDetailCandidatesPerCell()
	{
		byte[] grass = BuildCheckerTexture(16, (32, 120, 45), (68, 160, 72));
		byte[] rock = BuildStripeTexture(16, (120, 96, 72), (86, 72, 62));
		PatternStamp grassStamp = PatternMiner.AnalyzeTexture(grass, 16, 16, "grass_detail");
		PatternStamp rockStamp = PatternMiner.AnalyzeTexture(rock, 16, 16, "rock_detail");
		byte[] minimap = CombineHorizontal(grass, rock, 16, 16);

		MinimapTilesetDecomposition decomposition = MinimapTilesetPatternMatcher.Decompose(
			minimap,
			width: 32,
			height: 16,
			patterns:
			[
				new TilesetPatternCandidate("grass", "grass_detail", "dev", "alpha", grassStamp),
				new TilesetPatternCandidate("rock", "rock_detail", "dev", "alpha", rockStamp)
			],
			gridSize: 2,
			maxCandidates: 1);

		Assert.Equal("grass", decomposition.Cells.Single(c => c.CellX == 0 && c.CellY == 0).Candidates[0].CandidateId);
		Assert.Equal("rock", decomposition.Cells.Single(c => c.CellX == 1 && c.CellY == 0).Candidates[0].CandidateId);
	}

	private static byte[] BuildCheckerTexture(int size, (byte R, byte G, byte B) a, (byte R, byte G, byte B) b)
	{
		byte[] rgba = new byte[size * size * 4];
		for (int y = 0; y < size; y++)
		{
			for (int x = 0; x < size; x++)
			{
				(byte r, byte g, byte blue) = ((x / 4) + (y / 4)) % 2 == 0 ? a : b;
				int idx = ((y * size) + x) * 4;
				rgba[idx + 0] = r;
				rgba[idx + 1] = g;
				rgba[idx + 2] = blue;
				rgba[idx + 3] = 255;
			}
		}

		return rgba;
	}

	private static byte[] BuildSolidTexture(int size, byte r, byte g, byte b)
	{
		byte[] rgba = new byte[size * size * 4];
		for (int i = 0; i < size * size; i++)
		{
			int idx = i * 4;
			rgba[idx + 0] = r;
			rgba[idx + 1] = g;
			rgba[idx + 2] = b;
			rgba[idx + 3] = 255;
		}

		return rgba;
	}

	private static byte[] BuildStripeTexture(int size, (byte R, byte G, byte B) a, (byte R, byte G, byte B) b)
	{
		byte[] rgba = new byte[size * size * 4];
		for (int y = 0; y < size; y++)
		{
			for (int x = 0; x < size; x++)
			{
				(byte r, byte g, byte blue) = (x / 4) % 2 == 0 ? a : b;
				int idx = ((y * size) + x) * 4;
				rgba[idx + 0] = r;
				rgba[idx + 1] = g;
				rgba[idx + 2] = blue;
				rgba[idx + 3] = 255;
			}
		}

		return rgba;
	}

	private static byte[] CombineHorizontal(byte[] left, byte[] right, int tileWidth, int tileHeight)
	{
		byte[] rgba = new byte[tileWidth * 2 * tileHeight * 4];
		for (int y = 0; y < tileHeight; y++)
		{
			for (int x = 0; x < tileWidth; x++)
			{
				Array.Copy(left, ((y * tileWidth) + x) * 4, rgba, ((y * tileWidth * 2) + x) * 4, 4);
				Array.Copy(right, ((y * tileWidth) + x) * 4, rgba, ((y * tileWidth * 2) + tileWidth + x) * 4, 4);
			}
		}

		return rgba;
	}
}
