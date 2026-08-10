using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class DatasetVersionCatalogTests
{
    [Fact]
    public void DiscoversVlmProjectAndIgnoresControlNpzs()
    {
        string root = CreateTempRoot();
        try
        {
            string project = Path.Combine(root, "v50.1", "0_5_3_3368-Kalimdor");
            Directory.CreateDirectory(Path.Combine(project, "dataset"));
            Directory.CreateDirectory(Path.Combine(project, "liquids"));
            Directory.CreateDirectory(Path.Combine(project, "textures"));
            File.WriteAllText(Path.Combine(project, "dataset", "Kalimdor_24_53.json"), "{}");

            string control = Path.Combine(root, "v60", "control-v1");
            Directory.CreateDirectory(control);
            File.WriteAllBytes(Path.Combine(control, "flat.npz"), Array.Empty<byte>());

            IReadOnlyList<DatasetVersionCatalogEntry> entries = DatasetVersionCatalog.Discover(root);

            DatasetVersionCatalogEntry entry = Assert.Single(entries);
            Assert.Equal(DatasetSourceKind.VlmProject, entry.SourceKind);
            Assert.True(entry.Renderable);
            Assert.Equal("Kalimdor", entry.MapName);
            Assert.Equal(1, entry.TileCount);
            Assert.Contains("liquid_maps", entry.Signals);
            Assert.Contains("texture_assets", entry.Signals);
        }
        finally
        {
            DeleteTempRoot(root);
        }
    }

    [Fact]
    public void DiscoversZarrSummaryAndReportsCurrentLiquidSignalsAsNonRenderable()
    {
        string root = CreateTempRoot();
        try
        {
            string store = Path.Combine(root, "v60", "0_5_3_3368-Azeroth.zarr");
            Directory.CreateDirectory(store);
            File.WriteAllText(Path.Combine(store, "zarr.json"), "{\"zarr_format\":3}");
            foreach (string array in new[] { "liquid_mask", "liquid_height", "liquid_type_256", "object_mask" })
            {
                string arrayDirectory = Path.Combine(store, array);
                Directory.CreateDirectory(arrayDirectory);
                File.WriteAllText(Path.Combine(arrayDirectory, "zarr.json"), "{}");
            }

            IReadOnlyList<DatasetVersionCatalogEntry> entries = DatasetVersionCatalog.Discover(root);

            DatasetVersionCatalogEntry entry = Assert.Single(entries);
            Assert.Equal(DatasetSourceKind.ZarrStore, entry.SourceKind);
            Assert.False(entry.Renderable);
            Assert.Contains("liquid_mask", entry.Signals);
            Assert.Contains("liquid_height", entry.Signals);
            Assert.Contains("liquid_type_256", entry.Signals);
            Assert.Contains("Zarr tile decoding", entry.Diagnostic);
        }
        finally
        {
            DeleteTempRoot(root);
        }
    }

    [Fact]
    public void DiscoversExplicitRealTileObservationFolderWithoutPromotingItToTerrain()
    {
        string root = CreateTempRoot();
        try
        {
            string observations = Path.Combine(root, "real-tile-observations");
            Directory.CreateDirectory(observations);
            File.WriteAllBytes(Path.Combine(observations, "Elwynn_24_53.jpg"), new byte[] { 1, 2, 3 });

            IReadOnlyList<DatasetVersionCatalogEntry> entries = DatasetVersionCatalog.Discover(root);
            DatasetVersionCatalogEntry entry = Assert.Single(entries);

            Assert.Equal(DatasetSourceKind.RealTileObservation, entry.SourceKind);
            Assert.False(entry.Renderable);
            Assert.Equal(1, entry.TileCount);
            Assert.Contains("real_rgb", entry.Signals);

            IReadOnlyList<RealTileObservation> observationsFound =
                DatasetVersionCatalog.DiscoverRealTileObservations(observations);
            RealTileObservation observation = Assert.Single(observationsFound);
            Assert.Equal(RealTileObservationKind.Unknown, observation.Kind);
            Assert.True(observation.UsableAsModelInput);
            Assert.False(observation.UsableAsTarget);
            Assert.Contains("normalization", observation.Diagnostic);
        }
        finally
        {
            DeleteTempRoot(root);
        }
    }

    private static string CreateTempRoot()
    {
        string root = Path.Combine(Path.GetTempPath(), "wow-viewer-dataset-catalog-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        return root;
    }

    private static void DeleteTempRoot(string root)
    {
        if (Directory.Exists(root))
            Directory.Delete(root, recursive: true);
    }
}
