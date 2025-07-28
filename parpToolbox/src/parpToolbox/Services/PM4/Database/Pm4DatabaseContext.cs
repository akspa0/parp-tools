using Microsoft.EntityFrameworkCore;
using System.Text.Json;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Entity Framework DbContext for PM4 data analysis and object extraction.
    /// Uses SQLite as an embedded database for complex hierarchical queries.
    /// </summary>
    public class Pm4DatabaseContext : DbContext
    {
        private readonly string _databasePath;
        
        public Pm4DatabaseContext(string databasePath)
        {
            _databasePath = databasePath;
        }
        
        // DbSets for all PM4 entities
        public DbSet<Pm4File> Files { get; set; } = null!;
        public DbSet<Pm4Vertex> Vertices { get; set; } = null!;
        public DbSet<Pm4Triangle> Triangles { get; set; } = null!;
        public DbSet<Pm4TriangleVertex> TriangleVertices { get; set; } = null!;
        public DbSet<Pm4Surface> Surfaces { get; set; } = null!;
        public DbSet<Pm4Link> Links { get; set; } = null!;
        public DbSet<Pm4Placement> Placements { get; set; } = null!;
        public DbSet<Pm4SurfaceGroup> SurfaceGroups { get; set; } = null!;
        public DbSet<Pm4SurfaceGroupMember> SurfaceGroupMembers { get; set; } = null!;
        public DbSet<Pm4RawChunk> RawChunks { get; set; } = null!;
        
        // Hierarchical container decoding entities
        public DbSet<Pm4HierarchicalContainer> HierarchicalContainers { get; set; } = null!;
        public DbSet<Pm4HierarchicalContainerMember> HierarchicalContainerMembers { get; set; } = null!;
        
        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseSqlite($"Data Source={_databasePath}");
            
            // Enable detailed logging for debugging
            optionsBuilder.EnableSensitiveDataLogging();
            optionsBuilder.EnableDetailedErrors();
        }
        
        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);
            
            // Configure PM4 File
            modelBuilder.Entity<Pm4File>(entity =>
            {
                entity.HasKey(e => e.Id);
                entity.HasIndex(e => e.FileName);
                entity.HasIndex(e => e.FilePath);
                entity.Property(e => e.ProcessedAt).HasDefaultValueSql("datetime('now')");
            });
            
            // Configure Vertices with spatial indexing
            modelBuilder.Entity<Pm4Vertex>(entity =>
            {
                entity.HasKey(e => e.Id);
                entity.HasIndex(e => new { e.Pm4FileId, e.GlobalIndex });
                entity.HasIndex(e => new { e.X, e.Y, e.Z }); // Spatial queries
                entity.HasIndex(e => e.ChunkType);
                
                entity.HasOne(e => e.Pm4File)
                      .WithMany(f => f.Vertices)
                      .HasForeignKey(e => e.Pm4FileId)
                      .OnDelete(DeleteBehavior.Cascade);
            });
            
            // Configure Triangles
            modelBuilder.Entity<Pm4Triangle>(entity =>
            {
                entity.HasKey(e => e.Id);
                entity.HasIndex(e => new { e.Pm4FileId, e.GlobalIndex });
                entity.HasIndex(e => new { e.VertexA, e.VertexB, e.VertexC });
                
                entity.HasOne(e => e.Pm4File)
                      .WithMany(f => f.Triangles)
                      .HasForeignKey(e => e.Pm4FileId)
                      .OnDelete(DeleteBehavior.Cascade);
            });
            
            // Configure Triangle-Vertex relationships
            modelBuilder.Entity<Pm4TriangleVertex>(entity =>
            {
                entity.HasKey(e => e.Id);
                entity.HasIndex(e => new { e.TriangleId, e.VertexId, e.VertexPosition });
                
                entity.HasOne(e => e.Triangle)
                      .WithMany(t => t.TriangleVertices)
                      .HasForeignKey(e => e.TriangleId)
                      .OnDelete(DeleteBehavior.Cascade);
                      
                entity.HasOne(e => e.Vertex)
                      .WithMany(v => v.TriangleVertices)
                      .HasForeignKey(e => e.VertexId)
                      .OnDelete(DeleteBehavior.Cascade);
            });
            
            // Configure Surfaces with spatial indexing
            modelBuilder.Entity<Pm4Surface>(entity =>
            {
                entity.HasKey(e => e.Id);
                entity.HasIndex(e => new { e.Pm4FileId, e.GlobalIndex });
                entity.HasIndex(e => e.GroupKey);
                entity.HasIndex(e => new { e.BoundsCenterX, e.BoundsCenterY, e.BoundsCenterZ }); // Spatial clustering
                entity.HasIndex(e => new { e.MsviFirstIndex, e.IndexCount });
                
                entity.HasOne(e => e.Pm4File)
                      .WithMany(f => f.Surfaces)
                      .HasForeignKey(e => e.Pm4FileId)
                      .OnDelete(DeleteBehavior.Cascade);
            });
            
            // Configure Links with hierarchical indexing
            modelBuilder.Entity<Pm4Link>(entity =>
            {
                entity.HasKey(e => e.Id);
                entity.HasIndex(e => new { e.Pm4FileId, e.GlobalIndex });
                entity.HasIndex(e => e.ParentIndex); // Hierarchical queries
                entity.HasIndex(e => e.ReferenceIndex);
                entity.HasIndex(e => new { e.MspiFirstIndex, e.MspiIndexCount });
                
                entity.HasOne(e => e.Pm4File)
                      .WithMany(f => f.Links)
                      .HasForeignKey(e => e.Pm4FileId)
                      .OnDelete(DeleteBehavior.Cascade);
            });
            
            // Configure Placements with spatial indexing
            modelBuilder.Entity<Pm4Placement>(entity =>
            {
                entity.HasKey(e => e.Id);
                entity.HasIndex(e => new { e.Pm4FileId, e.GlobalIndex });
                entity.HasIndex(e => new { e.PositionX, e.PositionY, e.PositionZ }); // Spatial queries
                entity.HasIndex(e => e.Unknown4); // Links to MSLK.ParentIndex
                
                entity.HasOne(e => e.Pm4File)
                      .WithMany(f => f.Placements)
                      .HasForeignKey(e => e.Pm4FileId)
                      .OnDelete(DeleteBehavior.Cascade);
            });
            
            // Configure Surface Groups
            modelBuilder.Entity<Pm4SurfaceGroup>(entity =>
            {
                entity.HasKey(e => e.Id);
                entity.HasIndex(e => new { e.Pm4FileId, e.GroupIndex });
                entity.HasIndex(e => e.ClusteringMethod);
                entity.HasIndex(e => new { e.BoundsCenterX, e.BoundsCenterY, e.BoundsCenterZ });
                
                entity.HasOne(e => e.Pm4File)
                      .WithMany(f => f.SurfaceGroups)
                      .HasForeignKey(e => e.Pm4FileId)
                      .OnDelete(DeleteBehavior.Cascade);
            });
            
            // Configure Surface Group Members
            modelBuilder.Entity<Pm4SurfaceGroupMember>(entity =>
            {
                entity.HasKey(e => e.Id);
                entity.HasIndex(e => new { e.SurfaceGroupId, e.SurfaceId });
                
                entity.HasOne(e => e.SurfaceGroup)
                      .WithMany(g => g.Members)
                      .HasForeignKey(e => e.SurfaceGroupId)
                      .OnDelete(DeleteBehavior.Cascade);
                      
                entity.HasOne(e => e.Surface)
                      .WithMany(s => s.SurfaceGroupMembers)
                      .HasForeignKey(e => e.SurfaceId)
                      .OnDelete(DeleteBehavior.Cascade);
            });
        }
        
        /// <summary>
        /// Ensures the database is created and migrated.
        /// </summary>
        public async Task EnsureDatabaseCreatedAsync()
        {
            await Database.EnsureCreatedAsync();
        }
        
        /// <summary>
        /// Executes a raw SQL query for complex hierarchical analysis.
        /// </summary>
        public async Task<List<T>> ExecuteRawQueryAsync<T>(string sql, params object[] parameters) where T : class
        {
            return await Set<T>().FromSqlRaw(sql, parameters).ToListAsync();
        }
        
        /// <summary>
        /// Gets surface groups clustered by spatial proximity using SQL.
        /// </summary>
        public async Task<List<Pm4SurfaceGroup>> GetSpatiallyClusteredGroupsAsync(int pm4FileId, float proximityThreshold = 50.0f)
        {
            return await SurfaceGroups
                .Where(g => g.Pm4FileId == pm4FileId && g.ClusteringMethod == "Spatial")
                .Include(g => g.Members)
                    .ThenInclude(m => m.Surface)
                .ToListAsync();
        }
        
        /// <summary>
        /// Gets hierarchical object relationships using MPRL->MSLK links.
        /// </summary>
        public async Task<Dictionary<uint, List<Pm4Link>>> GetHierarchicalObjectsAsync(int pm4FileId)
        {
            var placements = await Placements
                .Where(p => p.Pm4FileId == pm4FileId)
                .ToListAsync();
                
            var links = await Links
                .Where(l => l.Pm4FileId == pm4FileId)
                .ToListAsync();
                
            var hierarchicalObjects = new Dictionary<uint, List<Pm4Link>>();
            
            foreach (var placement in placements)
            {
                var relatedLinks = links
                    .Where(l => l.ParentIndex == placement.Unknown4)
                    .ToList();
                    
                if (relatedLinks.Count > 0)
                {
                    hierarchicalObjects[placement.Unknown4] = relatedLinks;
                }
            }
            
            return hierarchicalObjects;
        }
    }
}
