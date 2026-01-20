from dataclasses import dataclass
from pathlib import Path
import faiss
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

@dataclass
class TextureEntry:
    texture_id: int
    full_path: str
    relative_path: str
    designkit: str
    embedding: np.ndarray

class TextureLibrary:
    """
    Texture embedding library with FAISS nearest-neighbor search.
    Uses 16-dim embeddings for improved texture differentiation.
    """
    
    def __init__(self, tileset_dir: Path, embedding_dim: int = 16):
        self.tileset_dir = tileset_dir
        self.embedding_dim = embedding_dim
        self.entries: dict[str, TextureEntry] = {}
        self.index: faiss.IndexFlatIP = None
        self._embedder = None
    
    def _build_embedder(self) -> torch.nn.Module:
        """Build lightweight texture embedder (ResNet18 truncated)."""
        if self._embedder is not None:
            return self._embedder
            
        print("Loading ResNet18 for Texture Embeddings...")
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC, keep up to avgpool -> 512-dim
        embedder = torch.nn.Sequential(*list(resnet.children())[:-1])
        # Project to embedding_dim
        embedder.add_module('flatten', torch.nn.Flatten())
        embedder.add_module('proj', torch.nn.Linear(512, self.embedding_dim))
        embedder.eval()
        self._embedder = embedder
        return embedder
    
    def compute_embedding(self, img: Image.Image) -> np.ndarray:
        """Compute texture embedding from PIL image."""
        if self._embedder is None: self._build_embedder()
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            tensor = transform(img).unsqueeze(0)
            emb = self._embedder(tensor).numpy().flatten()
            return emb / (np.linalg.norm(emb) + 1e-8)  # L2 normalize
    
    def build_from_directory(self, limit: int = 1000):
        """Scan tileset directory and build embedding index."""
        print(f"Building Texture Library from {self.tileset_dir} (Limit: {limit})...")
        self._build_embedder() # Ensure loaded
        
        embeddings = []
        texture_id = 0
        
        count = 0
        for root, _, files in os.walk(self.tileset_dir):
            for file in files:
                if not file.lower().endswith(".png"): continue
                
                png_path = Path(root) / file
                try:
                    rel_path = png_path.relative_to(self.tileset_dir)
                    designkit = "/".join(rel_path.parts[:-1])
                    
                    img = Image.open(png_path).convert("RGB")
                    emb = self.compute_embedding(img)
                    
                    entry = TextureEntry(
                        texture_id=texture_id,
                        full_path=str(png_path),
                        relative_path=str(rel_path),
                        designkit=designkit,
                        embedding=emb
                    )
                    self.entries[str(rel_path)] = entry
                    embeddings.append(emb)
                    texture_id += 1
                    count += 1
                    
                    if count >= limit:
                        print(f"Hit limit of {limit} textures.")
                        break
                except Exception as e:
                    print(f"Skipping {png_path}: {e}")
            if count >= limit: break
        
        if embeddings:
            # Build FAISS index
            embeddings_np = np.vstack(embeddings).astype('float32')
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings_np)
            print(f"Indexed {len(embeddings)} textures.")
        else:
            print("No textures found to index.")
    
    def save(self, path: Path):
        """Save library to disk."""
        data = {
            "entries": {k: {
                "texture_id": v.texture_id,
                "full_path": v.full_path,
                "relative_path": v.relative_path,
                "designkit": v.designkit,
                "embedding": v.embedding.tolist()
            } for k, v in self.entries.items()},
            "embedding_dim": self.embedding_dim
        }
        path.write_text(json.dumps(data, indent=2))
        if self.index:
            faiss.write_index(self.index, str(path.with_suffix('.faiss')))
    
    def load(self, path: Path):
        """Load library from disk."""
        if not path.exists():
            print(f"Texture Library not found at {path}")
            return

        data = json.loads(path.read_text())
        self.embedding_dim = data["embedding_dim"]
        for k, v in data["entries"].items():
            self.entries[k] = TextureEntry(
                texture_id=v["texture_id"],
                full_path=v["full_path"],
                relative_path=v["relative_path"],
                designkit=v["designkit"],
                embedding=np.array(v["embedding"])
            )
        
        faiss_path = path.with_suffix('.faiss')
        if faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
