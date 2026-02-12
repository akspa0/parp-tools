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
class ObjectEntry:
    name: str
    asset_path: str
    designkit: str
    object_type: str
    embedding: np.ndarray
    instances: list[dict]

class ObjectLibrary:
    """
    Object embedding library for matching minimap crops to known objects.
    """
    
    def __init__(self, library_dir: Path, embedding_dim: int = 128):
        self.library_dir = library_dir
        self.embedding_dim = embedding_dim
        self.entries: dict[str, ObjectEntry] = {}
        self.index: faiss.IndexFlatIP = None
        self._embedder = None
    
    def _build_embedder(self) -> torch.nn.Module:
        """Build object embedder (ResNet50 for more detail)."""
        if self._embedder is not None:
             return self._embedder

        print("Loading ResNet50 for Object Embeddings...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove final FC
        embedder = torch.nn.Sequential(*list(resnet.children())[:-1])
        embedder.add_module('flatten', torch.nn.Flatten())
        embedder.add_module('proj', torch.nn.Linear(2048, self.embedding_dim))
        embedder.eval()
        self._embedder = embedder
        return embedder
    
    def compute_embedding(self, crop: Image.Image, mask: Image.Image = None) -> np.ndarray:
        """Compute object embedding from masked crop."""
        if self._embedder is None: self._build_embedder()
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        if mask:
            # Apply mask (black out non-object pixels)
            crop = crop.copy()
            # Ensure mask is same size
            if mask.size != crop.size:
                 mask = mask.resize(crop.size)
            
            mask_arr = np.array(mask.convert('L'))
            crop_arr = np.array(crop)
            # Ensure 3 channels
            if len(crop_arr.shape) == 2:
                 crop_arr = np.stack([crop_arr]*3, axis=-1)
                 
            # Masking
            mask_bool = mask_arr > 0
            crop_arr[~mask_bool] = 0
            crop = Image.fromarray(crop_arr)
        
        with torch.no_grad():
            tensor = transform(crop).unsqueeze(0)
            emb = self._embedder(tensor).numpy().flatten()
            return emb / (np.linalg.norm(emb) + 1e-8)
    
    def match_crop(self, crop: Image.Image, mask: Image.Image = None, k: int = 5) -> list[tuple[str, float]]:
        """Match a crop against the library."""
        if self.index is None or self.index.ntotal == 0:
             return []

        emb = self.compute_embedding(crop, mask)
        emb = emb.reshape(1, -1).astype('float32')
        D, I = self.index.search(emb, k)
        
        results = []
        names = list(self.entries.keys())
        for idx, score in zip(I[0], D[0]):
            if idx < len(names):
                results.append((names[idx], float(score)))
        return results

    def build_from_directory(self, limit: int = 1000):
        """Scan library directory for objects (simplified implementation)."""
        print(f"Building Object Library from {self.library_dir} (Limit: {limit})...")
        self._build_embedder()
        
        embeddings = []
        count = 0
        
        # Expecting directory structure: library_dir/ObjectName/crop.png
        for root, dirs, files in os.walk(self.library_dir):
             for dir_name in dirs:
                  obj_dir = Path(root) / dir_name
                  # Look for representative crop
                  crop_path = obj_dir / "crop_0.png"
                  if crop_path.exists():
                       try:
                            img = Image.open(crop_path).convert("RGB")
                            emb = self.compute_embedding(img)
                            
                            entry = ObjectEntry(
                                 name=dir_name,
                                 asset_path=f"World/wmo/{dir_name}.wmo", # Placeholder
                                 designkit="Unknown",
                                 object_type="wmo",
                                 embedding=emb,
                                 instances=[]
                            )
                            self.entries[dir_name] = entry
                            embeddings.append(emb)
                            count += 1
                       except Exception as e:
                            print(f"Failed to process object {dir_name}: {e}")
                  
                  if count >= limit: break
             if count >= limit: break

        if embeddings:
            embeddings_np = np.vstack(embeddings).astype('float32')
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings_np)
            print(f"Indexed {len(embeddings)} objects.")
        else:
             print("No objects found to index.")
