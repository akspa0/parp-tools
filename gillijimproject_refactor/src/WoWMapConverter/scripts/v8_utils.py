import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import struct
import os

def load_adt_bin(path):
    """
    Load VLM binary tile format (.bin).
    Returns dictionary with stitched global maps.
    """
    if not os.path.exists(path): return None
    
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'VLM1': return None
        version = struct.unpack('<i', f.read(4))[0]
        flags = struct.unpack('<i', f.read(4))[0]
        n_chunks = struct.unpack('<i', f.read(4))[0]

        # Offset Table (256 * 8 bytes)
        offsets = []
        for _ in range(256):
            o, s = struct.unpack('<ii', f.read(8))
            offsets.append((o, s))

        # Prepare Global Maps
        height_map = np.zeros((145, 145), dtype=np.float32)
        normal_map = np.zeros((145, 145, 3), dtype=np.float32) 
        mccv_map = np.zeros((145, 145, 4), dtype=np.uint8) 
        shadow_map = np.zeros((1024, 1024), dtype=np.uint8)
        alpha_maps = np.zeros((4, 1024, 1024), dtype=np.uint8)

        for i in range(256):
            off, size = offsets[i]
            if off == 0 or size == 0: continue
            
            f.seek(off)
            
            # Read Heights (145 floats = 580 bytes)
            h_bytes = f.read(145 * 4)
            if len(h_bytes) < 145 * 4:
                h_bytes = h_bytes + b'\x00' * (145 * 4 - len(h_bytes))
            h_vals = np.frombuffer(h_bytes, dtype=np.float32)

            # Read Normals (145 * 3 bytes = 435 bytes)
            n_bytes = f.read(145 * 3)
            if len(n_bytes) < 145 * 3:
                n_bytes = n_bytes + b'\x00' * (145 * 3 - len(n_bytes))
            n_vals = np.frombuffer(n_bytes, dtype=np.int8).reshape(145, 3) 

            # Read MCCV (145 * 4 bytes = 580 bytes)
            c_bytes = f.read(145 * 4)
            if len(c_bytes) < 145 * 4:
                c_bytes = c_bytes + b'\x80' * (145 * 4 - len(c_bytes))  # Default gray
            c_vals = np.frombuffer(c_bytes, dtype=np.uint8).reshape(145, 4)

            # Read Shadows (512 bytes)
            s_bytes = f.read(512)
            if len(s_bytes) < 512:
                s_bytes = s_bytes + b'\x00' * (512 - len(s_bytes))
            s_bits = np.unpackbits(np.frombuffer(s_bytes, dtype=np.uint8))
            s_bits = s_bits.reshape(64, 64)

            # Read Alphas (4 * 4096 bytes)
            a_bytes = f.read(4096 * 4)
            if len(a_bytes) < 4096 * 4:
                a_bytes = a_bytes + b'\x00' * (4096 * 4 - len(a_bytes))
            a_vals = np.frombuffer(a_bytes, dtype=np.uint8).reshape(4, 64, 64)

            # --- Stitching ---
            r = i // 16
            c = i % 16
            
            # Heights/Normals/MCCV (Vertex Grid)
            # Outer 9x9 indices: 0, 17, 34 ... 136 (start of each row)
            for row in range(9):
                src_idx = row * 17 # 0, 17, 34.. (17=9+8)
                vals_row = h_vals[src_idx : src_idx+9]
                
                # Dest pos
                dr = r * 8 + row
                dc = c * 8 
                
                height_map[dr, dc : dc+9] = vals_row
                
                # Normals
                normal_map[dr, dc : dc+9] = n_vals[src_idx : src_idx+9]
                
                # MCCV
                mccv_map[dr, dc : dc+9] = c_vals[src_idx : src_idx+9]

            # Shadows/Alpha (Pixel Grid)
            # Use chunks
            dr = r * 64
            dc = c * 64
            
            shadow_map[dr : dr+64, dc : dc+64] = s_bits
            alpha_maps[:, dr : dr+64, dc : dc+64] = a_vals

        return {
            'heights': height_map, 
            'normals': normal_map, 
            'mccv': mccv_map, 
            'shadows': shadow_map, 
            'alphas': alpha_maps
        }

def generate_synthetic_minimap(heightmap, normalmap, alpha_layers, texture_colors):
    """
    Generate a synthetic flat projection minimap from terrain data.
    
    Args:
        heightmap: [H, W] normalized heights
        normalmap: [3, H, W] normalized normals
        alpha_layers: [4, H, W] splat masks
        texture_colors: [4, 3] RGB colors for each layer
    """
    # Simple Phong lighting based on normals
    # Light direction (Sun)
    light_dir = torch.tensor([0.5, 0.7, 0.5])
    light_dir = light_dir / torch.linalg.norm(light_dir)
    
    # Dot product
    n = normalmap * 2 - 1 # [0,1] -> [-1,1]
    diffuse = torch.einsum('chw,c->hw', n, light_dir).clamp(0, 1)
    
    # Texture blending
    base_color = torch.zeros(3, heightmap.shape[0], heightmap.shape[1])
    for i in range(min(4, alpha_layers.shape[0])):
        mask = alpha_layers[i]
        color = torch.tensor(texture_colors[i]).view(3, 1, 1)
        base_color += mask * color
        
    # Combine
    final = base_color * (0.5 + 0.5 * diffuse)
    return final.clamp(0, 1)

class VectorMatcher:
    """
    Simple exact-nearest-neighbor matcher for embeddings (FAISS placeholder).
    """
    def __init__(self, dimension=16):
        self.dim = dimension
        self.vectors = []
        self.ids = []
        
    def add(self, vectors, ids):
        """Add vectors [N, D] and corresponding IDs."""
        self.vectors.append(vectors)
        self.ids.extend(ids)
        
    def build(self):
        if self.vectors:
            self.index = torch.from_numpy(np.vstack(self.vectors)).float()
        else:
            self.index = torch.empty(0, self.dim)
            
    def search(self, query, k=1):
        """Find k nearest neighbors for query [1, D]."""
        if self.index.shape[0] == 0:
            return [], []
            
        dists = torch.cdist(query, self.index) # [1, N]
        scores, indices = torch.topk(dists, k, largest=False)
        
        found_ids = [self.ids[i] for i in indices[0].tolist()]
        return scores[0].tolist(), found_ids
