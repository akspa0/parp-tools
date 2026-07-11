import math
import torch
from harvester.pm4_asset_matching.models import (
    Pm4SegmentSignalRecord,
    Pm4AssetReferenceSignalRecord,
    Pm4Bounds3
)
from harvester.pm4_asset_matching.scorer import score_ratio, score_distance, compute_bounds_overlap_ratio

class V25Pm4GuideHandler:
    """Combines machine learning object predictions with hand-decoded symbolic PM4 collision coordinates."""
    def __init__(self, asset_library=None):
        # asset_library is a list of Pm4AssetReferenceSignalRecord
        self.asset_library = asset_library if asset_library is not None else []
        
    def guide_placements(self, predicted_placements, pm4_segments, snap_distance=15.0, strict_mode=False):
        """Align predicted placements with PM4 segment centroids and match WMO/M2 names.
        
        Args:
            predicted_placements: List of dicts, each with keys:
                - 'coords': list of float (x, y, z) in world coordinates
                - 'rotations': list of float (roll, pitch, yaw) in degrees
                - 'class_id': int (predicted class)
                - 'exist_prob': float (existence probability)
            pm4_segments: List of Pm4SegmentSignalRecord representing parsed collision blocks
            snap_distance: Max distance in meters to snap predicted object to PM4 centroid
            strict_mode: If True, rejects predicted placements not snapped to any PM4 segment
            
        Returns:
            guided: List of guided placement dicts with snapped coordinates and matched model names
        """
        guided_list = []
        used_segments = set()
        
        for pred in predicted_placements:
            if pred.get("exist_prob", 0.0) < 0.5:
                continue
                
            pred_coord = pred["coords"]
            # Find the closest PM4 segment
            best_dist = float("inf")
            best_seg = None
            best_seg_idx = -1
            
            for idx, seg in enumerate(pm4_segments):
                if seg.bounds is None:
                    continue
                # Calculate segment center centroid
                cx = 0.5 * (seg.bounds.min[0] + seg.bounds.max[0])
                cy = 0.5 * (seg.bounds.min[1] + seg.bounds.max[1])
                cz = 0.5 * (seg.bounds.min[2] + seg.bounds.max[2])
                
                dist = math.sqrt(
                    (pred_coord[0] - cx) ** 2 +
                    (pred_coord[1] - cy) ** 2 +
                    (pred_coord[2] - cz) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_seg = seg
                    best_seg_idx = idx
            
            snapped = False
            guided_placement = pred.copy()
            
            if best_dist <= snap_distance and best_seg is not None:
                cx = 0.5 * (best_seg.bounds.min[0] + best_seg.bounds.max[0])
                cy = 0.5 * (best_seg.bounds.min[1] + best_seg.bounds.max[1])
                cz = 0.5 * (best_seg.bounds.min[2] + best_seg.bounds.max[2])
                
                # Snap coordinates to segment centroid
                guided_placement["coords"] = [cx, cy, cz]
                guided_placement["pm4_segment_idx"] = best_seg_idx
                used_segments.add(best_seg_idx)
                snapped = True
                
                # Attempt to resolve the asset name using the PM4 library scorer
                best_match_name = None
                best_match_score = -1.0
                
                for asset in self.asset_library:
                    if asset.bounds is None:
                        continue
                    # Compute score based on bounding box ratio
                    seg_span = tuple(best_seg.bounds.max[i] - best_seg.bounds.min[i] for i in range(3))
                    asset_span = tuple(asset.bounds.max[i] - asset.bounds.min[i] for i in range(3))
                    
                    span_score0 = score_ratio(seg_span[0], asset_span[0])
                    span_score1 = score_ratio(seg_span[1], asset_span[1])
                    span_score2 = score_ratio(seg_span[2], asset_span[2])
                    shape_score = (span_score0 + span_score1 + span_score2) / 3.0
                    
                    if shape_score > best_match_score:
                        best_match_score = shape_score
                        best_match_name = asset.asset_path
                        
                if best_match_score > 0.6:
                    guided_placement["resolved_asset_name"] = best_match_name
                    guided_placement["match_confidence"] = best_match_score
                    
            if snapped or not strict_mode:
                guided_list.append(guided_placement)
                
        return guided_list
