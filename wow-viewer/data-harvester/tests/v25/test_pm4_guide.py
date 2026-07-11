import pytest
from harvester.pm4_asset_matching.models import Pm4Bounds3
from harvester.v25.pm4_guide import V25Pm4GuideHandler

class DummySegment:
    def __init__(self, bounds):
        self.bounds = bounds

class DummyAsset:
    def __init__(self, bounds, asset_path):
        self.bounds = bounds
        self.asset_path = asset_path

def test_pm4_coordinate_snapping():
    """Verify that predicted coordinates within snap distance are snapped to PM4 centroids."""
    # Centroid: (10.0, 20.0, 5.0)
    seg = DummySegment(
        bounds=Pm4Bounds3(min=(8.0, 18.0, 4.0), max=(12.0, 22.0, 6.0))
    )
    
    # Target library matching
    asset = DummyAsset(
        bounds=Pm4Bounds3(min=(0.0, 0.0, 0.0), max=(4.0, 4.0, 2.0)), # size: 4x4x2
        asset_path="World/wmo/Dungeon/TestDungeon.wmo"
    )
    
    handler = V25Pm4GuideHandler(asset_library=[asset])
    
    # 1. Close prediction (distance = sqrt(1^2 + 1^2 + 1^2) = 1.732 <= 15.0)
    pred_close = [
        {
            "coords": [9.0, 19.0, 4.0],
            "rotations": [0.0, 0.0, 45.0],
            "class_id": 2,
            "exist_prob": 0.9
        }
    ]
    
    guided_close = handler.guide_placements(pred_close, [seg], snap_distance=15.0)
    assert len(guided_close) == 1
    assert guided_close[0]["coords"] == [10.0, 20.0, 5.0] # Centroid of segment bounds
    assert guided_close[0]["resolved_asset_name"] == "World/wmo/Dungeon/TestDungeon.wmo"
    assert guided_close[0]["match_confidence"] == 1.0 # Exact size match

def test_pm4_strict_rejection():
    """Verify that predictions outside snap distance are rejected in strict mode."""
    seg = DummySegment(
        bounds=Pm4Bounds3(min=(0.0, 0.0, 0.0), max=(2.0, 2.0, 2.0)) # Centroid: (1.0, 1.0, 1.0)
    )
    
    # Pred coordinate: (50.0, 50.0, 50.0) -> distance is ~85.0
    pred = [
        {
            "coords": [50.0, 50.0, 50.0],
            "rotations": [0.0, 0.0, 0.0],
            "class_id": 5,
            "exist_prob": 0.8
        }
    ]
    
    handler = V25Pm4GuideHandler()
    
    # Non-strict mode should retain the prediction
    guided_non_strict = handler.guide_placements(pred, [seg], snap_distance=10.0, strict_mode=False)
    assert len(guided_non_strict) == 1
    assert guided_non_strict[0]["coords"] == [50.0, 50.0, 50.0]
    
    # Strict mode should reject the prediction
    guided_strict = handler.guide_placements(pred, [seg], snap_distance=10.0, strict_mode=True)
    assert len(guided_strict) == 0
