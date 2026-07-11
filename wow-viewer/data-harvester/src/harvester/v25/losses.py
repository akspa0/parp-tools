import torch
import torch.nn as nn
import torch.nn.functional as F

class V25UnifiedLoss(nn.Module):
    """Unified loss function combining segmentation, mesh heights, object placements, and texturing."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred_outputs, target_outputs):
        """Calculate weighted multi-task losses.
        
        Args:
            pred_outputs: Dict of predicted outputs containing:
                - "mask_logits": (B, 1, 256, 256)
                - "placements": dict of class_logits, coords, rotations, exist_logits
                - "h_257": (B, 257, 257)
                - "mtex_logits": (B, vocab_size)
                - "mcly_logits": (B, 4, 16, 16, 4)
                - "alpha_256": (B, 4, 256, 256)
            target_outputs: Dict of ground-truth targets containing:
                - "mask": (B, 1, 256, 256)
                - "placements": dict of class_ids, coords, rotations, exist
                - "h_257": (B, 257, 257)
                - "mtex_labels": (B, vocab_size)
                - "mcly_labels": (B, 16, 16, 4)
                - "alpha_256": (B, 4, 256, 256)
                
        Returns:
            losses: Dict of specific loss components and the final "loss" tensor
        """
        # 1. Footprint mask segmentation loss
        loss_mask = self.bce(pred_outputs["mask_logits"], target_outputs["mask"])
        
        # 2. Edge-aligned terrain heightmap loss
        loss_height = self.l1(pred_outputs["h_257"], target_outputs["h_257"])
        
        # 3. Object placement coordinates, rotations, classifications, and existences
        pred_p = pred_outputs["placements"]
        tgt_p = target_outputs["placements"]
        
        # Mask coordinates and rotations by ground-truth existence
        exist_mask = tgt_p["exist"].view(-1, tgt_p["exist"].shape[1], 1) # (B, max_objects, 1)
        loss_coords = self.mse(pred_p["coords"] * exist_mask, tgt_p["coords"] * exist_mask)
        loss_rotations = self.mse(pred_p["rotations"] * exist_mask, tgt_p["rotations"] * exist_mask)
        
        # Existence prediction loss
        loss_exist = self.bce(pred_p["exist_logits"], tgt_p["exist"].unsqueeze(-1).float())
        
        # Classification prediction loss
        pred_class = pred_p["class_logits"].view(-1, pred_p["class_logits"].shape[-1])
        tgt_class = tgt_p["class_ids"].view(-1).long()
        loss_class = self.ce(pred_class, tgt_class)
        
        # 4. Texturing losses: MTEX paths, MCLY indices, and MCAL alpha canvas
        loss_mtex = self.bce(pred_outputs["mtex_logits"], target_outputs["mtex_labels"].float())
        loss_mcly = self.ce(pred_outputs["mcly_logits"], target_outputs["mcly_labels"].long())
        loss_alpha = self.l1(pred_outputs["alpha_256"], target_outputs["alpha_256"])
        
        # Unified weighted sum
        total_loss = (
            1.0 * loss_mask +
            5.0 * loss_height +
            2.0 * loss_coords +
            1.0 * loss_rotations +
            1.0 * loss_exist +
            1.0 * loss_class +
            1.0 * loss_mtex +
            1.0 * loss_mcly +
            3.0 * loss_alpha
        )
        
        return {
            "loss": total_loss,
            "mask": loss_mask,
            "height": loss_height,
            "coords": loss_coords,
            "rotations": loss_rotations,
            "exist": loss_exist,
            "class": loss_class,
            "mtex": loss_mtex,
            "mcly": loss_mcly,
            "alpha": loss_alpha
        }
