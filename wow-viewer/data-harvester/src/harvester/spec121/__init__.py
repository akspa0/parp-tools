"""Spec 121: V7-style WDL-prior height reconstruction (small model lane).

Stage A predicts the 545-point WDL lattice from minimap RGB with a SegFormer-B0 backbone
(fallback: Spec 117's from-scratch LatticeNet). Stage B is the existing residual detailer fed
the predicted prior as its coarse input. Precise object masks are a loss-side signal only.
"""
