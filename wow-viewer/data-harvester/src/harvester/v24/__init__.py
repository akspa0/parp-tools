"""Spec 094 (V24): WDL prior + lattice detailer.

Wraps the existing C# WDL surfaces (WdlSummaryReader / WdlWriter via the
WowViewer.Tool.WdlRead shim) and builds the merged WDL prior, the minimap
cleaner, and the two small models (Stage A prior predictor, Stage B detailer).
"""
