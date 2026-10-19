"""Construct the GPL-3.0 LanGuide architecture from vendored source."""


def create_model(vision_model, text_model):
    # Only the import path changed from the upstream source. Parameter names
    # remain checkpoint compatible.
    from src.models.legacy_model import LanGuideMedSeg
    return LanGuideMedSeg(text_model, vision_model, project_dim=768)
