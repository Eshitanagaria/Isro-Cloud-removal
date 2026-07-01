"""
gan/ package  –  Person B
Exports the public API used by Person A (eval) and Person C (pipeline).
"""

from gan.generator     import CloudRemovalGenerator, StubGenerator
from gan.discriminator import PatchGANDiscriminator
from gan.predict       import predict, predict_batch
from gan.dataset       import SyntheticPatchDataset, build_dataloaders

__all__ = [
    "CloudRemovalGenerator",
    "StubGenerator",
    "PatchGANDiscriminator",
    "predict",
    "predict_batch",
    "SyntheticPatchDataset",
    "build_dataloaders",
]
