"""Expose separately pinned additions without changing the legacy hash inventory.

The original package and all its provenance checks remain byte-identical. Python
imports additive modules through an extra package search location; their own
receipts and the complete additive release manifest pin those modules.
"""
from pathlib import Path
import adversarial_study

_addition_path = Path(__file__).resolve().parents[1] / "proposal_additions"
if _addition_path.is_dir():
    adversarial_study.__path__.append(str(_addition_path))
