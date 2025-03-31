from dataclasses import dataclass, field
from os.path import join as pjoin
from typing import List

from . import decision_mappings, info_mappings
from .base import Dataset
from .dataloaders import PytorchLoader
from .registry import register_dataset
from .. import constants as c
from ..evaluation import metrics as m

__all__ = ["cue_conflict_4", "cue_conflict_9", "cue_conflict_14", "cue_conflict_19", "cue_conflict_24", "cue_conflict_29", "filled_silhouette_cue_conflict"]


@dataclass
class TextureShapeParams:
    path: str
    image_size: int = 224
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1)])
    decision_mapping: object = decision_mappings.ImageNetProbabilitiesTo16ClassesMapping()
    info_mapping: object = info_mappings.ImageNetInfoMapping()
    experiments: List = field(default_factory=list)
    contains_sessions: bool = False


def _get_dataset(name, *args, **kwargs):
    params = TextureShapeParams(path=pjoin(c.DATASET_DIR, name))
    return Dataset(name=name,
                   params=params,
                   loader=PytorchLoader,
                   *args,
                   **kwargs)


@register_dataset(name="cue-conflict-4")
def cue_conflict_4(*args, **kwargs):
    return _get_dataset(name="cue-conflict-4", *args, **kwargs)


@register_dataset(name="cue-conflict-9")
def cue_conflict_9(*args, **kwargs):
    return _get_dataset(name="cue-conflict-9", *args, **kwargs)


@register_dataset(name="cue-conflict-14")
def cue_conflict_14(*args, **kwargs):
    return _get_dataset(name="cue-conflict-14", *args, **kwargs)


@register_dataset(name="cue-conflict-19")
def cue_conflict_19(*args, **kwargs):
    return _get_dataset(name="cue-conflict-19", *args, **kwargs)


@register_dataset(name="cue-conflict-24")
def cue_conflict_24(*args, **kwargs):
    return _get_dataset(name="cue-conflict-24", *args, **kwargs)


@register_dataset(name="cue-conflict-29")
def cue_conflict_29(*args, **kwargs):
    return _get_dataset(name="cue-conflict-29", *args, **kwargs)


@register_dataset(name="filled-silhouette-cue-conflict")
def filled_silhouette_cue_conflict(*args, **kwargs):
    return _get_dataset(name="filled-silhouette-cue-conflict", *args, **kwargs)