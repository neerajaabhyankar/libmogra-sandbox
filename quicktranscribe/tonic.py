import json
import numpy as np
from typing import Dict


def read_metadata(metadata_file) -> Dict:
    """Annotated data"""
    with open(metadata_file) as json_data:
        metadata = json.load(json_data)
    return metadata


def read_tonic(tonic_file, verbose=False) -> float:
    """Annotated data"""
    tonic = float(np.loadtxt(tonic_file))
    if verbose:
        print(f"base tonic = {tonic}")
    return tonic


def write_tonic(tonic_file, tonic):
    """Annotate data"""
    np.savetxt(tonic_file, [tonic])


def infer_tonic(audio_file) -> Dict[float, float]:
    """Return a candidate list + probabilities"""
    raise NotImplemented
