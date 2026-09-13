"""
ElderGuard - public dataset loader (stub, M1)
ECE 310 Group 8

We shortlisted two public CSI datasets:
  - Widar 3.0 (Tsinghua University): multi-person, multi-room, multi-activity
    CSI traces with amplitude and phase across subcarriers.
  - UT-HAR: CSI-based human activity recognition dataset, smaller and simpler
    to get running first.

This file is intentionally a stub for M1: the functions define the interface
the rest of the pipeline (preprocess.py, and the M2 model code) expects, so
that swapping in real parsing code for whichever dataset we finalize does not
require touching anything downstream.

Splitting strategy (drafted here, used from M2 onward):
  We split by person AND by environment/room, not by individual clip. Every
  clip from a given (person, room) pair goes entirely into either train or
  test, never both -- this is what "leave-one-person/environment-out" means
  in the report, and it is the main thing distinguishing our evaluation setup
  from most of the papers we reviewed.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CSISample:
    amplitude: np.ndarray     # [T, n_subcarriers]
    phase: np.ndarray         # [T, n_subcarriers]
    label: str                # "fall" | "normal_activity" | "no_activity"
    person_id: str
    environment_id: str


def load_widar3(root_dir: str):
    """TODO (M2): parse Widar 3.0 .mat/.dat files into a list[CSISample].
    Left unimplemented in M1 -- see README for current download/setup status.
    """
    raise NotImplementedError("Widar 3.0 parsing scheduled for M2.")


def load_ut_har(root_dir: str):
    """TODO (M2): parse UT-HAR CSV/.mat files into a list[CSISample]."""
    raise NotImplementedError("UT-HAR parsing scheduled for M2.")


def leave_one_out_split(samples, held_out_person=None, held_out_environment=None):
    """Splits `samples` (list[CSISample]) into train/test so that the held-out
    person and/or environment appears ONLY in the test set. At least one of
    held_out_person / held_out_environment must be given.
    """
    if held_out_person is None and held_out_environment is None:
        raise ValueError("Specify held_out_person and/or held_out_environment.")

    def is_held_out(s):
        person_match = held_out_person is not None and s.person_id == held_out_person
        env_match = held_out_environment is not None and s.environment_id == held_out_environment
        return person_match or env_match

    test = [s for s in samples if is_held_out(s)]
    train = [s for s in samples if not is_held_out(s)]
    return train, test
