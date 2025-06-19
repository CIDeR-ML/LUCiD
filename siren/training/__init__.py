"""
Training module for JAXSiren on PhotonSim data.

This module provides classes and functions for training SIREN networks
on PhotonSim lookup tables and datasets.
"""

from .trainer import SIRENTrainer, TrainingConfig
from .monitor import TrainingMonitor, LiveTrainingCallback
from .analyzer import TrainingAnalyzer
from .dataset import PhotonSimDataset

__all__ = [
    'SIRENTrainer',
    'TrainingConfig',
    'TrainingMonitor', 
    'LiveTrainingCallback',
    'TrainingAnalyzer',
    'PhotonSimDataset',
]