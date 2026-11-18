"""SIREN training on PhotonSim tables: the trainer, its dataset, its monitor and its analyzer.

This package deliberately **imports nothing**. Import the module you need:

    from lucid.siren.training.trainer import Recipe, SIRENTrainer
    from lucid.siren.training.dataset import PhotonSimDataset
    from lucid.siren.training.monitor import TrainingMonitor, LiveTrainingCallback
    from lucid.siren.training.analyzer import TrainingAnalyzer

Its modules pull in optax, matplotlib and, for the monitor, IPython (declared by no extra), which
the forward simulator does not need. Nothing under ``lucid/`` may import this package itself --
only its modules -- and ``tests/test_imports.py`` keeps it so.

Modules
-------
trainer
    ``Recipe`` (the working point; ``nominal.json`` is the committed one) and ``SIRENTrainer``:
    Adam (or AdamW, SGD) with a plateau schedule, warm start, checkpoints; writes the emitter.
dataset
    ``PhotonSimDataset``: the lookup table and its batching.
monitor
    Live progress: ``TrainingMonitor``, ``LiveTrainingCallback`` (matplotlib, IPython).
analyzer
    ``TrainingAnalyzer``: post-training evaluation plots.
prediction_plot
    ``PredictionComparisonCallback``: in-training prediction-vs-truth PDFs.
train, validate
    The CLIs: ``lucid-train-siren`` (``python -m lucid.siren.training.train``) and the
    validation suite (``python -m lucid.siren.training.validate``).

Loading a trained emitter for the forward simulator is ``lucid.siren.emitter.load_emitter``.
"""
