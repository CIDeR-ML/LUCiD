"""
Box propagation module - imports from existing propagate_box.py with optimizations
"""

# Import all functions from the existing box propagation module
from ..propagate_box import *

# TODO: Future optimization - replace box intersection with vectorized version
# from .geometry import ray_box_intersection_vectorized