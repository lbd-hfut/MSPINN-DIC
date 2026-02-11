import os
import sys
import ast
import time
import copy
import cv2
import jax
import math
import tqdm
import shutil
import glob
import pickle
import jax.numpy as jnp
from jax import lax
import numpy as np
from PIL import Image
from functools import partial
from math import factorial
from typing import List, Tuple, Dict, NamedTuple
from sklearn.cluster import KMeans
from types import SimpleNamespace
from scipy.ndimage import label
import scipy.io as sio
from scipy.io import loadmat
from scipy.io import savemat
from matplotlib import pyplot as plt

def seed_everything(seed_value: int):
    """
    Fix random seeds for NumPy and JAX.
    Returns
    -------
    key : jax.random.PRNGKey
        Base JAX random key (should be split explicitly later).
    """
    # ---------- NumPy ----------
    np.random.seed(seed_value)
    # ---------- JAX ----------
    key = jax.random.PRNGKey(seed_value)
    return key

key = seed_everything(42)
