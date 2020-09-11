import tensorflow as tf
import numpy as np
import random
import os


def set_seed(seed=200):
    """set global seed to fix random-generated value for reproducible.
    available at Functional API, tf.keras.Sequential and tf.keras subclass.

    NOTE: operation seed is not fixed.
    You need to call this before the operations you want to reproduce.
    Even after `set_seed`, different random-generated values are returned:
    >>> tf.random.set_seed(0)
    >>> tf.random.uniform([1])
    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.29197514], dtype=float32)>
    >>> tf.random.uniform([1])
    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.5554141], dtype=float32)>

    However, if you continue as follows, reproducibility is ensured:
    >>> tf.random.set_seed(0)
    >>> tf.random.uniform([1])
    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.29197514], dtype=float32)>
    >>> tf.random.uniform([1])
    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.5554141], dtype=float32)>
    """
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
