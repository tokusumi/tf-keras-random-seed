# Fixing global random seed for Tensorflow.keras

![](https://github.com/tokusumi/tf-keras-random-seed/workflows/Tests/badge.svg)

## Introduction
This repository is built for testing how to fix global random seed for tensorflow.keras (2.x) with GitHub Actions.

Tests:
Python 3.6+ and Tensorflow 2.0+, especially:

* Functional API
* tf.keras.Sequential
* tf.keras subclass

## Example

```python
import tensorflow as tf
import numpy as np
import random
import os


def set_seed(seed=200):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(0)
toy_model = tf.keras.Sequential(
    tf.keras.layers.Dense(2, input_shape=(10,))
)

# working some code...

# reprodece the model, the following model has same initial variable (randomly generated) as above model: 
set_seed(0)
reproducible_toy_model = tf.keras.Sequential(
    tf.keras.layers.Dense(2, input_shape=(10,))
)
```

See tests/test_seed.py for more details.

