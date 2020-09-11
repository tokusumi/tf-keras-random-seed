import typing
import tensorflow as tf
import numpy as np
import pytest
import pdb
from tf_keras_random_seed.seed import set_seed


def build_helper(layers, inp=(16, 16, 3)):
    inp = tf.keras.Input(inp)
    x = inp
    for layer in layers:
        x = layer(x)
    model = tf.keras.Model(inp, x)
    return model


def is_reproducible(build: typing.Callable[..., tf.keras.Model]):
    model = build()
    sub = build()
    for var, sub_var in zip(model.variables, sub.variables):
        if not np.all(var.numpy() == sub_var.numpy()):
            return False
    return True


def test_fail_initializer_functional_api():
    """this test is failed. seed is not fixed."""

    def build():
        layers = [
            tf.keras.layers.Conv2D(4, 3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2),
        ]
        model = build_helper(layers, (16, 16, 3))
        return model

    assert not is_reproducible(build)


def test_functional_api():
    """call `set_seed` just before generating random variables"""

    def build():
        set_seed(200)
        layers = [
            tf.keras.layers.Conv2D(4, 3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2),
        ]
        model = build_helper(layers, (16, 16, 3))
        return model

    assert is_reproducible(build)


def test_fail_due_to_call_in_invalid_place_functional_api():
    """this test is failed. `set_seed` must be called at just before generating random variables"""
    # not reproducible. if call `build` twice, not same variables are generated in each built models.
    set_seed(200)

    def build():
        layers = [
            tf.keras.layers.Conv2D(4, 3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2),
        ]
        model = build_helper(layers, (16, 16, 3))
        return model

    assert not is_reproducible(build)


def test_fail_initializer_sequential():
    """this test is failed. seed is not fixed."""

    def build():
        layers = [
            tf.keras.layers.Conv2D(4, 3, input_shape=(16, 16, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2),
        ]
        model = tf.keras.Sequential(layers)
        return model

    assert not is_reproducible(build)


def test_sequential():
    """call `set_seed` just before generating random variables"""

    def build():
        set_seed(200)
        layers = [
            tf.keras.layers.Conv2D(4, 3, input_shape=(16, 16, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2),
        ]
        model = tf.keras.Sequential(layers)
        return model

    assert is_reproducible(build)


def test_fail_due_to_call_in_invalid_place_sequential():
    """this test is failed. `set_seed` must be called at just before generating random variables"""
    # not reproducible. if call `build` twice, not same variables are generated in each built models.
    set_seed(200)

    def build():
        layers = [
            tf.keras.layers.Conv2D(4, 3, input_shape=(16, 16, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2),
        ]
        model = tf.keras.Sequential(layers)
        return model

    assert not is_reproducible(build)


def test_fail_initializer_subclass():
    """this test is failed. seed is not fixed."""

    def build():
        class SubClass(tf.keras.Model):
            def __init__(self, *args, **kwargs):
                super(SubClass, self).__init__(*args, **kwargs)
                self.a = tf.keras.layers.Conv2D(4, 3)
                self.b = tf.keras.layers.BatchNormalization()
                self.c = tf.keras.layers.Activation("relu")
                self.d = tf.keras.layers.Flatten()
                self.e = tf.keras.layers.Dense(2)

            def call(self, x):
                x = self.a(x)
                x = self.b(x)
                x = self.c(x)
                x = self.d(x)
                return self.e(x)

        layers = [SubClass()]
        model = build_helper(layers, (16, 16, 3))
        return model

    assert not is_reproducible(build)


def test_subclass():
    """call `set_seed` just before generating random variables"""

    def build():
        set_seed(200)

        class SubClass(tf.keras.Model):
            def __init__(self, *args, **kwargs):
                super(SubClass, self).__init__(*args, **kwargs)
                self.a = tf.keras.layers.Conv2D(4, 3)
                self.b = tf.keras.layers.BatchNormalization()
                self.c = tf.keras.layers.Activation("relu")
                self.d = tf.keras.layers.Flatten()
                self.e = tf.keras.layers.Dense(2)

            def call(self, x):
                x = self.a(x)
                x = self.b(x)
                x = self.c(x)
                x = self.d(x)
                return self.e(x)

        layers = [SubClass()]
        model = build_helper(layers, (16, 16, 3))
        return model

    assert is_reproducible(build)


def test_fail_due_to_call_in_invalid_place_subclass():
    """this test is failed. `set_seed` must be called at just before generating random variables"""
    # not reproducible. if call `build` twice, not same variables are generated in each built models.
    set_seed(200)

    def build():
        class SubClass(tf.keras.Model):
            def __init__(self, *args, **kwargs):
                super(SubClass, self).__init__(*args, **kwargs)
                self.a = tf.keras.layers.Conv2D(4, 3)
                self.b = tf.keras.layers.BatchNormalization()
                self.c = tf.keras.layers.Activation("relu")
                self.d = tf.keras.layers.Flatten()
                self.e = tf.keras.layers.Dense(2)

            def call(self, x):
                x = self.a(x)
                x = self.b(x)
                x = self.c(x)
                x = self.d(x)
                return self.e(x)

        layers = [SubClass()]
        model = build_helper(layers, (16, 16, 3))
        return model

    assert not is_reproducible(build)