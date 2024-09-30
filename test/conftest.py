#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest

def pytest_addoption(parser):
    parser.addoption("--gpu", action="store", default=0, help="GPU to be used. Defaults to 0.")
    parser.addoption("--cpu", action="store_true", default=False, help="Run tests on CPU. Overrides --gpu setting.")
    parser.addoption("--seed", action="store", default=42, help="Set Sionna random seed. Defaults to 42.")

def pytest_configure(config):

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if not config.getoption("cpu"):
        gpu = config.getoption("gpu")
        print("\n===============================================\n")
        print(f"           Running tests on GPU {gpu}")
        print("\n===============================================\n")
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpu}"
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("\n========================================\n")
        print("           Running tests on CPU")
        print("\n========================================\n")
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    try:
        import sionna
    except:
        import sys
        sys.path.append("..")
        import sionna

@pytest.fixture(scope="class", autouse=True)
def set_class_random_seed(request):
    """Set random seed for all test classes"""
    import sionna
    sionna.config.seed = request.config.getoption("seed")

@pytest.fixture(scope="function", autouse=True)
def set_function_random_seed(request):
    """Set random seed for every individual test"""
    import sionna
    sionna.config.seed = request.config.getoption("seed")

@pytest.fixture(scope="function", autouse=True)
def clean_memory(request):
    """Clean-up memory after each test"""
    yield
    import tensorflow as tf
    import gc
    tf.keras.backend.clear_session()
    gc.collect()

@pytest.fixture
def only_gpu():
    import tensorflow as tf
    if not len(tf.config.list_physical_devices('GPU'))>0:
        pytest.skip("Skip test (only on GPU).")