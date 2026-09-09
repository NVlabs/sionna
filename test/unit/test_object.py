#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch
import numpy as np
from sionna.phy import config, dtypes, Object


def test_default_properties():
    obj = Object()
    assert obj.precision == config.precision
    assert obj.device == config.device
    assert obj.torch_rng == config.torch_rng(obj.device)
    assert obj.np_rng == config.np_rng
    assert obj.py_rng == config.py_rng
    assert obj.dtype == dtypes[config.precision]["torch"]["dtype"]
    assert obj.cdtype == dtypes[config.precision]["torch"]["cdtype"]
    assert obj.np_dtype == dtypes[config.precision]["np"]["dtype"]
    assert obj.np_cdtype == dtypes[config.precision]["np"]["cdtype"]


def test_precision_setter(precision):
    obj = Object(precision=precision)
    assert obj.precision == precision
    assert obj.dtype == dtypes[precision]["torch"]["dtype"]
    assert obj.cdtype == dtypes[precision]["torch"]["cdtype"]
    assert obj.np_dtype == dtypes[precision]["np"]["dtype"]
    assert obj.np_cdtype == dtypes[precision]["np"]["cdtype"]


def test_device_setter(device):
    obj = Object(device=device)
    assert obj.device == device
    assert obj.torch_rng == config.torch_rng(device)


def test_convert(precision, device):
    obj = Object(precision=precision, device=device)

    # None stays None
    assert obj._convert(None) is None

    # Strings stay as strings
    assert obj._convert("test") == "test"

    # Ints stay as ints (used for shapes/indices)
    assert obj._convert(1) == 1
    assert type(obj._convert(1)) is int

    # Floats are converted to tensors
    float_result = obj._convert(1.0)
    assert isinstance(float_result, torch.Tensor)
    assert float_result.dtype == obj.dtype
    assert float_result.device == torch.device(obj.device)

    # Complex values are converted to tensors
    complex_result = obj._convert(1.0j)
    assert isinstance(complex_result, torch.Tensor)
    assert complex_result.dtype == obj.cdtype
    assert complex_result.device == torch.device(obj.device)

    # Lists preserve structure, converting elements
    list_result = obj._convert(["1", 2, 3])
    assert list_result[0] == "1"  # strings stay strings
    assert list_result[1] == 2  # ints stay ints
    assert list_result[2] == 3

    # Tuples preserve structure
    tuple_result = obj._convert((1, 2, 3))
    assert tuple_result == (1, 2, 3)
    assert isinstance(tuple_result, tuple)

    # Numpy arrays are converted to tensors
    assert torch.equal(
        obj._convert(np.array([1, 2, 3])),
        torch.tensor([1, 2, 3], dtype=obj.dtype, device=obj.device),
    )

    # Integer tensors keep their dtype
    assert torch.equal(
        obj._convert(torch.tensor([1, 2, 3], dtype=torch.int32)),
        torch.tensor([1, 2, 3], dtype=torch.int32, device=obj.device),
    )
    assert torch.equal(
        obj._convert(np.array([1, 2, 3], dtype=np.int32)),
        torch.tensor([1, 2, 3], dtype=torch.int32, device=obj.device),
    )

    # Test list of numpy arrays
    data = [np.array([1.0, 2, 3]), np.array([4.0, 5, 6])]
    output = obj._convert(data)
    for i in range(len(data)):
        assert torch.equal(
            output[i], torch.tensor(data[i], dtype=obj.dtype, device=obj.device)
        )

    # Test dict of numpy arrays
    data = {"a": np.array([1.0, 2, 3]), "b": np.array([4.0, 5, 6])}
    output = obj._convert(data)
    for k, v in data.items():
        assert torch.equal(
            output[k], torch.tensor(v, dtype=obj.dtype, device=obj.device)
        )

    # Test dict of mixed types - floats/complex become tensors
    data = {"a": np.array([1.0, 2, 3]), "b": 4.0, "c": 5.0j}
    output = obj._convert(data)
    assert torch.equal(
        output["a"], torch.tensor(data["a"], dtype=obj.dtype, device=obj.device)
    )
    assert isinstance(output["b"], torch.Tensor)
    assert output["b"].dtype == obj.dtype
    assert isinstance(output["c"], torch.Tensor)
    assert output["c"].dtype == obj.cdtype


def test_convert_invalid_precision():
    """Test that invalid precision raises ValueError."""
    with pytest.raises(ValueError, match="Invalid precision"):
        Object(precision="invalid")


def test_convert_invalid_device():
    """Test that invalid device raises ValueError."""
    with pytest.raises(ValueError, match="Invalid device"):
        Object(device="invalid_device")


def test_get_shape(precision, device):
    """Test the _get_shape method."""
    obj = Object(precision=precision, device=device)

    # Scalar has empty shape
    assert obj._get_shape(1) == ()
    assert obj._get_shape(1.0) == ()

    # Tensors return their shape
    t = torch.randn(3, 4, 5)
    assert obj._get_shape(t) == (3, 4, 5)

    # Numpy arrays return their shape
    arr = np.zeros((2, 3))
    assert obj._get_shape(arr) == (2, 3)

    # Lists of tensors return list of shapes
    tensors = [torch.randn(2, 3), torch.randn(4, 5, 6)]
    shapes = obj._get_shape(tensors)
    assert shapes == [(2, 3), (4, 5, 6)]

    # Dicts of tensors return dict of shapes
    tensor_dict = {"a": torch.randn(1, 2), "b": torch.randn(3, 4, 5)}
    shape_dict = obj._get_shape(tensor_dict)
    assert shape_dict == {"a": (1, 2), "b": (3, 4, 5)}

    # Nested structures work recursively
    nested = {"x": [torch.randn(2, 2), torch.randn(3, 3)], "y": torch.randn(4, 4)}
    nested_shapes = obj._get_shape(nested)
    assert nested_shapes == {"x": [(2, 2), (3, 3)], "y": (4, 4)}


def test_composed_objects_accept_matching_devices(device):
    """Objects on the same device can be composed normally."""
    parent = Object(device=device)
    child = Object(device=device)

    parent.child = child

    assert parent.child is child
    assert parent._modules["child"] is child


def test_composed_objects_reject_mismatched_devices(device):
    """A device mismatch is reported when the child is assigned."""
    child_device = next(
        (candidate for candidate in config.available_devices if candidate != device),
        None,
    )
    if child_device is None:
        pytest.skip("Test requires two available devices")
    parent = Object(device=device)
    child = Object(device=child_device)

    with pytest.raises(
        ValueError,
        match=r"Object is on device .* but its child \(Object\) is on device",
    ):
        parent.child = child


def test_composed_objects_check_children_in_containers(device):
    """Container-held child objects follow the same device invariant."""
    child_device = next(
        (candidate for candidate in config.available_devices if candidate != device),
        None,
    )
    if child_device is None:
        pytest.skip("Test requires two available devices")
    parent = Object(device=device)
    child = Object(device=child_device)

    with pytest.raises(
        ValueError,
        match=r"Object is on device .* but its callbacks\[0\] "
        r"\(Object\) is on device",
    ):
        parent.callbacks = [child]


def test_device_check_skips_non_container_values(device, monkeypatch):
    """Leaf assignments must not require Python object identity."""
    parent = Object(device=device)

    def unexpected_id(_):
        pytest.fail("id() must only be called for container cycle detection")

    monkeypatch.setattr("sionna.phy.object.id", unexpected_id, raising=False)
    parent.flag = True
    parent.count = 1
    parent.tensor = torch.ones(1, device=device)
