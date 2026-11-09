"""PyTorch-compatible training storage without ever unpickling incoming metadata."""

from io import BytesIO
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
import torch

from iris.application.scientific_torch_state import (
    decode_scientific_torch_state_v1,
    encode_scientific_torch_state_v1,
)
from iris.artifacts.scientific_checkpoint_v1 import ScientificTensorV1


def _state() -> dict[str, torch.Tensor]:
    return {
        "adamw/0/exp_avg": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "model/configuration_identity": torch.arange(32, dtype=torch.uint8),
        "runtime/cpu": torch.Generator().manual_seed(43).get_state(),
    }


def _descriptors(state: dict[str, torch.Tensor]) -> tuple[ScientificTensorV1, ...]:
    return tuple(
        ScientificTensorV1(
            key=name,
            shape=tuple(value.shape),
            dtype="float32" if value.dtype == torch.float32 else "uint8",
            byte_length=value.numel() * value.element_size(),
        )
        for name, value in sorted(state.items())
    )


def test_canonical_pt_is_torch_compatible_but_reader_never_calls_torch_load() -> None:
    state = _state()
    encoded = encode_scientific_torch_state_v1('{"step":7}', state)
    assert encode_scientific_torch_state_v1('{"step":7}', state) == encoded
    external = torch.load(BytesIO(encoded), weights_only=True, map_location="cpu")
    assert external["control_json"] == '{"step":7}'
    assert all(torch.equal(external["tensors"][key], value) for key, value in state.items())
    with patch("torch.load", side_effect=AssertionError("must not unpickle input")):
        actual = decode_scientific_torch_state_v1(
            encoded, control_json='{"step":7}', descriptors=_descriptors(state)
        )
    assert all(torch.equal(actual[key], value) for key, value in state.items())
    state["adamw/0/exp_avg"].add_(1)
    assert not torch.equal(actual["adamw/0/exp_avg"], state["adamw/0/exp_avg"])


@pytest.mark.parametrize(
    "kind",
    ["metadata", "extra", "compressed", "truncated_storage", "trailing", "control", "descriptor"],
)
def test_modified_archives_reject_without_unpickling(kind: str) -> None:
    state = _state()
    encoded = encode_scientific_torch_state_v1("{}", state)
    descriptors = _descriptors(state)
    control = "{}"
    if kind == "trailing":
        encoded += b"unexpected suffix"
    elif kind == "control":
        control = '{"different":true}'
    elif kind == "descriptor":
        descriptors = (descriptors[0].model_copy(update={"shape": (6,)}), *descriptors[1:])
    else:
        output = BytesIO()
        with ZipFile(BytesIO(encoded)) as source, ZipFile(output, "w") as target:
            for info in source.infolist():
                data = source.read(info.filename)
                if kind == "metadata" and info.filename == "archive/data.pkl":
                    # An unsupported GLOBAL is data, never evaluated by our reader.
                    data = b"cbuiltins\neval\n(V1 + 2\ntR."
                if kind == "truncated_storage" and info.filename == "archive/data/0":
                    data = data[:-1]
                if kind == "compressed":
                    info.compress_type = ZIP_DEFLATED
                target.writestr(info, data)
            if kind == "extra":
                target.writestr("../unexpected", b"never extracted")
        encoded = output.getvalue()
    with (
        patch("torch.load", side_effect=AssertionError("must not unpickle input")),
        pytest.raises(ValueError),
    ):
        decode_scientific_torch_state_v1(encoded, control_json=control, descriptors=descriptors)


def test_nonfinite_training_tensor_is_rejected() -> None:
    state = _state()
    state["adamw/0/exp_avg"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        encode_scientific_torch_state_v1("{}", state)
