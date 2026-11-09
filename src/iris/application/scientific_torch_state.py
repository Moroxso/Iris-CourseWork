"""Canonical PyTorch training-state container, verified without unpickling input.

Storage lengths/shapes come from the separately validated scientific manifest.
Read only bounded stored ZIP members, reconstruct owned tensors, then reproduce
the complete torch.save archive and require byte equality. Never evaluate the
incoming data.pkl, call torch.load, register safe globals or extract ZIP paths.
"""

from io import BytesIO
from zipfile import ZIP_STORED, ZipFile

import torch

from iris.artifacts.scientific_checkpoint_v1 import ScientificTensorV1

_LIMIT = 512 * 1024**2
_FIXED = {
    "archive/data.pkl",
    "archive/.format_version",
    "archive/.storage_alignment",
    "archive/byteorder",
    "archive/version",
    "archive/.data/serialization_id",
}


def encode_scientific_torch_state_v1(control_json: str, tensors: dict[str, torch.Tensor]) -> bytes:
    """Create a normal torch.save file; the delivery caller excludes trainable weights."""
    if type(control_json) is not str or len(control_json.encode("utf-8")) > 16 * 1024**2:
        raise ValueError("Bounded explicit training control JSON required")
    if type(tensors) is not dict or not tensors or len(tensors) > 16384:
        raise ValueError("Bounded training tensor inventory required")
    owned = {}
    size = 0
    for name, value in sorted(tensors.items()):
        if (
            type(name) is not str
            or not name
            or len(name) > 512
            or type(value) is not torch.Tensor
            or value.layout != torch.strided
            or value.dtype not in (torch.float32, torch.uint8)
            or value.is_inference()
            or value.device.type != "cpu"
            or value.numel() == 0
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Unsupported training tensor")
        size += value.numel() * value.element_size()
        if size > _LIMIT:
            raise ValueError("Training storage limit exceeded")
        owned[name] = value.detach().contiguous().clone()
    stream = BytesIO()
    torch.save(
        {
            "format": "iris-scientific-training-state-v1",
            "control_json": control_json,
            "tensors": owned,
        },
        stream,
        pickle_protocol=2,
    )
    result = stream.getvalue()
    if len(result) > _LIMIT:
        raise ValueError("Training file limit exceeded")
    return result


def decode_scientific_torch_state_v1(
    content: bytes, *, control_json: str, descriptors: tuple[ScientificTensorV1, ...]
) -> dict[str, torch.Tensor]:
    """Reject every archive other than this exact bounded encoding, without pickle."""
    if (
        type(content) is not bytes
        or not 1 <= len(content) <= _LIMIT
        or type(descriptors) is not tuple
        or not 1 <= len(descriptors) <= 16384
    ):
        raise ValueError("Bounded training archive and descriptors required")
    entries = tuple(
        ScientificTensorV1.model_validate(t.model_dump(mode="python")) for t in descriptors
    )
    names = tuple(t.key for t in entries)
    if names != tuple(sorted(set(names))) or sum(t.byte_length for t in entries) > _LIMIT:
        raise ValueError("Training storage descriptor inventory differs")
    tensors = {}
    expected = _FIXED | {f"archive/data/{index}" for index in range(len(entries))}
    with ZipFile(BytesIO(content)) as archive:
        infos = archive.infolist()
        if len(infos) != len(expected) or {info.filename for info in infos} != expected:
            raise ValueError("Training ZIP inventory differs")
        for info in infos:
            if info.compress_type != ZIP_STORED or info.flag_bits & 1 or info.file_size > _LIMIT:
                raise ValueError("Compressed/encrypted/oversized training ZIP member rejected")
        for index, entry in enumerate(entries):
            member = f"archive/data/{index}"
            if archive.getinfo(member).file_size != entry.byte_length:
                raise ValueError("Training ZIP storage length differs")
            raw = bytearray(archive.read(member))
            dtype = torch.float32 if entry.dtype == "float32" else torch.uint8
            # This version deliberately supports only nonempty owned storages.
            if not raw:
                raise ValueError("Empty training storages are not supported")
            value = torch.frombuffer(raw, dtype=dtype).reshape(entry.shape).clone()
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite training storage rejected")
            tensors[entry.key] = value
    if encode_scientific_torch_state_v1(control_json, tensors) != content:
        raise ValueError("Training PyTorch archive is noncanonical or has altered pickle metadata")
    return tensors


__all__ = ["decode_scientific_torch_state_v1", "encode_scientific_torch_state_v1"]
