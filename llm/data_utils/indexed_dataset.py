"""Memory-mapped token dataset ``<prefix>.bin`` + ``<prefix>.idx`` (the format of MiniLLM's processed data).

Trimmed from fairseq/data/indexed_dataset.py (MIT License, Copyright (c) Facebook, Inc. and its affiliates).
"""
import struct

import numpy as np
from torch.utils.data import Dataset

_MAGIC = b"MMIDIDX\x00\x00"
_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64,
           8: np.uint16, 9: np.uint32}
_CODES = {dtype: code for code, dtype in _DTYPES.items()}


class MMapIndexedDataset(Dataset):
    def __init__(self, prefix):
        self._open(prefix)

    def _open(self, prefix):
        self.prefix = prefix
        with open(prefix + ".idx", "rb") as f:
            assert f.read(9) == _MAGIC, f"{prefix}.idx is not an mmap indexed dataset"
            assert struct.unpack("<Q", f.read(8)) == (1,), "unsupported index version"
            self.dtype = _DTYPES[struct.unpack("<B", f.read(1))[0]]
            length, _ = struct.unpack("<QQ", f.read(16))  # number of items, number of documents
            offset = f.tell()
        index = memoryview(np.memmap(prefix + ".idx", mode="r", order="C"))
        self.sizes = np.frombuffer(index, dtype=np.int32, count=length, offset=offset)
        self.pointers = np.frombuffer(index, dtype=np.int64, count=length, offset=offset + self.sizes.nbytes)
        self._data = memoryview(np.memmap(prefix + ".bin", mode="r", order="C"))

    # memoryviews cannot be pickled: DataLoader workers re-open the files instead
    def __getstate__(self):
        return self.prefix

    def __setstate__(self, prefix):
        self._open(prefix)

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, i):
        return np.frombuffer(self._data, dtype=self.dtype, count=int(self.sizes[i]), offset=int(self.pointers[i]))


class MMapIndexedDatasetBuilder:
    def __init__(self, prefix, dtype):
        self.prefix, self.dtype, self.sizes = prefix, dtype, []
        self._data = open(prefix + ".bin", "wb")

    def add_item(self, ids):
        array = np.asarray(ids, dtype=np.int64).astype(self.dtype)  # -1 wraps to the dtype's max value
        self._data.write(array.tobytes(order="C"))
        self.sizes.append(array.size)

    def finalize(self):
        self._data.close()
        sizes = np.array(self.sizes, dtype=np.int32)
        pointers = (np.cumsum(sizes, dtype=np.int64) - sizes) * np.dtype(self.dtype).itemsize
        doc_idx = np.zeros(1, dtype=np.int64)
        with open(self.prefix + ".idx", "wb") as f:
            f.write(_MAGIC + struct.pack("<QB", 1, _CODES[self.dtype]) + struct.pack("<QQ", len(sizes), len(doc_idx)))
            f.write(sizes.tobytes(order="C") + pointers.tobytes(order="C") + doc_idx.tobytes(order="C"))
