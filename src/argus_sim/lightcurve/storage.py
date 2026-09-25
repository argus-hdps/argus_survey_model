"""File access for a visit store on local disk or in an S3 bucket, with a count of the read requests made.

Every read goes through ``Storage``. On S3 each read call is one ranged GET, so ``io_stats()["get_requests"]``
counts the GETs a query costs. Parquet files are opened with ``pre_buffer=True``: pyarrow then merges the column
chunks of a row group into one read, where without it it makes one read per column chunk.
"""

from __future__ import annotations

import io
import os
import threading

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


class _CountingFile:
    """A read-only file object over a pyarrow file that counts read calls and bytes (for ``pa.PythonFile``)."""

    def __init__(self, f: pa.NativeFile, stats: dict, lock: threading.Lock):
        self._f, self._stats, self._lock = f, stats, lock
        self._pos = 0
        self._size = f.size()
        self.closed = False

    def _count(self, n: int) -> None:
        with self._lock:
            self._stats["get_requests"] += 1
            self._stats["bytes_read"] += n

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self._pos
        n = max(0, min(n, self._size - self._pos))
        data = self._f.read_at(n, self._pos) if n else b""
        self._pos += len(data)
        self._count(len(data))
        return data

    def seek(self, pos: int, whence: int = 0) -> int:
        self._pos = pos if whence == 0 else self._pos + pos if whence == 1 else self._size + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def size(self) -> int:
        return self._size

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True
        self._f.close()


class Storage:
    """A store directory: a local path, or ``s3://bucket/prefix`` read through ``pyarrow.fs.S3FileSystem``."""

    def __init__(self, path: str, storage_options: dict | None = None):
        """Open ``path``. ``storage_options`` go to ``pyarrow.fs.S3FileSystem`` for an s3:// path."""
        self.stats = {"get_requests": 0, "bytes_read": 0}
        self._lock = threading.Lock()
        if path.startswith("s3://"):
            self.fs = pafs.S3FileSystem(**(storage_options or {}))
            self.root = path[len("s3://") :].rstrip("/")
            self.uri = path.rstrip("/")
            self.local = False
        else:
            if storage_options:
                raise ValueError("storage_options apply only to s3:// paths.")
            self.fs = pafs.LocalFileSystem()
            self.root = os.path.abspath(os.path.expanduser(path))
            self.uri = self.root
            self.local = True

    def _path(self, rel: str) -> str:
        return f"{self.root}/{rel}"

    def exists(self, rel: str) -> bool:
        """Return whether the store holds the file ``rel``."""
        return self.fs.get_file_info(self._path(rel)).type == pafs.FileType.File

    def read_bytes(self, rel: str) -> bytes:
        """Return the whole file ``rel`` in one read."""
        with self.fs.open_input_file(self._path(rel)) as f:
            data = f.read_at(f.size(), 0)
        with self._lock:
            self.stats["get_requests"] += 1
            self.stats["bytes_read"] += len(data)
        return data

    def load_npz(self, rel: str) -> dict:
        """Return the arrays of an .npz file as a dict."""
        with np.load(io.BytesIO(self.read_bytes(rel))) as z:
            return {k: z[k] for k in z.files}

    def parquet(self, rel: str) -> pq.ParquetFile:
        """Open a Parquet file; reading its footer costs one or two reads."""
        f = _CountingFile(self.fs.open_input_file(self._path(rel)), self.stats, self._lock)
        return pq.ParquetFile(pa.PythonFile(f, mode="r"), pre_buffer=True)

    def io_stats(self) -> dict:
        """Return the read requests (``get_requests``) and bytes read so far."""
        with self._lock:
            return dict(self.stats)
