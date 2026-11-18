"""Small file helpers shared by manifests, checkpoints and evaluation records."""

from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def atomic_output(path: Path, *, binary: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {} if binary else {"encoding": "utf-8", "newline": ""}
    with tempfile.NamedTemporaryFile(
        mode="wb" if binary else "w", dir=path.parent,
        prefix=f".{path.name}.", delete=False, **options
    ) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            temporary.replace(path)
        finally:
            handle.close()
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    with atomic_output(path) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
