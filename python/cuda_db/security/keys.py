"""Strict hashed API-key storage and its small administration CLI.

The complete bearer token is shown only by ``generate``.  Stores contain an
identifier and SHA-256 digest, never a token or secret that can authenticate.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1
KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class KeyFileError(ValueError):
    """An API-key store is missing, unreadable, or violates its schema."""


@dataclass(frozen=True)
class ApiKeyRecord:
    key_id: str
    sha256: str


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise KeyFileError("API-key store contains a duplicate JSON field")
        result[name] = value
    return result


def validate_key_id(key_id: str) -> str:
    if not isinstance(key_id, str) or KEY_ID_PATTERN.fullmatch(key_id) is None:
        raise KeyFileError(
            "key id must contain 1 to 64 ASCII letters, digits, underscores, or hyphens"
        )
    return key_id


def load_key_file(
    path: str | os.PathLike[str], *, require_nonempty: bool = True
) -> tuple[ApiKeyRecord, ...]:
    """Load and strictly validate a versioned key store."""
    store_path = Path(path)
    try:
        raw = store_path.read_bytes()
    except OSError as exc:
        raise KeyFileError("API-key store could not be read") from exc
    if not raw.strip():
        raise KeyFileError("API-key store is empty")
    try:
        text = raw.decode("utf-8")
        document = json.loads(text, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KeyFileError("API-key store is not valid UTF-8 JSON") from exc
    return _validate_document(document, require_nonempty=require_nonempty)


def _validate_document(
    document: Any, *, require_nonempty: bool
) -> tuple[ApiKeyRecord, ...]:
    if not isinstance(document, dict) or set(document) != {"schema_version", "keys"}:
        raise KeyFileError("API-key store has an invalid top-level schema")
    if type(document["schema_version"]) is not int or document[
        "schema_version"
    ] != SCHEMA_VERSION:
        raise KeyFileError("unsupported API-key store schema version")
    entries = document["keys"]
    if not isinstance(entries, list):
        raise KeyFileError("API-key store keys must be a list")
    if require_nonempty and not entries:
        raise KeyFileError("API-key store must contain at least one key")

    records: list[ApiKeyRecord] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"id", "sha256"}:
            raise KeyFileError("API-key record has an invalid schema")
        key_id = validate_key_id(entry["id"])
        digest = entry["sha256"]
        if not isinstance(digest, str) or DIGEST_PATTERN.fullmatch(digest) is None:
            raise KeyFileError("API-key digest must be 64 lowercase hexadecimal characters")
        if key_id in seen:
            raise KeyFileError("API-key store contains a duplicate key id")
        seen.add(key_id)
        records.append(ApiKeyRecord(key_id=key_id, sha256=digest))
    return tuple(records)


def _document(records: Sequence[ApiKeyRecord]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "keys": [{"id": record.key_id, "sha256": record.sha256} for record in records],
    }


def _write_key_file(path: Path, records: Sequence[ApiKeyRecord]) -> None:
    """Atomically replace ``path`` with a mode-0600 key store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(_document(records), indent=2) + "\n").encode("utf-8")
    descriptor = -1
    temporary_name = ""
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as temporary:
            descriptor = -1
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = ""
        os.chmod(path, 0o600)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise KeyFileError("API-key store could not be written") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


@contextmanager
def _store_lock(path: Path):
    """Serialize administrative read-modify-replace operations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise KeyFileError("API-key store could not be locked") from exc
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def generate_key(key_id: str, store: str | os.PathLike[str]) -> str:
    """Add a key and return its one-time plaintext bearer token."""
    key_id = validate_key_id(key_id)
    store_path = Path(store)
    with _store_lock(store_path):
        if store_path.exists():
            records = list(load_key_file(store_path, require_nonempty=False))
        else:
            records = []
        if any(record.key_id == key_id for record in records):
            raise KeyFileError("API-key id already exists")

        token = f"{key_id}.{secrets.token_urlsafe(32)}"
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        records.append(ApiKeyRecord(key_id=key_id, sha256=digest))
        _write_key_file(store_path, records)
    return token


def revoke_key(key_id: str, store: str | os.PathLike[str]) -> None:
    """Remove a key by id, leaving a valid (possibly empty) management store."""
    key_id = validate_key_id(key_id)
    store_path = Path(store)
    with _store_lock(store_path):
        records = list(load_key_file(store_path, require_nonempty=False))
        remaining = [record for record in records if record.key_id != key_id]
        if len(remaining) == len(records):
            raise KeyFileError("API-key id does not exist")
        _write_key_file(store_path, remaining)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage hashed cuda-db API keys")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "revoke"):
        command = commands.add_parser(name)
        command.add_argument("--key-id", required=True)
        command.add_argument("--store", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "generate":
            # This is the sole intentional plaintext-token output path.
            print(generate_key(arguments.key_id, arguments.store))
        else:
            revoke_key(arguments.key_id, arguments.store)
    except KeyFileError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
