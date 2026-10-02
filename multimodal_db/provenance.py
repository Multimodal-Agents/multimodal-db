"""Provenance: an append-only, hash-chained ledger of what changed, from where, and when.

Each record is content-addressed: SHA-256 over a canonical JSON envelope of its chain, the
previous digest on that chain, any other parents, a header (where it came from) and a payload
(what it says). Key order never changes a digest, and a timestamp is kept beside the record,
outside the hash. Opening a ledger recomputes every digest and refuses a file whose complete
lines were edited; only a torn last line (a crash mid-write) is dropped. The scheme follows
the v3 episode store in donaldfilimon/wdbx, minus its governance layer.

Stdlib only, so anything can depend on it."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

DOMAIN = "mmdb-provenance-v1"
SCHEMA = 1
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = {"seq", "chain", "prev", "parents", "header", "payload", "digest", "recorded_at"}


class LedgerCorrupt(ValueError):
    def __init__(self, path: Path, line: int, reason: str):
        super().__init__(f"{path}: line {line}: {reason}")
        self.path, self.line, self.reason = path, line, reason


class Replay(ValueError):
    pass


def _check(obj: Any, depth: int = 0) -> None:
    # floats have no single canonical spelling, so a digest over one would not be portable
    if depth > 32:
        raise ValueError("nested deeper than 32")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"key {k!r} is not a string")
            _check(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _check(v, depth + 1)
    elif isinstance(obj, float):
        raise ValueError(f"float {obj!r} has no canonical form; store it as a string or int")
    elif obj is not None and not isinstance(obj, (str, int, bool)):
        raise ValueError(f"{type(obj).__name__} is not canonical JSON")


def canonical(obj: Any) -> bytes:
    """Sorted keys, no whitespace, UTF-8. The bytes a digest is taken over."""
    _check(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digests(values: Iterable[str]) -> list[str]:
    out = sorted(set(values))
    for d in out:
        if not _DIGEST_RE.match(d):
            raise ValueError(f"{d!r} is not a sha256 hex digest")
    return out


def digest(chain: str, header: dict[str, Any], payload: dict[str, Any], *, prev: str | None = None,
           parents: Iterable[str] = (), domain: str = DOMAIN, schema: int = SCHEMA) -> str:
    envelope = {"domain": domain, "schema": schema, "chain": chain, "prev": prev,
                "parents": _digests(parents), "header": header, "payload": payload}
    return hashlib.sha256(canonical(envelope)).hexdigest()


def fingerprint(data: str | bytes) -> str:
    """SHA-256 of a file's bytes. A renderer keeps this for every file it writes."""
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def changed(path: str | os.PathLike, expected: str) -> bool:
    """True when the file is gone or no longer what was written: a hand edit to pull back."""
    p = Path(path)
    return not p.is_file() or fingerprint(p.read_bytes()) != expected


@dataclass
class Record:
    seq: int
    chain: str
    digest: str
    header: dict[str, Any]
    payload: dict[str, Any]
    prev: str | None = None
    parents: list[str] = field(default_factory=list)
    recorded_at: str = ""

    def computed_digest(self) -> str:
        return digest(self.chain, self.header, self.payload, prev=self.prev, parents=self.parents)

    def receipt(self) -> dict[str, Any]:
        """Ids and digests only, no header or payload content. ``event`` rides along when the
        payload names one."""
        out = {"seq": self.seq, "chain": self.chain, "digest": self.digest, "prev": self.prev,
               "parents": list(self.parents), "recorded_at": self.recorded_at}
        if "event" in self.payload:
            out["event"] = self.payload["event"]
        return out


class Ledger:
    """One JSONL file, verified in full when opened. Single writer: the store already is."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._records: list[Record] = []
        self._heads: dict[str, int] = {}
        self._by_chain: dict[str, list[int]] = {}
        self._seen: set[str] = set()
        self._load()

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records)

    def _index(self, rec: Record) -> None:
        i = len(self._records)
        self._records.append(rec)
        self._heads[rec.chain] = i
        self._by_chain.setdefault(rec.chain, []).append(i)
        self._seen.add(rec.digest)

    def _verify(self, rec: Record, line: int) -> None:
        if rec.seq != len(self._records) + 1:
            raise LedgerCorrupt(self.path, line, f"seq {rec.seq}, expected {len(self._records) + 1}")
        head = self.head(rec.chain)
        if rec.prev != (head.digest if head else None):
            raise LedgerCorrupt(self.path, line, f"prev does not match the head of chain {rec.chain!r}")
        if rec.computed_digest() != rec.digest:
            raise LedgerCorrupt(self.path, line, "digest does not match the record; it was edited")
        if rec.digest in self._seen:
            raise LedgerCorrupt(self.path, line, "duplicate digest")

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        lines = raw.split(b"\n")
        torn = lines.pop()  # b"" when the file ends cleanly
        for n, line in enumerate(lines, 1):
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict) or set(obj) != _FIELDS:
                    raise ValueError("unexpected fields")
                rec = Record(**obj)
            except (ValueError, TypeError) as e:
                raise LedgerCorrupt(self.path, n, f"unreadable: {e}") from e
            self._verify(rec, n)
            self._index(rec)
        if torn:
            try:
                json.loads(torn)
            except ValueError:
                pass
            else:  # a whole record missing its newline was touched by hand, not torn by a crash
                raise LedgerCorrupt(self.path, len(lines) + 1, "complete record with no newline")
            # a crash mid-append; everything before it verified, so drop only the tail
            with self.path.open("r+b") as f:
                f.truncate(len(raw) - len(torn))

    def verify(self) -> int:
        """Re-read the file from disk and check every record. Returns the count."""
        return len(Ledger(self.path))

    def head(self, chain: str) -> Record | None:
        i = self._heads.get(chain)
        return None if i is None else self._records[i]

    def history(self, chain: str) -> list[Record]:
        return [self._records[i] for i in self._by_chain.get(chain, [])]

    def append(self, chain: str, header: dict[str, Any], payload: dict[str, Any],
               parents: Iterable[str] = ()) -> Record:
        head = self.head(chain)
        prev = head.digest if head else None
        parents = _digests(parents)
        d = digest(chain, header, payload, prev=prev, parents=parents)
        if d in self._seen:
            raise Replay(f"record {d} is already in {self.path.name}")
        rec = Record(seq=len(self._records) + 1, chain=chain, digest=d, header=header, payload=payload,
                     prev=prev, parents=parents,
                     recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        line = json.dumps(asdict(rec), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as f:
            f.write(line.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        self._index(rec)
        return rec
