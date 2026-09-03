"""mmdb.toml: where the store lives and which endpoints serve models.

File lookup: the ``path`` argument, then ``MMDB_CONFIG``, then ``./mmdb.toml``. With no
file at all you get one ``local`` endpoint that embeds in-process with
``unsloth/bge-small-en-v1.5`` (needs the ``local`` extra).
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

KINDS = ("openai", "huggingface", "ollama", "fake")
DEFAULT_EMBED_MODEL = "unsloth/bge-small-en-v1.5"


class ConfigError(ValueError):
    pass


@dataclass
class Endpoint:
    """One place models are served from. ``kind`` picks the LlamaIndex classes."""

    name: str
    kind: str
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    chat_model: str | None = None
    embed_model: str | None = None
    timeout: float = 120.0
    context_window: int = 8192

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ConfigError(f"endpoint {self.name!r}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind in ("openai", "ollama") and not self.base_url:
            raise ConfigError(f"endpoint {self.name!r}: kind {self.kind!r} needs base_url")
        if self.base_url:
            self.base_url = self.base_url.rstrip("/")

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or None
        return None


@dataclass
class Config:
    root: Path
    endpoints: dict[str, Endpoint] = field(default_factory=dict)
    default_embed: str | None = None
    default_llm: str | None = None
    chunk_size: int = 512
    chunk_overlap: int = 64
    path: Path | None = None

    def endpoint(self, name: str | None, role: str) -> Endpoint:
        """The endpoint for ``role`` ("embed" or "llm"): explicit name, then
        ``MMDB_EMBED`` / ``MMDB_LLM``, then ``[defaults]`` in the file."""
        default = self.default_embed if role == "embed" else self.default_llm
        name = name or os.environ.get(f"MMDB_{role.upper()}") or default
        if not name:
            raise ConfigError(
                f"no {role} endpoint chosen: pass --{role}, set MMDB_{role.upper()}, "
                f"or set [defaults].{role} in mmdb.toml"
            )
        try:
            return self.endpoints[name]
        except KeyError:
            known = ", ".join(self.endpoints) or "none"
            raise ConfigError(f"unknown endpoint {name!r}; configured: {known}") from None


def default_config(root: Path | None = None) -> Config:
    local = Endpoint(name="local", kind="huggingface", embed_model=DEFAULT_EMBED_MODEL)
    return Config(root=root or Path("data"), endpoints={"local": local}, default_embed="local")


def find_config_path(explicit: str | os.PathLike | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        return p
    for candidate in (os.environ.get("MMDB_CONFIG"), "mmdb.toml"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def load_config(path: str | os.PathLike | None = None, root: str | os.PathLike | None = None) -> Config:
    """Root precedence: the ``root`` argument, ``MMDB_ROOT``, ``[store].root``, ``./data``."""
    found = find_config_path(path)
    cfg = parse_config(found) if found else default_config()
    if root:
        cfg.root = Path(root)
    elif os.environ.get("MMDB_ROOT"):
        cfg.root = Path(os.environ["MMDB_ROOT"])
    return cfg


_ENDPOINT_KEYS = {f.name for f in fields(Endpoint)} - {"name"}


def parse_config(path: Path) -> Config:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    store = raw.get("store", {})
    defaults = raw.get("defaults", {})

    endpoints: dict[str, Endpoint] = {}
    for name, spec in raw.get("endpoints", {}).items():
        unknown = set(spec) - _ENDPOINT_KEYS
        if unknown:
            raise ConfigError(f"endpoint {name!r}: unknown keys {sorted(unknown)}")
        endpoints[name] = Endpoint(name=name, **spec)
    if not endpoints:
        endpoints = default_config().endpoints

    root = Path(store.get("root", "data"))
    if not root.is_absolute():
        root = (path.parent / root).resolve()

    cfg = Config(
        root=root,
        endpoints=endpoints,
        default_embed=defaults.get("embed"),
        default_llm=defaults.get("llm"),
        chunk_size=int(store.get("chunk_size", 512)),
        chunk_overlap=int(store.get("chunk_overlap", 64)),
        path=path,
    )
    for role, chosen in (("embed", cfg.default_embed), ("llm", cfg.default_llm)):
        if chosen and chosen not in endpoints:
            raise ConfigError(f"[defaults].{role} = {chosen!r} is not a configured endpoint")
    return cfg
