import socket

import pytest

from multimodal_db.store import Store
from multimodal_db.testing import FakeEmbedding

_real_connect = socket.socket.connect
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Anything leaving the box fails loudly. Loopback stays open: asyncio's Windows
    self-pipe is a socketpair, and one probe test wants a refused local port."""

    def connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            raise RuntimeError(f"network access during tests: {address!r}")
        return _real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "data") as s:
        yield s


@pytest.fixture
def embed():
    return FakeEmbedding()


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "gfx1151.md").write_text(
        "# Strix Halo\n\ngfx1151 is the Strix Halo APU target. ROCm on Windows supports it from 7.1.\n",
        "utf-8",
    )
    (d / "hip-errors.md").write_text(
        "# HIP error codes\n\nhipErrorNoBinaryForGpu means the code object was not built for this gfx target.\n",
        "utf-8",
    )
    (d / "notes").mkdir()
    (d / "notes" / "triton.txt").write_text(
        "Triton on ROCm needs the AMD backend and a matching gfx arch in TRITON_OVERRIDE_ARCH.\n", "utf-8"
    )
    (d / "page.html").write_text(
        "<html><head><title>Compat matrix</title><style>x{}</style></head>"
        "<body><h1>Matrix</h1><p>RDNA 3.5 rows: gfx1150 and gfx1151.</p><script>var a=1;</script></body></html>",
        "utf-8",
    )
    (d / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    (d / ".hidden.md").write_text("skip me", "utf-8")
    return d
