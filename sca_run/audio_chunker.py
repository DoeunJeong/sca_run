from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class PCMChunker:
    """Accumulate PCM16LE bytes until a fixed chunk size is reached."""

    chunk_bytes: int

    def __post_init__(self) -> None:
        if self.chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be > 0")
        self._buf = bytearray()

    def reset(self) -> None:
        self._buf.clear()

    def feed(self, data: bytes) -> Iterator[bytes]:
        """Feed raw PCM bytes; yield full chunks."""
        if not data:
            return
        self._buf.extend(data)
        while len(self._buf) >= self.chunk_bytes:
            chunk = bytes(self._buf[: self.chunk_bytes])
            del self._buf[: self.chunk_bytes]
            yield chunk

    def flush(self, pad: bool = False, pad_byte: int = 0) -> Optional[bytes]:
        """Return remaining bytes.

        If pad=True, pads the remainder to a full chunk and returns it.
        """
        if not self._buf:
            return None
        if not pad:
            out = bytes(self._buf)
            self._buf.clear()
            return out
        # pad to full chunk
        missing = self.chunk_bytes - len(self._buf)
        self._buf.extend(bytes([pad_byte]) * missing)
        out = bytes(self._buf)
        self._buf.clear()
        return out
