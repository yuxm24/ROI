# SPDX-License-Identifier: Apache-2.0
"""Small rank-aware logging facade for training processes."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator


def _rank_filter(enabled: bool) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        if enabled:
            return function

        @wraps(function)
        def skip(*args: Any, **kwargs: Any) -> None:
            return None

        return skip

    return decorate


class _RankLogger:
    def __init__(self, name: str, rank: int, local_rank: int, world_size: int) -> None:
        self._rank = rank
        self._local_rank = local_rank
        self._world_size = world_size
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    def _emit(self, level: int, message: object, *args: Any, **kwargs: Any) -> None:
        depth = max(0, int(kwargs.pop("ctx_level", 0)))
        prefix = "  " * depth
        self.logger.log(level, prefix + str(message), *args, **kwargs)

    def debug(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.INFO, message, *args, **kwargs)

    def warning(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.WARNING, message, *args, **kwargs)

    def error(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.ERROR, message, *args, **kwargs)

    def critical(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.CRITICAL, message, *args, **kwargs)

    @property
    def rank_zero_only(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return _rank_filter(self._rank == 0)

    @property
    def local_zero_only(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return _rank_filter(self._local_rank == 0)

    @staticmethod
    @contextmanager
    def _first(selected: bool) -> Iterator[None]:
        import torch.distributed as distributed

        active = distributed.is_available() and distributed.is_initialized()
        if active and not selected:
            distributed.barrier()
        try:
            yield
        finally:
            if active and selected:
                distributed.barrier()

    @property
    def rank_zero_first(self) -> Callable[[], Iterator[None]]:
        return lambda: self._first(self._rank == 0)

    @property
    def local_zero_first(self) -> Callable[[], Iterator[None]]:
        return lambda: self._first(self._local_rank == 0)

    def is_rank_zero(self) -> bool:
        return self._rank == 0

    def rank(self) -> int:
        return self._rank

    def local_rank(self) -> int:
        return self._local_rank

    def world_size(self) -> int:
        return self._world_size


class PureOverwatch(_RankLogger):
    def __init__(self, name: str) -> None:
        super().__init__(name, rank=0, local_rank=0, world_size=1)


class DistributedOverwatch(_RankLogger):
    def __init__(self, name: str) -> None:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
            raise ValueError("Invalid distributed rank metadata")
        super().__init__(name, rank=rank, local_rank=local_rank, world_size=world_size)


def initialize_overwatch(name: str) -> _RankLogger:
    """Select the logger from the torchrun environment."""
    if "WORLD_SIZE" in os.environ:
        return DistributedOverwatch(name)
    return PureOverwatch(name)
