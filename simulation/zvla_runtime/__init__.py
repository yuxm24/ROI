"""Dependency-light protocols shared with external simulator runtimes.

Modules in this package must remain importable without the Z-VLA training
stack. In particular, the sealed LIBERO simulator environment intentionally
does not carry PyTorch 2.9, mmengine, or Megatron-Core.
"""
