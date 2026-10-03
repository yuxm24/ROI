"""Compatibility entry point for source-checkout torchrun commands.

The installable training implementation lives in ``zvla.posttrain.train``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zvla.posttrain.train import main  # noqa: E402

if __name__ == "__main__":
    main()
