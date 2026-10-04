import os, sys
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
try:
    from bswap.crashlog import install as _install
    _install()
except Exception:
    pass
from bswap.cli import main
raise SystemExit(main())
