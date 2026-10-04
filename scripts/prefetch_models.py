"""Download and SHA-check the pose + matting models."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bswap.models import ModelStore
dest = os.environ.get("BSWAP_MODELS") or (sys.argv[1] if len(sys.argv) > 1 else "models")
store = ModelStore(Path(dest))
missing = store.missing()
if not missing:
    print("models already installed", store.dir)
else:
    def prog(name, done, total, bps, verifying):
        print(("verify" if verifying else "get"), name, int(done), "/", int(total), flush=True)
    store.ensure(missing, progress=prog)
print("OK", store.dir)
