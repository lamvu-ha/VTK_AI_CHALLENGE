import os
import sys

# ── Disable TensorFlow in transformers & force PyTorch ──
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # no CUDA DLL scan
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch as _t  # noqa: F401
except Exception:
    pass


# Redirect Hugging Face cache to local data/hf_cache on drive E: (avoid drive C disk space limit)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_HF_CACHE = os.path.join(_PROJECT_ROOT, "data", "hf_cache")
os.makedirs(_HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = _HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = _HF_CACHE

# Reconfigure stdout/stderr to utf-8 for Windows console support
if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.ranking_editor.app_gui import main

if __name__ == "__main__":
    print("=" * 60)
    print("  LAUNCHING AIC 2026 VIDEO RETRIEVAL GUI STUDIO")
    print("=" * 60)
    main()
