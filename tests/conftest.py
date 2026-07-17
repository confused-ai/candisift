"""Pytest bootstrap: force the deterministic offline stub LLM.

config.py calls load_dotenv() at import, which would otherwise pull a real
ANTHROPIC_API_KEY from .env and make the suite hit the live Claude API (slow,
non-deterministic, costs money). Blank the provider keys here — this runs before
any test module imports ats.config, and load_dotenv(override=False) won't
clobber an already-set var. Result: has_llm is False, the stub is used.
"""
import os

for _var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
    os.environ[_var] = ""

# Pin the model defaults too: an operator's .env (e.g. CANDISIFT_PERSONA_MODEL=all-free)
# would otherwise leak into Settings() and make cost-estimate assertions
# non-deterministic. Pre-set wins over load_dotenv(override=False).
os.environ["CANDISIFT_PERSONA_MODEL"] = "claude-haiku-4-5"
os.environ["CANDISIFT_SYNTH_MODEL"] = "claude-opus-4-8"
