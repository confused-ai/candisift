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
