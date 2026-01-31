from pathlib import Path
import sys
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.llm import LLM
# %%-------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    llm = LLM()
    print(f"Config path: {llm.config_path}")
    v = llm.get_embedding("Hello world")
    print(f"Embedding dim: {len(v)}")
