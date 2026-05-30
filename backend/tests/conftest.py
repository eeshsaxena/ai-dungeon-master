"""pytest configuration — adds backend dir to sys.path."""
import sys
from pathlib import Path

BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND))
