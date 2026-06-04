import sys
import os

# Ensure project root is on sys.path when running tests from pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
