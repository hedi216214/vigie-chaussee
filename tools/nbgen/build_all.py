"""Régénère tous les notebooks : python tools/nbgen/build_all.py"""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for f in ["nb00.py", "nb01_02.py", "nb03.py", "nb04.py", "nb05.py"]:
    runpy.run_path(os.path.join(HERE, f), run_name="__main__")
print("notebooks écrits dans", os.path.abspath(os.path.join(HERE, "..", "..", "notebooks")))
