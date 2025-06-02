import os
import sys

from generating.runners.biobj import OptimizationRunner

if __name__ == "__main__":
    # Emulate Django’s trick of adding project root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    print(f"Project root added to sys.path: {os.path.join(os.path.dirname(__file__))}")

    # Run optimizations
    OptimizationRunner().run()
