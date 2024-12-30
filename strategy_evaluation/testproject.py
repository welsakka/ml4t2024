"""
Student Name: Waleed Elsakka
GT User ID: welsakka3
GT ID: 904053428
"""

import ManualStrategy as ms
import datetime as dt
import sys
import subprocess

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python testproject.py")
        sys.exit(1)

    subprocess.run(["python", "ManualStrategy.py"])
    subprocess.run(["python", "experiment1.py"])
    subprocess.run(["python", "experiment2.py"])
    subprocess.run(["python", "StrategyLearner.py"])