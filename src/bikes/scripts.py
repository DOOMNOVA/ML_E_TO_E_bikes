#Scripts for CLI  application

import warnings

#disable mlflow warnings
warnings.filterwarnings(action ="ignore", category=UserWarning)


import argparse
import json
import sys

from bikes import settings
from bikes.io import configs

#parsers

parser = argparse.ArgumentParser(description="Run an AI/ML job from YAML/JSON configs.")
parser.add_argument("files",nargs="*",help="Config files for the job(local path only).")
parser.add_argument("-e", "--extras",nargs="*",default=[],help="Additional config strings for the job.")
parser.add_argument("-s","--schema",action="store_true",help="Print settings schema and exit")


#scripts
def main(argv: list[str] | None = None) -> int:
    args = parser.parse_args(argv)
    if args.schema:
        #print schema settings here
        return 0
    #execute main application logic 
    
