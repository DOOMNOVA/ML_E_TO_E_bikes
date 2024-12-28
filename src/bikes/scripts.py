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
    """
    The main function parses arguments, handles schema settings, parses files and strings, merges
    configurations, validates models, and runs the application logic.
    
    :param argv: The `argv` parameter in the `main` function is a list of strings representing the
    command-line arguments passed to the script. It can be `None` if no arguments are provided when the
    function is called
    :type argv: list[str] | None
    :return: The `main` function returns an integer value of 0.
    """
    args = parser.parse_args(argv)
    if args.schema:
        #print schema settings here
        schema = settings.MainSettings.model_json_schema()
        json.dump(schema,sys.stdout,indent=4)
        return 0
    #execute main application logic 
    files = [configs.parse_file(file) for file in args.files]
    strings = [configs.parse_string(string) for string in args.extras]
    if not (files or strings):
        raise RuntimeError("No configs provided.")
    
    config = configs.merge_configs[*files,*strings]
    object_ = configs.to_object(config)
    setting = settings.MainSettings.model_validate(object_)
    with setting.job as runner:
        runner.run()
        return 0
    
    
    
