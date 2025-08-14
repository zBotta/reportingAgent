
"""
reportParamGridSearch.py

Takes a Language Model and tests the given parameters in a Grid Search approach.

The indexes given are the selected rows of the database

"""
import sys
from conf.logManager import Logger
import logging

from pathlib import Path
root_path = str(Path(__file__).absolute().parent.parent)
sys.path.append(root_path) # import root project to env

logging.setLoggerClass(Logger)
log = logging.getLogger(__name__)
log.info(f"Added ENV = {root_path}")


from projectSetup import Setup
from mods.metricsEvaluator import MetricsEvaluator
from mods.dataHandler import DataHandler, Report
from mods.testBench import TestBench
from mods.reportGenerator import ReportGenerator
from mods.modelLoader import ModelLoader


def main(**kwargs):
    # Treat arguments
    print(f"Parameters passed to main script: \n{kwargs}")
    model_id = kwargs["model_id"][0]
    kwargs.pop("model_id")
    start_idx = kwargs["start_idx"][0]
    kwargs.pop("start_idx")
    end_idx = kwargs["end_idx"][0]
    kwargs.pop("end_idx")
    param_dict = kwargs.copy()

    dh = DataHandler()
    env = Setup()
    met_eval = MetricsEvaluator()
    # Load data
    df_reports = dh.import_reports() 

    # Load model
    ml = ModelLoader(model_id='microsoft/phi-2', device=env.device, torch_dtype=env.torch_dtype)
    param_dict = ml.get_dict_without_none_parameters(kwargs)
    print(f"Parameters passed to main script: \n{param_dict}")
    tb = TestBench(MetricsEvaluator = met_eval, DataHandler=dh, ModelLoader=ml)
    model, tokenizer = ml.load_model(hf_token=env.config["HF_TOKEN"])
    rg = ReportGenerator(model, tokenizer, output_type=Report)

    # Test different prompts and tuning parameters on model
    report_idx_list = list(range(start_idx, end_idx + 1))
    df_reports_filtered = df_reports.iloc[report_idx_list]
    tb.eval_gs_param(report_data=df_reports_filtered,
                     report_generator = rg,
                     prompt_method_list=["C"],
                     param_dict=param_dict)


if __name__ == "__main__":
    # Define arguments of the program
    import argparse
    from app.conf.projectConfig import Config as cf
    parser = argparse.ArgumentParser(prog="ReportParamGridSearch",
                                     description="Takes a Language Model and tests the given parameters in a Grid Search approach." \
                                     "The indexes given are the selected rows of the reference report database",
                                     argument_default=argparse.SUPPRESS) # This avoids assigning None values
    parser.add_argument("--model_id", type=str, nargs=1, required=True)
    parser.add_argument("--start_idx", type=int, nargs=1, required=True)
    parser.add_argument("--end_idx", type=int, nargs=1, required=True)
    for argument in cf.MODEL.PARAM_LIST:
        if argument == "do_sample":
            parser.add_argument("--" + argument, type=bool, nargs='+', required=False)
        elif argument == "top_k":
            parser.add_argument("--" + argument, type=int, nargs='+', required=False)
        else:
            parser.add_argument("--" + argument, type=float, nargs='+', required=False)
        
    args = parser.parse_args()
    main(**vars(args))
