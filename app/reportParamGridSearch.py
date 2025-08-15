
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
import time

from app.conf.projectConfig import Config as cf


def main(**kwargs):
    # clear log
    Logger.clear()
    # Treat arguments
    print(f"Parameters passed to main script: \n{kwargs}")
    log.info(f"Parameters passed to main script: \n{kwargs}")
    model_id = kwargs["model_id"][0]
    kwargs.pop("model_id")
    start_idx = kwargs["start_idx"][0]
    kwargs.pop("start_idx")
    end_idx = kwargs["end_idx"][0]
    kwargs.pop("end_idx")
    prompt_method_list = kwargs["prompt_method"]
    kwargs.pop("prompt_method")
    max_workers = kwargs["max_workers"][0]
    kwargs.pop("max_workers")
    is_threaded_process = kwargs["use_threaded"]
    kwargs.pop("use_threaded")
    dataset_filename = kwargs["dataset_filename"]
    kwargs.pop("dataset_filename")
    param_dict = kwargs.copy()

    start_time = time.time()
    dh = DataHandler()
    env = Setup()
    met_eval = MetricsEvaluator()
    # Load data
    df_reports = dh.import_reports(xlsx_file_name=dataset_filename)
    ml = ModelLoader(model_id=model_id, device=env.device, torch_dtype=env.torch_dtype)
    print(f"Generation parameters: \n{param_dict}")
    tb = TestBench(MetricsEvaluator = met_eval, DataHandler=dh, ModelLoader=ml)
    # Print out the expected number of combinations (output rows on results df)
    # Load LM
    model, tokenizer = ml.load_model(hf_token=env.config["HF_TOKEN"])
    rg = ReportGenerator(model, tokenizer, output_type=Report)

    # Test different prompts and tuning parameters on model
    report_idx_list = list(range(start_idx, end_idx + 1))
    tb.print_number_of_combinations(report_data=report_idx_list, param_dict=param_dict, prompt_method_list=prompt_method_list)
    df_reports_filtered = df_reports.iloc[report_idx_list]
    if is_threaded_process:
        tb.eval_gs_param_threaded(report_data=df_reports_filtered,
                                report_generator = rg,
                                prompt_method_list=prompt_method_list,
                                param_dict=param_dict,
                                max_workers=max_workers)
    else:
        tb.eval_gs_param(report_data=df_reports_filtered,
                         report_generator = rg,
                         prompt_method_list=prompt_method_list,
                         param_dict=param_dict)
    print("reportParamGridSearch time --- %s minutes ---" % ((time.time() - start_time)/60))
    log.info("reportParamGridSearch time --- %s minutes ---" % ((time.time() - start_time)/60))

if __name__ == "__main__":
    # Define arguments of the program
    import argparse
    parser = argparse.ArgumentParser(prog="ReportParamGridSearch",
                                     description="Takes a Language Model and tests the given parameters in a Grid Search approach." \
                                     "The indexes given are the selected rows of the reference report database",
                                     argument_default=argparse.SUPPRESS) # This avoids assigning None values
    parser.add_argument("--model_id", type=str, nargs=1, required=True)
    parser.add_argument("--start_idx", type=int, nargs=1, required=True)
    parser.add_argument("--end_idx", type=int, nargs=1, required=True)
    parser.add_argument("--prompt_method", type=str, nargs="+", required=True)
    parser.add_argument("--max_workers", type=int, nargs=1, required=False, default=4)
    parser.add_argument("--dataset_filename", type=str, required=True)
    parser.add_argument("--use_threaded", type=bool, nargs=1, required=True)
    for argument in cf.MODEL.PARAM_LIST:
        if argument == "do_sample":
            parser.add_argument("--" + argument, type=bool, nargs='+', required=False)
        elif argument == "top_k":
            parser.add_argument("--" + argument, type=int, nargs='+', required=False)
        else:
            parser.add_argument("--" + argument, type=float, nargs='+', required=False)
        
    args = parser.parse_args()
    main(**vars(args))

# Call example: 
# python app/reportParamGridSearch.py --model_id microsoft/phi-2 --prompt_method B C --max_workers 4 --dataset_filename pharma_dev_reports_collection.xlsx --start_idx 1 --end_idx 5  --temperature 0.7 1.0 1.3 --top_p 0.3 0.6 0.9 --top_k 30 50 70 --max_new_tokens 300 --do_sample True