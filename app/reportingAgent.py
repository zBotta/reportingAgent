
"""
reportAgent.py

main app script

"""
from conf.logManager import Logger
import logging

import sys
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


def main():
    env = Setup()
    met_eval = MetricsEvaluator()
    # Load data
    dh = DataHandler()
    df_reports = dh.import_reports() 
    # Load model
    tb = TestBench(MetricsEvaluator = met_eval, DataHandler=dh)
    ml = ModelLoader(model_id='microsoft/phi-2', device=env.device, torch_dtype=env.torch_dtype)
    model, tokenizer = ml.load_model(hf_token=env.config["HF_TOKEN"])
    rg = ReportGenerator(model, tokenizer, output_type=Report)

    # Test different prompts on model
    report_idx_list = [20]
    tb.eval_gs_param(report_data=df_reports,
                     report_idx_list = report_idx_list, 
                     report_generator = rg,
                     prompt_method_list=["C"],
                     param_dict={"temperature": [0.7, 1.3],
                                 "top_p": [0.6, 1]} )


if __name__ == "__main__":
    main()