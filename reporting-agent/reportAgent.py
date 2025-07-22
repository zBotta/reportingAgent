
"""
reportAgent.py

main app script

"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent)) # import root project to env

from setup import Setup
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
    df_reports = dh.import_reports() # CHECK: If dataset increases, move from github repo
    # Load model
    tb = TestBench(MetricsEvaluator = met_eval, DataHandler=dh)
    ml = ModelLoader(model_id='microsoft/phi-2', device=env.device, torch_dtype=env.torch_dtype)
    model, tokenizer = ml.load_model(hf_token=env.config["HF_TOKEN"])
    rg = ReportGenerator(model, tokenizer, output_type=Report)

    # Test different prompts on model
    report_idx_list = [20]
    df_prompts = tb.eval_diff_prompts(df_reports, 
                                      report_idx_list = report_idx_list, 
                                      report_generator = rg )
    df_prompts


if __name__ == "__main__":
    main()