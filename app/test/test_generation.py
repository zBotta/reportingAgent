"""
test_generation.py script
"""

import os
import shutil
from projectSetup import Setup
from conf.projectConfig import Config as cf
from mods.testBench import TestBench
from mods.metricsEvaluator import MetricsEvaluator
from mods.dataHandler import DataHandler, Report
from mods.promptGenerator import PromptGenerator
from mods.reportGenerator import ReportGenerator
from mods.modelLoader import ModelLoader
import time

import logging
from app.conf.logManager import Logger
logging.setLoggerClass(Logger)
log = logging.getLogger(__name__)

env = Setup()
met_eval = MetricsEvaluator()
# Load data
dh = DataHandler()

report_data = dh.import_reports()
# Load model
model_id = 'openai-community/gpt2' 
ml = ModelLoader(model_id=model_id, device=env.device, torch_dtype=env.torch_dtype)
model, tokenizer = ml.load_model(hf_token=env.config["HF_TOKEN"])

rg = ReportGenerator(model, tokenizer, output_type=Report)
tb = TestBench(MetricsEvaluator = met_eval, DataHandler=dh, ModelLoader=ml)

# For clearing results and full method is working
app_folder_dest = cf.TESTS.T_TB_RESULTS_F
filename_prefix = cf.TESTS.T_TB_FILENAME_PREFIX
folder_path = os.path.join(cf.APP_PATH, app_folder_dest)

def clear_test_bench_folder():
    """ Deletes folder with previous tests files """
    if os.path.isdir(folder_path): 
        shutil.rmtree(folder_path) 

def test_structured_outputs():
    """ Test that the structured outputs work with outlines """
    report_idx = 20
    row = report_data.loc[report_idx]
    # Generate prompt
    five_ws = row.loc['what':'contingency_actions']
    prompt_gen = PromptGenerator(**five_ws.to_dict())
    prompt_method = cf.MODEL.DEFAULT_PROMPT_METHOD
    gen_prompt = prompt_gen.create_prompt(prompt_method)
    # Generate parameters
    gen_param = ml.get_default_tunable_parameters()
    gen_param.update({"max_new_tokens": 300})
    tb.clear_df_results()
    res = tb.generate_one_param_set(res={},
                                    gen_prompt = gen_prompt,
                                    gen_param = gen_param,
                                    row = row,
                                    report_generator = rg,
                                    prompt_method = prompt_method)
    title = res["title"]
    report = res["report"]
    assert ( len(title) > 0 and len(report) > 0 )

def test_several_prompts_default_param():
    """ Testing prompts A, B and C with default parameters
    prompt
    param_dict = {}     
    """
    report_idx_list = [20]
    report_data_filtered = report_data.iloc[report_idx_list]
    clear_test_bench_folder()
    tb.eval_gs_param(report_data=report_data_filtered,
                     report_generator = rg,
                     prompt_method_list=cf.TEST_BENCH.PROMPT_METHODS,
                     param_dict={},
                     xlsx_file_name = filename_prefix,
                     app_folder_destination = app_folder_dest)
    
    # Check the file creation, check the string prefix in the filename
    for filenames in os.listdir(folder_path):
        assert filename_prefix in filenames

def test_param_grid_search():
    start_time = time.time()
    # Get a part of the the Database
    report_idx_list = [20]
    report_data_filtered = report_data.iloc[report_idx_list]
    clear_test_bench_folder()
    tb.eval_gs_param(report_data=report_data_filtered,
                     report_generator = rg,
                     prompt_method_list=["C"],
                     param_dict={"temperature": [0.7, 1.3],
                                 "top_p": [0.6, 1],
                                 "max_new_tokens": [300]},
                     xlsx_file_name = filename_prefix,
                     app_folder_destination = app_folder_dest )

    log.info("test_grid_search time --- %s seconds ---" % (time.time() - start_time))

    # Check the file creation, check the string prefix in the filename
    for filenames in os.listdir(folder_path):
        assert filename_prefix in filenames

def test_threaded_grid_search():
    start_time = time.time()
    report_idx_list = [20]
    report_data_filtered = report_data.iloc[report_idx_list]
    clear_test_bench_folder()
    df_res = tb.eval_gs_param_threaded(report_data=report_data_filtered,
                                       report_generator = rg,
                                       prompt_method_list=["C"],
                                       param_dict={"temperature": [0.7, 1.3],
                                                   "top_p": [0.6, 1],
                                                   "max_new_tokens": [300]},
                                       xlsx_file_name = filename_prefix,
                                       app_folder_destination = app_folder_dest,
                                       max_workers = 4 )

    log.info("test_thread time --- %s seconds ---" % (time.time() - start_time))
    
    # Check the file creation, check the string prefix in the filename
    for filenames in os.listdir(folder_path):
        assert filename_prefix in filenames