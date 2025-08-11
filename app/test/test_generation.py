
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
app_folder_dest = cf.TEST_BENCH.TB_RESULTS_F
filename_prefix = cf.TEST_BENCH.FILENAME_PREFIX
folder_path = os.path.join(cf.APP_PATH, app_folder_dest)

def clear_test_bench_folder():
    """ Deletes folder with previous tests files """
    if os.path.isdir(folder_path): 
        shutil.rmtree(folder_path) 

def test_structured_outputs():
    """ Test that the structured outputs work with outlines """
    report_idx = 20
    row = report_data.loc[report_idx, 'what':'contingency_actions']
    prompt_gen = PromptGenerator(**row.to_dict())
    prompt_method = cf.MODEL.DEFAULT_PROMPT_METHOD
    gen_prompt = prompt_gen.create_prompt(prompt_method)
    gen_param = ml.get_default_tunable_parameters()
    tb.clear_df_results()
    title, report = tb.generate_one_param_set(res={},
                                              gen_prompt = gen_prompt,
                                              gen_param = gen_param,
                                              report_data = report_data,
                                              report_idx = report_idx,
                                              report_generator = rg,
                                              prompt_method = prompt_method)
    assert ( len(title) > 0 and len(report) > 0 )

def test_several_prompts_default_param():
    """ Testing prompts A, B and C with default parameters
    prompt
    param_dict = {}     
    """
    report_idx_list = [20]
    clear_test_bench_folder()
    tb.eval_gs_param(report_data=report_data,
                     report_idx_list = report_idx_list, 
                     report_generator = rg,
                     prompt_method_list=cf.TEST_BENCH.PROMPT_METHODS,
                     param_dict={} )
    
    # Check the file creation, check the string prefix in the filename
    for filenames in os.listdir(folder_path):
        assert filename_prefix in filenames


def test_several_prompts_several_params():
    report_idx_list = [20]
    clear_test_bench_folder()
    df_prompts = tb.eval_gs_param(report_data=report_data,
                                report_idx_list = report_idx_list, 
                                report_generator = rg,
                                prompt_method_list=["C"],
                                param_dict={"temperature": [0.7, 1.3],
                                            "top_p": [0.6, 1]} )
    
    # Check the file creation, check the string prefix in the filename
    for filenames in os.listdir(folder_path):
        assert filename_prefix in filenames
