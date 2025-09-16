
import pandas as pd
import os
import shutil
from projectSetup import Setup
from conf.projectConfig import Config as cf
from mods.dataHandler import DataHandler, Report
from mods.apiReportGenerator import ApiReport, ApiReports

from mods.testBench import TestBench
from mods.metricsEvaluator import MetricsEvaluator
from mods.promptGenerator import PromptGenerator
from mods.reportGenerator import ReportGenerator
from mods.modelLoader import ModelLoader

def test_import_excel():
    dh = DataHandler()
    df = dh.import_reports()
    assert df.columns.to_list() == cf.DATA.DF_COLUMNS

def test_export_excel():
    dh = DataHandler()
    # Create a dataframe
    d1 = {"a": 1, "b": 2, "c": 44, "d": 551}
    d2 = {"a": 4, "b": 31, "c": 66, "d": 666}
    df = pd.DataFrame([d1,d2])
    app_folder_dest = cf.TESTS.T_EXP_DEST_F
    filename_prefix = cf.TESTS.T_EXP_PREFIX
    folder_path = os.path.join(cf.APP_PATH, app_folder_dest)
    if os.path.isdir(folder_path): # delete folder with previous tests files
        shutil.rmtree(folder_path) 
    # Export it to excel
    dh.export_df_to_excel(df=df,
                          xlsx_file_name=filename_prefix, 
                          app_folder_destination=app_folder_dest)
    
    # Check the file creation, check the string prefix in the filename
    for filenames in os.listdir(folder_path):
        assert filename_prefix in filenames
    
def test_api_export():
    dh = DataHandler()
    # Create a single report with the structured outputs
    api_rep = ApiReport(
        report_name='Evening Collision Near Lyon on N346 Route', 
        what='a car accident', 
        when='Wednesday at 7:30 pm', 
        where='near Lyon on the N346 route', 
        who='two drivers returning home after work', 
        how='Vehicle A, a motorcycle, was riding in the left lane speeding to overtake traffic while Vehicle B, a delivery van ahead, slowed abruptly due to road construction ahead', 
        why='Vehicle A’s rider failed to anticipate the sudden deceleration of Vehicle B, causing a rear-end collision', 
        contingency_actions='Traffic officers quickly closed the affected lane to assist the injured motorcyclist and detour traffic; emergency medical teams arrived within minutes; authorities collected CCTV footage from nearby cameras and eyewitness accounts to determine fault', 
        report='On Wednesday at 7:30 pm, during rush hour near Lyon on the N346 route, a traffic accident occurred involving a motorcycle (Vehicle A) and a delivery van (Vehicle B). As the van slowed unexpectedly because of roadworks, the motorcyclist behind failed to reduce speed, resulting in a rear-end collision. The motorcyclist sustained minor injuries and was treated on-site by emergency responders. The incident caused temporary lane closures and substantial traffic delays. Investigators gathered CCTV footage and eyewitness statements to clarify the circumstances. This event highlighted the dangers of inattentiveness under changing road conditions, emphasizing the need for caution during construction zones.'
        )
    report_data = ApiReports(reports=[api_rep])

    app_folder_dest = cf.TESTS.T_API_DEST_F
    filename_prefix =  cf.TESTS.T_API_PREFIX
    folder_path = os.path.join(cf.APP_PATH, app_folder_dest)
    if os.path.isdir(folder_path): # delete folder with previous tests files
        shutil.rmtree(folder_path) 
    dh.export_to_excel_from_api_response(report_data=report_data, 
                                         model_name="gpt2", 
                                         filename=filename_prefix,
                                         sheet_name="api-test",
                                         app_folder_destination=app_folder_dest,
                                         )
    # Check the file creation, check the string prefix in the filename
    for filenames in os.listdir(folder_path):
        assert filename_prefix in filenames

def test_export_mean_scores():
    env = Setup()
    met_eval = MetricsEvaluator()
    # Load data
    dh = DataHandler()

    # Load model
    model_id = 'openai-community/gpt2' 
    ml = ModelLoader(model_id=model_id, device=env.device, torch_dtype=env.torch_dtype)
    tb = TestBench(MetricsEvaluator = met_eval, DataHandler=dh, ModelLoader=ml)
    tb.set_experiment_id()

    fake_row_1 = {'report_idx': 34, 'prompt_method': 'C', 'temperature': 0.7, 'top_p': 0.3, 'top_k': 30,
                   'max_new_tokens': 300, 'do_sample': True, 'repetition_penalty': 1, 'bs_precision': 1.0, 'bs_recall': 1.0,
                     'bs_f1': 1.0, 'rouge1': 0.2, 'rouge2': 0.1, 'rougeL': 0.1, 'rougeLsum':0.05, 
                     'bleu':0.4, 'b_1_grams':0.8,'b_2_grams':0.3,'b_3_grams':0.2,'b_4_grams':0.1, 
                     'be_sim': 1.0, 'ce_sim': 1.0, 'title': 'Wrong tablet counting', 'report': 'On July 2, 2025, at 3:30 PM, Erik Hansen loaded the wrong tablet counting disk during changeover on Bottle Packaging Line 2 for Batch RX500 of Neurocet 50 mg. Sarah Yoon from QA discovered the issue during AQL sampling. The line was stopped, 500 bottles were segregated, and rework and retraining were initiated.'}
    fake_row_2 = {'report_idx': 35, 'prompt_method': 'C', 'temperature': 0.7, 'top_p': 0.3, 'top_k': 30,
                   'max_new_tokens': 300, 'do_sample': True, 'repetition_penalty': 1, 'bs_precision': 1.0, 'bs_recall': 1.0,
                   'bs_f1': 1.0, 'rouge1': 0.2, 'rouge2': 0.1, 'rougeL': 0.1, 'rougeLsum':0.05, 
                   'bleu':0.4, 'b_1_grams':0.4,'b_2_grams':0.4,'b_3_grams':0.3,'b_4_grams':0.2, 
                   'be_sim': 1.0, 'ce_sim': 1.0, 'title': 'Wrong tablet counting', 'report': 'On July 2, 2025, at 3:30 PM, Erik Hansen loaded the wrong tablet counting disk during changeover on Bottle Packaging Line 2 for Batch RX500 of Neurocet 50 mg. Sarah Yoon from QA discovered the issue during AQL sampling. The line was stopped, 500 bottles were segregated, and rework and retraining were initiated.'}
    fake_row_3 = {'report_idx': 34, 'prompt_method': 'B', 'temperature': 0.7, 'top_p': 0.3, 'top_k': 30,
                   'max_new_tokens': 300, 'do_sample': True, 'repetition_penalty': 1, 'bs_precision': 1.0, 'bs_recall': 1.0,
                     'bs_f1': 1.0, 'rouge1': 0.2, 'rouge2': 0.1, 'rougeL': 0.1, 'rougeLsum':0.05, 
                     'bleu':0.4, 'b_1_grams':0.8,'b_2_grams':0.3,'b_3_grams':0.2,'b_4_grams':0.1, 
                     'be_sim': 1.0, 'ce_sim': 1.0, 'title': 'Wrong tablet counting', 'report': 'On July 2, 2025, at 3:30 PM, Erik Hansen loaded the wrong tablet counting disk during changeover on Bottle Packaging Line 2 for Batch RX500 of Neurocet 50 mg. Sarah Yoon from QA discovered the issue during AQL sampling. The line was stopped, 500 bottles were segregated, and rework and retraining were initiated.'}
    fake_row_4 = {'report_idx': 35, 'prompt_method': 'B', 'temperature': 0.7, 'top_p': 0.3, 'top_k': 30,
                   'max_new_tokens': 300, 'do_sample': True, 'repetition_penalty': 1, 'bs_precision': 1.0, 'bs_recall': 1.0,
                   'bs_f1': 1.0, 'rouge1': 0.2, 'rouge2': 0.1, 'rougeL': 0.1, 'rougeLsum':0.05, 
                   'bleu':0.4, 'b_1_grams':0.4,'b_2_grams':0.4,'b_3_grams':0.3,'b_4_grams':0.2, 
                   'be_sim': 1.0, 'ce_sim': 1.0, 'title': 'Wrong tablet counting', 'report': 'On July 2, 2025, at 3:30 PM, Erik Hansen loaded the wrong tablet counting disk during changeover on Bottle Packaging Line 2 for Batch RX500 of Neurocet 50 mg. Sarah Yoon from QA discovered the issue during AQL sampling. The line was stopped, 500 bottles were segregated, and rework and retraining were initiated.'}
    
    fake_df = pd.DataFrame([fake_row_1, fake_row_2, fake_row_3, fake_row_4])
    n_comb = 1
    prompt_method_list = ['B', 'C']

    app_folder_dest = cf.TESTS.T_AN_RESULTS_F
    filename_prefix =  cf.TESTS.T_AN_FILENAME_PREFIX
    folder_path = os.path.join(cf.APP_PATH, app_folder_dest)
    if os.path.isdir(folder_path): # delete folder with previous tests files
        shutil.rmtree(folder_path) 

    tb.export_mean_scores_on_experiment_id(exp_id= filename_prefix + "-" + tb.experiment_id, 
                                           prompt_method_list=prompt_method_list,
                                           top_score=cf.ANALYSIS.TOP_SCORE,
                                           df = fake_df, # tb.df_res
                                           app_folder_dest=cf.TESTS.T_AN_RESULTS_F
                                           )

    tb.export_stat_analysis_on_experiment_id(exp_id=filename_prefix + "-" + tb.experiment_id,
                                             prompt_method_list=prompt_method_list,
                                             top_score_list=cf.ANALYSIS.TOP_SCORE_LIST,
                                             top_k_param=n_comb, # export ALL the parameter combinations
                                             df = fake_df, # tb.df_res
                                             app_folder_dest=cf.TESTS.T_AN_RESULTS_F
                                             )
    
    # Check the file creation, check the string prefix in the filename
    count = 0
    for filenames in os.listdir(folder_path):
        count+= 1 if filename_prefix in filenames else 0
    
    assert count == 3