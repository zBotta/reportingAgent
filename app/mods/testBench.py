"""
testBench.py
"""

import pandas as pd
from conf.projectConfig import Config as cf
from mods.metricsEvaluator import MetricsEvaluator
from mods.dataHandler import DataHandler
from mods.promptGenerator import PromptGenerator
from mods.reportGenerator import ReportGenerator


class TestBench:

  def __init__(self, MetricsEvaluator : MetricsEvaluator, DataHandler: DataHandler):
    self.prompt_methods = cf.TEST_BENCH.PROMPT_METHODS
    self.m_eval = MetricsEvaluator
    self.dh = DataHandler
    self.df_diff_prompt_res : pd.DataFrame.dtypes = pd.DataFrame({})
    print("\n Test Bench loaded")
  
  def eval_diff_prompts(self, report_data : pd.DataFrame.dtypes, report_idx_list : list, report_generator: ReportGenerator):
    scores = {}
    for report_idx in report_idx_list:
      row = report_data.loc[report_idx, 'what':'contingency_actions']
      prompt_gen = PromptGenerator(**row.to_dict())
      for prompt_method in self.prompt_methods:
        prompt = prompt_gen.create_prompt(prompt_method)
        # The model in the report generator has a structured output with outlines library
        output = report_generator.generate_report(prompt)
        print(f"\nThe model output is: \n{output}")
        # obtain title and report from the structured output
        title, report = self.dh.get_title_and_report(model_output = output) 
        ref_report = report_data.event_description[report_idx]
        t_models = cf.TEST_BENCH.T_MODELS
        self.m_eval.proc_scores(ref_text = ref_report, pred_text_list = [report], t_models = t_models, is_test_bench = True)
        # update row of the DataFrame
        scores.update({'report_idx': report_idx, 'prompt_method': prompt_method})
        scores.update(self.m_eval.get_scores())
        scores.update({"title": title, "report": report})
        self.df_diff_prompt_res = pd.concat([self.df_diff_prompt_res, pd.DataFrame.from_dict(scores)], axis=0) 

    return self.df_diff_prompt_res