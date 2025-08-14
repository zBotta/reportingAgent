"""
testBench.py
"""

import pandas as pd
from conf.projectConfig import Config as cf
import logging
from app.conf.logManager import Logger
from mods.metricsEvaluator import MetricsEvaluator
from mods.dataHandler import DataHandler, Report
from mods.promptGenerator import PromptGenerator
from mods.reportGenerator import ReportGenerator
from mods.modelLoader import ModelLoader
from itertools import product

from concurrent.futures import ThreadPoolExecutor, as_completed


logging.setLoggerClass(Logger)
log = logging.getLogger(__name__)


class TestBench:
  """ A test bench for making tests on a single model. 
      NB: If testing several models, a TestBench class should be created for testing each model """
  __test__ = False # Specify to pytest to not collect test from this class
  def __init__(self, 
               MetricsEvaluator : MetricsEvaluator,
               DataHandler: DataHandler,
               ModelLoader: ModelLoader):
    self.m_eval = MetricsEvaluator
    self.dh = DataHandler
    self.ml = ModelLoader
    self.clear_df_results()
    log.info("Test Bench loaded")
  
  def clear_df_results(self):
    self.df_res : pd.DataFrame.dtypes = pd.DataFrame({})

  # def set_model_loader(self, ModelLoader: ModelLoader):
  #   self.ml = ModelLoader

  # def eval_diff_prompts(self, 
  #                       report_data : pd.DataFrame.dtypes, 
  #                       report_idx_list : list, 
  #                       report_generator: ReportGenerator):
  #   scores = {}
  #   gen_param = self.ml.get_default_tunable_parameters()
  #   for report_idx in report_idx_list:
  #     row = report_data.loc[report_idx, 'what':'contingency_actions']
  #     prompt_gen = PromptGenerator(**row.to_dict())
  #     for prompt_method in cf.TEST_BENCH.PROMPT_METHODS:
  #       prompt = prompt_gen.create_prompt(prompt_method)
  #       # The model in the report generator has a structured output with outlines library
  #       output = report_generator.generate_report(prompt)
  #       log.debug(f"\nThe model output is: \n{output}")
  #       # obtain title and report from the structured output
  #       title, report = self.dh.get_title_and_report(model_output = output) 
  #       ref_report = report_data.event_description[report_idx]
  #       t_models = cf.TEST_BENCH.T_MODELS
  #       self.m_eval.proc_scores(ref_text = ref_report, pred_text_list = [report], t_models = t_models, is_test_bench = True)
  #       # update row of the DataFrame
  #       scores.update({'report_idx': report_idx, 'prompt_method': prompt_method})
  #       scores.update(self.m_eval.get_scores())
  #       scores.update(gen_param)
  #       scores.update({"title": title, "report": report})
  #       self.df_res = pd.concat([self.df_res, pd.DataFrame.from_dict(scores)], axis=0) 

  #   return self.df_res

  def _get_param_combinations(self, 
                             param_dict: dict) -> list:
    """ Take the param_dict with all the parameters and keys and calculate the combination of each parameter.
        The method uses product (cartesian product) to obtain the combination of each parameter.
        e.g. If we have give param_dict = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [9]}

              The result is a list of dictionaries with the combined tuning parameters:
              [ {'a': 1, 'b': 4, 'c': 9},
                {'a': 1, 'b': 5, 'c': 9},
                {'a': 1, 'b': 6, 'c': 9},
                {'a': 2, 'b': 4, 'c': 9},
                {'a': 2, 'b': 5, 'c': 9},
                {'a': 2, 'b': 6, 'c': 9},
                {'a': 3, 'b': 4, 'c': 9},
                {'a': 3, 'b': 5, 'c': 9},
                {'a': 3, 'b': 6, 'c': 9}]

        @param_dict: a dictionary with parameter names as keys and a list of param values
        return: A list of dictionary with each combination
    """
    # Parse all default parameters and include them in the user input param_dict
    default_param = self.ml.get_default_tunable_parameters()
    if len(default_param.keys()) > len(param_dict.keys()):
      not_in_param_dict_k = set(default_param) - set(param_dict) # returns a set with missing keys
    # update dictionary. Default parameters have always one value define in it.
    param_dict.update({k: [default_param[k]] for k in not_in_param_dict_k}) # remember that format is a dict={key: list}
    # Generate combinations
    res_list = list(product(*param_dict.values()))
    key_list = [k for k in param_dict.keys()]
    combi_list = []
    tmp = {}

    for param_list in res_list:
      tmp = {k: v for k, v in zip(key_list, param_list)}
      combi_list.append(tmp)

    return combi_list

  def __check_param(self, param_dict: dict):
    """ Checks if the testing parameters are part of the model tunable parameters.
        Checks if param_dict has the form 
        {"param_name": [val1, val2, val3, ...]}

     Raises an error if the if the given parameters are not part of the tunable parameters 
     or if param_dict does not have a good type."""
    
    # TODO: Discuss if this should be done to check parameters. 
    #       For instance, max_new_tokens is not in the tunable_parameters but it is actually a parameter. 
    #       This throws an error when it should not do it.
    # tunable_params = self.ml.get_default_tunable_parameters()
    # if not set(param_dict) <= set(tunable_params): # is the set of given param included in set of tunable param
    #   not_tunable_params = set(param_dict) - set(tunable_params)
    #   raise ValueError(f"Warning: The given parameters: {not_tunable_params} are not tunable parameters") 
    # else:
    
    # check all values in param_dict are lists
    v_is_list = True
    for k, v in param_dict.items():
      v_is_list = True * v_is_list if type(v) is list else False
    # check param_dict is a dict
    is_dict = type(param_dict) is dict
    if not is_dict:
      raise TypeError(f"Argument param_dict must be a dictionary, e.g. : ('param_name': [val1, val2, val3, ...])")
    if not v_is_list:
      raise TypeError(f"Argument param_dict must have a list as values:  e.g. : ('param_name': [val1, val2, val3, ...])")
      
  def eval_gs_param(self, 
                    report_data : pd.DataFrame.dtypes, 
                    report_generator: ReportGenerator,
                    prompt_method_list: list = cf.TEST_BENCH.PROMPT_METHODS,
                    param_dict: dict = {},
                    xlsx_file_name= cf.TEST_BENCH.TB_FILENAME_PREFIX,
                    app_folder_destination: str = cf.TEST_BENCH.TB_RESULTS_F ):
    """
    This method tests a list of parameters for the given dataframe (report_data).
    Uses a grid search principle, running a report generation for each parameter in param_dict.

      param_dict: dictionary containing the parameter range to test, e.g. {"temperature": [0.1, 0.3, 0.6]}
                  If empty, the method generates once for each prompt method with the default parameters.
    """
    self.clear_df_results()

    if len(param_dict) > 0:
      self.__check_param(param_dict=param_dict)

    for report_idx, row in report_data.iterrows():
      for prompt_method in prompt_method_list:
        self.param_grid_search_on_row(row=row,
                              report_generator=report_generator,
                              prompt_method=prompt_method,
                              param_dict=param_dict) # If param_dict is
    # Export experiment to Excel
    treat_model_id = self.dh.treat_model_name_for_filename(self.ml.model_id)
    self.dh.export_df_to_excel(df=self.df_res,
                              xlsx_file_name= xlsx_file_name + "-" + treat_model_id,
                              app_folder_destination=app_folder_destination)
    self.clear_df_results()

  def param_grid_search_on_row(self, 
                               row: pd.Series.dtypes,
                               report_generator: ReportGenerator, 
                               prompt_method: str = cf.MODEL.DEFAULT_PROMPT_METHOD, 
                               param_dict: dict = {}):
    """
    The method is executed once for each row in the database.
    A prompt_method is given in order to start the grid search.
    param_dict: Generates a grid search for a given dict with the parameters range to test:
                e.g. {"temperature": [0.1, 0.3, 0.6], top_p: [0.5, 0.6, 0.7]}
                If empty, the method generates once with the default parameters.
                In this example, the grid search will test: temperature = 0.1 with all the values of top_p.
                                                            temperature = 0.3 with all the values of top_p.
                                                            temperature = 0.6 with all the values of top_p.
    """
    # Generate prompt on ith row
    five_ws = row.loc['what':'contingency_actions']
    prompt_gen = PromptGenerator(**five_ws.to_dict())

    res = {} # results dict saved as a row
    gen_prompt = prompt_gen.create_prompt(prompt_method)
    
    if len(param_dict) > 0:# A set of parameters is given -> Generate with grid search
      param_combi_list = self._get_param_combinations(param_dict)
      for new_gen_param in param_combi_list:
        gen_param.update(new_gen_param)
        self.generate_one_param_set(res=res,
                                    gen_prompt = gen_prompt,
                                    gen_param = gen_param,
                                    row = row,
                                    report_generator = report_generator,
                                    prompt_method = prompt_method)
        self.df_res = pd.concat([self.df_res, pd.DataFrame.from_dict(res)], axis=0)
    else:  # No parameters given -> Generate with default parameters
      self.generate_one_param_set(res=res,
                                  gen_prompt = gen_prompt,
                                  gen_param = gen_param,
                                  row = row,
                                  report_generator = report_generator,
                                  prompt_method = prompt_method)
      self.df_res = pd.concat([self.df_res, pd.DataFrame.from_dict(res)], axis=0)
        
  def generate_one_param_set(self,
                             res: dict, 
                             gen_prompt: str, 
                             gen_param: dict,
                             row: pd.Series.dtypes,
                             report_generator: ReportGenerator, 
                             prompt_method: str
                             ) -> dict:
    """ Generates a report with one set of parameters (gen_param)
    """
    # reference report is in event_description column  
    ref_report = row.event_description 

    # generate and obtain title and report from the structured output
    # if empty object, catch the error and give a default value
    log.info(f"Ref_row:{row.name}: Generating text with the following parameters:\n{gen_param}")
    print(f"Ref_row:{row.name}: Generating text with the following parameters:\n{gen_param}")
    try:
      output, gen_param = report_generator.generate_report(prompt=gen_prompt, **gen_param)
      log.debug(f"\nThe output of the model {self.ml.model_id} is: \n{output}")
      title, report = self.dh.get_title_and_report(model_output = output)    
    except Exception as e:
      log.error(f'Failed to generate report with \ngeneration_parameters {gen_param}: {e}')
      title = 'FAIL'
      report = f'FAILED_REPORT: {e}'

    self.m_eval.proc_scores(ref_text = ref_report, pred_text_list = [report], is_test_bench = True)
    
    # update row of the DataFrame and add it up
    res.update({'report_idx': row.name, 'prompt_method': prompt_method})
    res.update(gen_param)
    res.update(self.m_eval.get_scores())
    res.update({"title": title, "report": report})

    return res

  def eval_gs_param_threaded(self, 
                            report_data : pd.DataFrame.dtypes, 
                            report_generator: ReportGenerator,
                            prompt_method_list: list = cf.TEST_BENCH.PROMPT_METHODS,
                            param_dict: dict = {},
                            xlsx_file_name= cf.TEST_BENCH.TB_FILENAME_PREFIX,
                            app_folder_destination: str = cf.TEST_BENCH.TB_RESULTS_F,
                            max_workers=4) -> pd.DataFrame.dtypes:
      """
      Run generate_report_from_row on all rows of a DataFrame using multithreading.
      Preserves the original row order in the returned DataFrame.

      Parameters
      ----------
      report_data : pandas.DataFrame
          Input data.
      report_generator: ReportGenerator.
          The report Generator object containing the model, tokenizer and output_type (structured outputs)
      prompt_method : str
          Prompt generation method.
      param_dict : dict
          Arguments for generating the reports in dict, where values are in a list.
      max_workers : int, optional
          Number of worker threads. Defaults to 4.

      Returns
      -------
      pandas.DataFrame
          DataFrame of generated reports and metadata in the same order as `df`.
      """

      param_combi_list = self._get_param_combinations(param_dict)
      # nbr_rows in df * param combinations
      results_len = len(report_data) * len(param_combi_list) * len(prompt_method_list) # Preallocate list for ordered results
      log.info(f"Results file is expected to have {results_len} rows.") # change to debug

      # Create a new dict (gen_param) to update it with new grid search param
      

      with ThreadPoolExecutor(max_workers=max_workers) as executor:
          res = {}
          futures = {
              executor.submit(self.generate_one_param_set,
                              res,
                              PromptGenerator(**dict(row.loc['what':'contingency_actions'])).create_prompt(prompt_method),
                              gen_param, # gen_param.update(new_gen_param)
                              row,
                              report_generator,
                              prompt_method
              ) : idx
              for idx, row in report_data.iterrows() for gen_param in param_combi_list for prompt_method in prompt_method_list
          }

          for future in as_completed(futures):
              idx = futures[future]
              try:
                  result = future.result()
                  log.info(f"future result = {result}")
                  self.df_res = pd.concat([self.df_res, pd.DataFrame.from_dict(result)], axis=0)
              except Exception as e:
                  log.error(f"FAILED report export: {e} on row={idx}")
                  
      
      # Export experiment to Excel
      # self.df_res = pd.DataFrame(results)
      self.dh.export_df_to_excel(df=self.df_res,
                                xlsx_file_name= xlsx_file_name,
                                app_folder_destination=app_folder_destination)
      df = self.df_res
      self.clear_df_results()

      return df

