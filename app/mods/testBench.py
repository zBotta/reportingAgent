"""
testBench.py
"""

import pandas as pd
import numpy as np
from conf.projectConfig import Config as cf
import logging
from app.conf.logManager import Logger
from mods.metricsEvaluator import MetricsEvaluator
from mods.dataHandler import DataHandler, Report
from mods.promptGenerator import PromptGenerator
from mods.reportGenerator import ReportGenerator
from mods.modelLoader import ModelLoader
from itertools import product
from datetime import datetime as dt
import re

from concurrent.futures import ThreadPoolExecutor, as_completed


logging.setLoggerClass(Logger)
log = logging.getLogger(__name__)


class TestBench:
  """ A test bench for making parameter grid search on a single model. 
      NB: If testing several models, a TestBench class should be created for testing each model 
      This class handles as well the statistical analysis of the TestBench outputs
  """
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
    self.experiment_id: str = "" # id containing model charged and
  
  def clear_df_results(self):
    self.df_res : pd.DataFrame.dtypes = pd.DataFrame({})

  def set_experiment_id(self):
    """ a unique id with timestamp to identify the current experiment in the Test Bench
        We are going to use the experiment id as a unique for the export files.
        """
    # Add time of creation to filename
    treat_model_id = self.dh.treat_model_name_for_filename(self.ml.model_id)
    dt_creation = dt.now().strftime("%d-%m%Y_%H-%M-%S")
    exp_id = cf.TEST_BENCH.TB_FILENAME_PREFIX + "-" + treat_model_id + "-" + dt_creation
    self.experiment_id = exp_id
    log.info(f"Starting experiment in TestBench with experiment_id={self.experiment_id}")

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
    if set(param_dict) != set(default_param):
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
    self.set_experiment_id()

    if len(param_dict) > 0:
      self.__check_param(param_dict=param_dict)

    for report_idx, row in report_data.iterrows():
      for prompt_method in prompt_method_list:
        self.param_grid_search_on_row(row=row,
                              report_generator=report_generator,
                              prompt_method=prompt_method,
                              param_dict=param_dict) # If param_dict is
    # Export experiment to Excel
    self.dh.export_df_to_excel(df=self.df_res,
                               xlsx_file_name= xlsx_file_name,
                                app_folder_destination=app_folder_destination)
    
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
      for gen_param in param_combi_list:
        self.generate_one_param_set(res=res,
                                    gen_prompt = gen_prompt,
                                    gen_param = gen_param,
                                    row = row,
                                    report_generator = report_generator,
                                    prompt_method = prompt_method)
        df_row = pd.DataFrame.from_dict(res)
        self.dh.export_df_row_to_tmp_csv(df_row, 
                                         xlsx_file_name=self.experiment_id,
                                         app_folder_destination=cf.ANALYSIS.AN_RESULTS_F) # pass the experiment id as filename
        
        self.df_res = pd.concat([self.df_res, df_row], axis=0)
    else:  # No parameters given -> Generate with default parameters
      gen_param = self.ml.get_default_tunable_parameters()
      self.generate_one_param_set(res=res,
                                  gen_prompt = gen_prompt,
                                  gen_param = gen_param,
                                  row = row,
                                  report_generator = report_generator,
                                  prompt_method = prompt_method)
      df_row = pd.DataFrame.from_dict(res)
      self.df_res = pd.concat([self.df_res, df_row], axis=0)
        
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
    log.info(f"Ref_row:{row.name} & prompt_method={prompt_method}: Generating text with the following parameters:\n{gen_param} ")
    print(f"Ref_row:{row.name} & prompt_method={prompt_method}: Generating text with the following parameters:\n{gen_param}")
    try:
      output, gen_param = report_generator.generate_report(prompt=gen_prompt, **gen_param)
      log.debug(f"\nThe output of the model {self.ml.model_id} is: \n{output}")
      title, report = self.dh.get_title_and_report(model_output = output)    
    except Exception as e:
      log.error(f'Failed to generate report with \ngeneration_parameters {gen_param}: {e}')

    self.m_eval.proc_scores(ref_text = ref_report, pred_text_list = [report], is_test_bench = True)
    
    # update row of the DataFrame and add it up
    res.update({'report_idx': row.name, 'prompt_method': prompt_method})
    res.update(gen_param)
    res.update(self.m_eval.get_scores())
    res.update({"title": title, "report": report})
    log.info(f'results: {res}')
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
      self.clear_df_results()
      self.set_experiment_id()
      param_combi_list = self._get_param_combinations(param_dict)

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
              for idx, row in report_data.iterrows() for prompt_method in prompt_method_list for gen_param in param_combi_list
          }

          for future in as_completed(futures):
              idx = futures[future]
              try:
                  result = future.result()
                  log.info(f"future result = {result}")
                  self.df_res = pd.concat([self.df_res, pd.DataFrame.from_dict(result)], axis=0)
              except Exception as e:
                  log.error(f"FAILED report export: {e} on row={idx}")
                  
      # Export experiment row to Excel
      self.dh.export_df_to_excel(df=self.df_res,
                                 xlsx_file_name= self.experiment_id,
                                 app_folder_destination=app_folder_destination)

      return self.df_res

  def print_number_of_combinations(self,
                                 report_data: pd.DataFrame.dtypes,
                                 param_dict: dict,
                                 prompt_method_list: list = cf.TEST_BENCH.PROMPT_METHODS
                                 ) -> int:
    
    param_combi_list = self._get_param_combinations(param_dict)
    # nbr_rows in df * param combinations * promp_method_list
    n_combinations = len(report_data) * len(param_combi_list) * len(prompt_method_list)
    log.info(f"Results file is expected to have {n_combinations} rows.") # change to debug
    print(f"Results file is expected to have {n_combinations} rows.") # change to debug
  
    return n_combinations

  ## ANALYSIS PART OF TEST BENCH RESULTS

  def get_param_and_scores_cols(self, tb_df):
    """ Obtains the parameters and scores columns from the results dataframe
        tb_df: The dataframe output of the test bench experiment"""
    col_nb_pm = tb_df.columns.get_loc("prompt_method")
    col_nb_bs = tb_df.columns.get_loc(cf.METRICS.BS_PRECISION_KEY)   # The first column of the parameters is bs_precision
    col_nb_ce = tb_df.columns.get_loc("title")
    param_col_names = tb_df.columns[col_nb_pm+1:col_nb_bs].to_numpy()  # list with the keys of the parameters
    scores_col_names = tb_df.columns[col_nb_bs:col_nb_ce].to_numpy()  # list with the keys of the scores
    return param_col_names, scores_col_names

# MEAN ANALYSIS
  def get_top_params_by_mean(self, df,
                            prompt_method:str,
                            top_score: str):
    df_ord_X = df[df.prompt_method == prompt_method]
    param_col_names, scores_col_names = self.get_param_and_scores_cols(df_ord_X)
    df_scores = df_ord_X[np.append(param_col_names, scores_col_names)]
    df_param_scores_mean = df_scores.groupby(param_col_names.tolist(), group_keys=True, as_index=False).mean()
    df_param_scores_mean.columns.name = "[" + top_score + "] ordered by best [mean]"
    return df_param_scores_mean

  def export_mean_scores_on_experiment_id(self, exp_id,
                                          prompt_method_list,
                                          top_score: str,
                                          df: pd.DataFrame.dtypes = None,
                                          app_folder_dest = cf.ANALYSIS.AN_RESULTS_F): 
    """ Export mean statistics from experiment id in Test Bench
        df: TestBench results dataframe to apply the mean on scores. If df is already in memory, we can avoid passing it as argument
    """

    # df is not in memory, we read the file with its unique id
    if df is None:
      self.dh.get_df_from_tb_exp_id_results(exp_id)

    for prompt_method in prompt_method_list:
      _df_mean = self.get_top_params_by_mean(df, prompt_method, top_score)
      self.dh.export_df_to_excel_by_sheet_name(_df_mean, 
                                               "an-mean-" + exp_id, 
                                               sheet_name="pm_" + prompt_method,
                                               app_folder_destination=app_folder_dest)

# STATISTICAL ANALYSIS

  def get_top_params_by_stat(self, df,
                            prompt_method:str,
                            stat: str, # "min", "25%", "50%", "75%", "max", "std"
                            top_score:str,
                            top_k_param: int = cf.ANALYSIS.TOP_K_PARAM,
                            ):
    """
    prompt_method : A B or C
    stat: the statistic used for the sorting strategy,
          e.g if stat="mean" is given, we are going to sort the best mean values.
    top_k_param: Return the best top K values. e.g if top_k_param = 1 will return the best parameter for the give stat.
    top_score: The metric that we are going to target for sorting the best parameters
    """

    df_ord_X = df[df.prompt_method == prompt_method]
    param_col_names, scores_col_names = self.get_param_and_scores_cols(df_ord_X)
    df_scores = df_ord_X[np.append(param_col_names, scores_col_names)]
    df_param_scores_stats = df_scores.groupby(param_col_names.tolist()).describe()

    # filter the TOP_K stats according to the given stat and top_score
    top_params = df_param_scores_stats.sort_values(by=(top_score,stat), ascending=False)[top_score].head(top_k_param)
    top_params.columns.name = "[" + top_score + "] ordered by best [" + stat + "]"
    return top_params

  def get_df_list_by_top_scores(self, df, 
                                prompt_method:str,
                                top_score_list,
                                top_k_param: int = cf.ANALYSIS.TOP_K_PARAM,
                                stat: str = "mean"):
    df_list = []
    for top_score in top_score_list:
      df_score = self.get_top_params_by_stat(df, prompt_method, stat, top_score, top_k_param)
      df_list.append(df_score)
    return df_list

  def export_stat_analysis_on_experiment_id(self, exp_id,
                                            prompt_method_list,
                                            top_score_list,
                                            top_k_param: int = cf.ANALYSIS.TOP_K_PARAM,
                                            stat: str = "mean",
                                            df = None,
                                            app_folder_dest = cf.ANALYSIS.AN_RESULTS_F):
    """ Export mean statistics from experiment id in Test Bench
        df: TestBench results dataframe to apply the mean on scores. If df is already in memory, we can avoid passing it as argument
    """
    # df is not in memory, we read the results file with its unique id
    if df is None:
      self.dh.get_df_from_tb_exp_id_results(exp_id)
    # Create 
    for prompt_method in prompt_method_list:
      df_list = self.get_df_list_by_top_scores(df, prompt_method, top_score_list, top_k_param, stat)
      for _df in df_list:
        score, stat = re.findall(r'(?<=\[).+?(?=\])',_df.columns.name)  # Catch the score and stat from columns.name
        sheet_name = score # each sheet takes the name of the score
        self.dh.export_df_to_excel_by_sheet_name(_df, 
                                                 "an-stats-pm_" + prompt_method + "-" + exp_id, 
                                                 sheet_name=sheet_name,
                                                 app_folder_destination=app_folder_dest)
