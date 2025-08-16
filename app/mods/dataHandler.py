"""
dataHandler.py
"""

from conf.projectConfig import Config as cf
from app.conf.logManager import Logger
import os
import logging
import pandas as pd
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

logging.setLoggerClass(Logger)
log = logging.getLogger(__name__)

class Report(BaseModel):
  """
  A pydantic class containing the structured outputs of the LMs
  """
  title: str
  report: str


class DataHandler:
  """
  This class can: 
  1. Import the reports from a database
  2. Handle the responses of the model with the `outlines` library.
  3. Export to Excel files
  """
  def __init__(self):
    pass
  
  def import_reports(self, xlsx_file_name = cf.DATA.DH_DEFAULT_DATASET_FILENAME):
      data_path = os.path.join(cf.APP_PATH, "datasets", xlsx_file_name)
      file_exists = self.check_file_exists(data_path)
      if file_exists:
        df_reports = pd.read_excel(data_path)
        df_reports.columns = cf.DATA.DF_COLUMNS
        log.info(f"Dataset loaded from path : {data_path}")
        return df_reports
      else:
        raise FileExistsError(f"File does not exist in {data_path}")

  def check_file_exists(self, file_path: str):
    """ Checks files destination. If does not exist, throws an error """
    if not os.path.exists(file_path):
      return False
    else:
      return True

  def check_folder_exists(self, folder_path: str):
    """ Checks folder destination and creates it if does not exist """
    if not os.path.isdir(folder_path):
      os.makedirs(folder_path)
      log.warning(f"Folder does not exist, creating new folder in: {folder_path}")
      return False
    
    return True

  def export_df_row_to_tmp_file(self,
                                df_row: pd.DataFrame.dtypes,
                                xlsx_file_name: str, 
                                app_folder_destination: str = cf.DATA.DH_DEFAULT_RESULTS_F):
    xlsx_file_name = "tmp-" + xlsx_file_name + ".csv"
    file_path = self._get_filename_path(xlsx_file_name, app_folder_destination)
    self.last_tmp_file_path = file_path
    if self.check_file_exists(file_path):
      is_header = False
      mode = "a" 
    else:
      is_header = True
      mode = "w"
    df_row.to_csv(file_path, index=False, header=is_header, sep=",", mode=mode) 

  def clear_tmp_file(self):
    """ Clear the last tmp file """
    if os.path.isdir(self.last_tmp_file_path): 
        os.remove(self.last_tmp_file_path) 

  def export_df_to_excel(self, 
                         df: pd.DataFrame.dtypes,
                         xlsx_file_name: str, 
                         app_folder_destination: str = cf.DATA.DH_DEFAULT_RESULTS_F):
    
    xlsx_file_name = xlsx_file_name + ".xlsx"
    excel_path = self._get_filename_path(xlsx_file_name, app_folder_destination)
    df.to_excel(excel_path, index=False)
    log.info(f"Saving df to excel in: {excel_path}")

  def export_df_to_excel_by_sheet_name(self, df: list,
                                       xlsx_file_name: str, 
                                       sheet_name: str,
                                       app_folder_destination: str = cf.DATA.DH_DEFAULT_RESULTS_F):
    
    xlsx_file_name = xlsx_file_name + ".xlsx"
    excel_path = self._get_filename_path(xlsx_file_name, app_folder_destination)
    
    if not self.check_file_exists(excel_path):
      mode = "w"
      print(f"Saving df to excel in: {excel_path}")
    else:
      mode = "a"
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode=mode) as writer: #, engine='xlsxwriter'
      df.to_excel(writer, sheet_name=sheet_name, index=True)

  def _get_filename_path(self, xlsx_file_name, app_folder_destination):
    """ Checks file path exists and adds time stamp to filename"""
    # # Add time of creation to filename
    # dt_creation = dt.now().strftime("%d-%m%Y %H-%M-%S")
    # _xlsx_file_name = xlsx_file_name + "-" + dt_creation + ".xlsx"
    # Check folder destination and create it if does not exist
    folder_path = os.path.join(cf.APP_PATH, app_folder_destination).__str__()
    self.check_folder_exists(folder_path)
    excel_path = os.path.join(cf.APP_PATH, app_folder_destination, xlsx_file_name).__str__()
    return excel_path

  def get_title_and_report(self, model_output: str, output_structure = Report) -> tuple:
    """
    Takes the model output and returns the Title and the Report text in a structured output.
    Remember that the output of the model has been conditioned to have a given output structure 
    of the form of a pydantic class called "Report" thanks to the ´outlines´ library.
    output_structure = the pydantic class Report
    model_output = the response of the model to the prompt (output structured by outlines)

    Output: A tuple with the title and the report texts
    """
    try:
      title = output_structure.model_validate_json(model_output).title.strip()
      report = output_structure.model_validate_json(model_output).report.strip()
    except Exception as e:
      log.error(f"Error while unpacking title or report from model output. Error: {e}")
      title = "NO PYDANTIC TITLE"
      report = "NO PYDANTIC REPORT"
    finally:
      return title, report
    
  def export_to_excel_from_api_response(self, 
                                        report_data :pd.DataFrame.dtypes, 
                                        model_name :str, 
                                        filename :str,
                                        app_folder_destination: str = cf.API.API_GEN_REPORTS_F):
    """
    Takes the model output (response) and converts it into a dataframe, then it saves it in datasets
    """
    # FastAPI exposes jsonable_encoder which essentially performs that same transformation on an arbitrarily nested structure of BaseModel:
    df = pd.DataFrame(jsonable_encoder(report_data)) 

    model_name = self.treat_model_name_for_filename(model_name)    
    folder_path = os.path.join(cf.APP_PATH, app_folder_destination).__str__()
    self.check_folder_exists(folder_path)
    xlsx_file_name = filename + "-" + model_name
    self.export_df_to_excel(df=df,
                            xlsx_file_name=xlsx_file_name,
                            app_folder_destination=app_folder_destination)

  def treat_model_name_for_filename( self, model_name: str):
    """ separate from model name the / and : values. 
      For instance from community/gpt2:xl 
        we will obtain community-gpt2_xl .  """
    if model_name.__contains__("/"):
      model_name = model_name.replace("/", "-")
    if model_name.__contains__(":"):
      model_name = model_name.replace(":", "_")
  
    return model_name

  def get_df_from_tb_exp_id_results(self, exp_id):
    """ Imports a unique test bench experiment id excel to a dataframe"""
    excel_path = os.path.join("app", cf.TEST_BENCH.TB_RESULTS_F, exp_id + ".xlsx")
    df = pd.read_excel(excel_path)
    return df