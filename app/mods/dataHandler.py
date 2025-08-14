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
from datetime import datetime as dt

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

  def check_file_exists(self, file_path: str):
    """ Checks files destination. If does not exist, throws an error """
    if not os.path.exists(file_path):
      raise FileExistsError(f"File does not exist in {file_path}")
    else:
      return True

  def check_folder_exists(self, folder_path: str):
    """ Checks folder destination and creates it if does not exist """
    if not os.path.isdir(folder_path):
      os.makedirs(folder_path)
      log.warning(f"Folder does not exist, creating new folder in: {folder_path}")

  def export_df_to_excel(self, 
                         df: pd.DataFrame.dtypes,
                         xlsx_file_name: str, 
                         app_folder_destination: str = cf.DATA.DH_DEFAULT_RESULTS_F):
    # Add time of creation to filename
    dt_creation = dt.now().strftime("%d-%m%Y %H-%M-%S")
    _xlsx_file_name = xlsx_file_name + "-" + dt_creation + ".xlsx"
    # Check folder destination and create it if does not exist
    folder_path = os.path.join(cf.APP_PATH, app_folder_destination).__str__()
    self.check_folder_exists(folder_path)
    excel_path = os.path.join(cf.APP_PATH, app_folder_destination, _xlsx_file_name).__str__()
    log.info(f"Saving df to excel in: {excel_path}")
    df.to_excel(excel_path, index=False)

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
