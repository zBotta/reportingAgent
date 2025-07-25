"""
dataHandler.py
"""

from conf.projectConfig import Config as cf
import os
import pandas as pd
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from datetime import datetime as dt


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
  
  def import_reports(self, xlsx_file_name = "Reports_dataset.xlsx"):
      data_path = os.path.join(cf.PROJECT_PATH, "datasets", xlsx_file_name)
      df_reports = pd.read_excel(data_path)
      df_reports.columns = ['type', 'what', 'when', 'where', 'who', 'how', 'why', 'contingency_actions', 'event_description', 'NbChr']
      print(f"\nDataset loaded from path : {data_path}")
      return df_reports

  def get_title_and_report(self, model_output: str, output_structure = Report) -> tuple:
    """
    Takes the model output and returns the Title and the Report text in a structured output.
    Remember that the output of the model has been conditioned to have a given output structure 
    of the form of a pydantic class called "Report" thanks to the ´outlines´ library.
    output_structure = the pydantic class Report
    model_output = the response of the model to the prompt (output structured by outlines)

    Output: A tuple with the title and the report texts
    """
    title = output_structure.model_validate_json(model_output).title.strip()
    report = output_structure.model_validate_json(model_output).report.strip()
    return title, report


  def export_to_excel_from_response(self, report_data :pd.DataFrame.dtypes, model_name :str, filename :str):
    """
    Takes the model output (response) and converts it into a dataframe, then it saves it in datasets/tests
    """
    # FastAPI exposes jsonable_encoder which essentially performs that same transformation on an arbitrarily nested structure of BaseModel:
    df = pd.DataFrame(jsonable_encoder(report_data)) 

    if model_name.__contains__("/"):
      model_name = model_name.split("/")[1]
      if model_name.__contains__(":"):
        model_name = model_name.split(":")[0]

    # Add time of creation to filename
    dt_creation = dt.now().strftime("%d-%m%Y %H-%M-%S")
    xlsx_file_name = filename + "-" + model_name + "-" + dt_creation + ".xlsx"
    excel_path = os.path.join(cf.PROJECT_PATH, "datasets", "tests", xlsx_file_name).__str__()
    print(f"Saving excel with reports to: {excel_path}")
    df.to_excel(excel_path, index=False)


# if __name__ == "__main__":
#   from dataHandler import DataHandler
  
#   dh = DataHandler()
  
#   df = dh.import_reports()
#   print(df.head())