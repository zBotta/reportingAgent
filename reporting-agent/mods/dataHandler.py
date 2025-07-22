"""
dataHandler.py
"""

from pathlib import Path
import pandas as pd
from pydantic import BaseModel


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
      project_path = Path(__file__).parent.parent.resolve()
      data_path = project_path / "datasets/" / xlsx_file_name
      df_reports = pd.read_excel(data_path)
      df_reports.columns = ['type', 'what', 'when', 'where', 'who', 'how', 'why', 'contingency_actions', 'event_description', 'NbChr']
      print(f"\nDataset loaded from path : {data_path}")
      return df_reports

  def get_title_and_report(self, model_output: str, output_structure = Report) -> tuple():
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


# if __name__ == "__main__":
#   from dataHandler import DataHandler
  
#   dh = DataHandler()
  
#   df = dh.import_reports()
#   print(df.head())