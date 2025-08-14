"""
projectConfig.py
"""

import logging
import os
from pathlib import Path
root_path = str(Path(__file__).absolute().parent.parent.parent.resolve())
app_path = str(Path(__file__).absolute().parent.parent.resolve())


class Logger():
   LOG_NAME = "reportAgent"
   LOG_LEVEL = logging.INFO # log file output, options: INFO, WARNING
   CONSOLE_LEVEL = logging.WARNING # console output, options: INFO, WARNING
   FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   DATE_TIME_FORMAT = "%m/%d/%Y %H:%M:%S"
   LOG_FILE = os.path.join(app_path, "logs", "logfile.log")
   LOG_DIR = os.path.join(app_path, "logs")


class Data():
   DF_COLUMNS = ['type', 'what', 'when', 'where', 'who', 'how', 'why', 'contingency_actions', 'event_description', 'NbChr']
   DH_DEFAULT_RESULTS_F = "results"  # Default results folder name to export results
   DH_DEFAULT_DATASET_FILENAME = "Reports_dataset.xlsx" # Default Excel Dataset to import reference reports and 5W's from


class pyTests():
   T_TB_FILENAME_PREFIX = "test-tb"
   T_TB_RESULTS_F = os.path.join("test", "test-bench") # Folder to put the test-bench results
   T_API_PREFIX = "test-api-export"
   T_API_DEST_F = os.path.join("test", "api")
   T_EXP_PREFIX = "test-export"
   T_EXP_DEST_F = os.path.join("test", "test_export")


class TestBench():
   PROMPT_METHODS = ['A', 'B', 'C']
   BERT_MODEL ="distilbert-base-uncased"
   BE_MODEL = "all-MiniLM-L6-v2"
   CE_MODEL = "ms-marco-MiniLM-L6-v2"
   TB_FILENAME_PREFIX = "tb"
   TB_RESULTS_F = os.path.join("results", "test-bench") 

class Model():
   MAX_NEW_TOKENS = 300
   DO_SAMPLE = True
   DEFAULT_PROMPT_METHOD = "C"
   PARAM_LIST = ["temperature", "top_k", "top_p", "max_new_tokens", "repetition_penalty", 
                 "frequency_penalty", "presence_penalty", "stop", "do_sample"]
   VAL_IF_NOT_IN_PARAM_LIST = None


class Api():
   API_GEN_REPORTS_F = "results/api"  # The folder to export the generated API reports
   GPT_API_MODEL = "gpt-4.1-mini"
   # ChatGPT API (pay)
   # OPTIONS: 
      # "gpt-4.1-mini"
      # "gpt-4.1" # Careful it is expensive !
      # "gpt-4.1-nano" # Does not work well for car A and B instructions

   GROQ_API_MODEL = "llama-3.3-70b-versatile"  
   # GROQ API (free but limited)
   # OPTIONS:
      # llama-3.1-8b-instant (Meta)  # Not enough details
      # llama-3.3-70b-versatile (Meta)  # Works well
      # gemma2-9b-it (GOogle) # Not working well and not handling long structure
      # deepseek-r1-distill-llama-70b (DeepSeek) # Not handling long structures  
   TRAFFIC_REPORT_ID = 1
   PHARMA_REPORT_ID = 2


class Config():
    TEST_BENCH = TestBench 
    MODEL = Model
    DATA = Data
    PROJECT_PATH = root_path
    APP_PATH = app_path
    API = Api()
    LOG = Logger()
    TESTS = pyTests()

