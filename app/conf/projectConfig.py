"""
projectConfig.py
"""

from pathlib import Path
root_path = str(Path(__file__).absolute().parent.parent.resolve())


class TestBench():
   PROMPT_METHODS = ['A', 'B', 'C']
   T_MODELS = {"bs_model": "distilbert-base-uncased", "be_model": "all-MiniLM-L6-v2", "ce_model": "ms-marco-MiniLM-L6-v2"}


class Model():
   MAX_NEW_TOKENS = 300


class Api():
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
    PROJECT_PATH = root_path
    API = Api()


if __name__ == "__main__":
  from projectConfig import Config

  print(Config.TEST_BENCH.PROMPT_METHODS)