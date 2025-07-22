"""
projectConfig.py
"""


class TestBench():
   PROMPT_METHODS = ['A', 'B', 'C']
   T_MODELS = {"bs_model": "distilbert-base-uncased", "be_model": "all-MiniLM-L6-v2", "ce_model": "ms-marco-MiniLM-L6-v2"}


class Config():
    TEST_BENCH = TestBench 


if __name__ == "__main__":
  from projectConfig import Config

  print(Config.TEST_BENCH.PROMPT_METHODS)