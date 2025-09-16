"""
setup.py file

"""
import torch
from dotenv import dotenv_values
import sys, os
from pathlib import Path
root_path = str(Path(__file__).absolute().parent)
sys.path.append(root_path) # import root project to env

from app.conf.logManager import Logger
import logging
logging.setLoggerClass(Logger)
log = logging.getLogger(__name__)

class Setup():

    def __init__(self):
        self.config = None
        self.device = None
        self.torch_dtype = None
        self.load_setup()

    def load_setup(self):
        self.__load_device()
        self.__load_env_variables()

    def __load_device(self):
        # set device to cuda if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch.float32 if torch.cuda.is_available() else torch.float32
        log.info(f"Loading device and environment variables:\n \
              device={self.device}, torch_dtype={self.torch_dtype}")

    def __load_env_variables(self):
        env_path = os.path.join(root_path,".env")
        self.config = dotenv_values(env_path)
        log.info(f"Loading environment variables from: {env_path}")


if __name__ == "__main__":
  from projectSetup import Setup

  env = Setup()
