"""
setup.py file

"""
import torch
from dotenv import dotenv_values


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
        print(f"\nLoading device and environment variables:\n \
              device={self.device}, torch_dtype={self.torch_dtype}")

    def __load_env_variables(self):
        self.config = dotenv_values(".env")

if __name__ == "__main__":
  from setup import Setup

  env = Setup()