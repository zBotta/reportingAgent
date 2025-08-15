              
from projectSetup import Setup
from conf.projectConfig import Config as cf
from mods.dataHandler import DataHandler, Report
from mods.promptGenerator import PromptGenerator
from mods.modelLoader import ModelLoader
import outlines


def test_get_default_parameters():
    env = Setup()
    model_id = 'openai-community/gpt2'
    ml = ModelLoader(model_id=model_id, device=env.device, torch_dtype=env.torch_dtype)
    dp = ml._get_default_parameters()
    is_value = True
    for ref_val in cf.MODEL.PARAM_LIST:
        is_value = (ref_val in dp) and is_value

    assert is_value
    
