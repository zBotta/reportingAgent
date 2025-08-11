""" modelLoader.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig
import outlines
from conf.projectConfig import Config as cf


class ModelLoader:
    """ This class permits to load and handle parameters of the models """
    def __init__(self, model_id: str, device, torch_dtype):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        torch.set_default_device(self.device)
        print(f"The default parameters of the model are:\n {self.get_default_tunable_parameters(verbose=True)}")

    def load_model(self, hf_token = ""):
        """ Imports the model and tokenizer from HF and returns a tuple with the model and the tokenizer.
        The model is compatible with the outlines structured output: e.g.
        
            model_outlines = outlines.from_transformers(model, tokenizer)
            result = model_outlines(prompt, output_type=PYDANTIC_CLASS, 
                                **kwargs = [max_new_tokens, temperature, top_k, etc])
        """

        model_kwargs = {}
        model_kwargs['token'] = hf_token

        if 'gpt2' in self.model_id.lower():
            tokenizer_class = GPT2Tokenizer
            model_class = GPT2LMHeadModel

        else:
            tokenizer_class = AutoTokenizer
            model_class = AutoModelForCausalLM

        if 'llama' in self.model_id.lower() or 'janus' in self.model_id.lower():
            # model_kwargs['trust_remote_code'] = True
            if hf_token == "":
                ConnectionError("Hugging Face token is needed for loading to Llama or Janus models")


        # Load tokenizer and model
        tokenizer = tokenizer_class.from_pretrained(self.model_id, **model_kwargs)
        model = model_class.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            **model_kwargs
        )

        model.to(self.device)
        
        # We charge the model and tokenizer into model_outlines to be able to handle structured outputs
        model_outlines = outlines.from_transformers(model, tokenizer)

        return model_outlines, tokenizer

<<<<<<< HEAD
    def get_default_parameters(self, verbose = False) -> dict:
        """ Get the default parameters of the model
        verbose: If True, the missing parameters are printed out
        """
        gc = GenerationConfig.from_pretrained(self.model_id)
        param_dict = {}
        for param_name in cf.MODEL.PARAM_LIST:
            try:
                att_val = gc.__getattribute__(param_name)
            except Exception as e:
                att_val = cf.MODEL.VAL_IF_NOT_IN_PARAM_LIST
                # TODO: change this print to debug in log
                if verbose:
                    print(f"No attribute {param_name} found in GenerationConfig, for model_id={self.model_id}")
            finally:    
                param_dict.update({param_name: att_val})
        return param_dict
    
    def get_default_tunable_parameters(self, verbose = False) -> dict:
        """ Filters the default parameter values and gets the tunable ones (value that are not cf.MODEL.VAL_IF_NOT_IN_PARAM_LIST)"""
        def_param = self.get_default_parameters(verbose=verbose)
        tunable_param = {k: v for k, v in def_param.items() if v is not cf.MODEL.VAL_IF_NOT_IN_PARAM_LIST}
        return tunable_param
    
=======

    def get_default_parameters(self):
        generation_config = GenerationConfig.from_pretrained(self.model_id) # changed gc to generation_config, collides with gc module (garbage collector )
        param_dict = {}
        for param_name in cf.MODEL.PARAM_LIST:
            try:
                att_val = generation_config.__getattribute__(param_name)
            except AttributeError:
                att_val = None
                print(f"No attribute '{param_name}' found in GenerationConfig for model '{self.model_id}'")
            except Exception as e:
                att_val = None
                print(f"An unexpected error occurred while getting attribute '{param_name}': {e}")
            finally:
                param_dict[param_name] = att_val  # Assign directly to the dictionary
        return param_dict
>>>>>>> be646dc5bbc3923eb585f37e1c81d76509eeac73
