""" modelLoader.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig
import outlines
from conf.projectConfig import Config as cf


class ModelLoader:
    def __init__(self, model_id: str, device, torch_dtype):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        torch.set_default_device(self.device)

    def load_model(self, hf_token = ""):
        """ Imports the model and tokenizer from HF and returns a model compatible with 
        the outlines structured output: e.g.
        
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

    def get_default_parameters(self):
        gc = GenerationConfig.from_pretrained(self.model_id)
        param_dict = {}
        for param_name in cf.MODEL.PARAM_LIST:
            param_dict.update({param_name,gc.__getattribute__(param_name)})
        return param_dict