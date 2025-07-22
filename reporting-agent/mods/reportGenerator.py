""" reportGenerator.py
"""

from mods.dataHandler import Report, DataHandler


class ReportGenerator:

    def __init__(self, model, tokenizer, output_type :Report):
        self.model = model
        self.tokenizer = tokenizer
        self.output_type = output_type # structured output (Report class, see DataHandler)

    def prepare_token_ids(self, pad_token_id=None, eos_token_id=None):
        '''
        Ensures pad_token_id and eos_token_id are defined.
        Uses tokenizer defaults or falls back to eos_token if needed.
        '''
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                # Set pad_token to eos_token if undefined
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    pad_token_id = self.tokenizer.eos_token_id
                else:
                    raise ValueError("Tokenizer has no pad_token or eos_token defined.")

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                raise ValueError("Tokenizer has no eos_token_id defined.")

        return pad_token_id, eos_token_id

    def generate_report(
        self,
        prompt: str,
        max_length: int = 1000,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1, # Use num_beam > 1 ONLY if do_sample == False
        pad_token_id: int = None,
        eos_token_id: int = None,
        **kwargs):
        """
        Text generation from the model. Since we are using the outlines library,
         there is no need for tokenizing and decoding the prompt.
         The prompt, the structured output and the kwargs are passed to the generation.
        """

        # inputs = self.tokenizer(prompt, return_tensors="pt")

        pad_token_id, eos_token_id = self.prepare_token_ids(pad_token_id, eos_token_id)

        generation_args = {
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }

        generation_args.update(kwargs)

        output = self.model(prompt, output_type=self.output_type, **generation_args)
        # outputs = self.model.generate(**inputs, **generation_args)
        # text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return output