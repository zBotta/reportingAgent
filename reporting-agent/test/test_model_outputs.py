from mods.dataHandler import DataHandler, Report
from mods.promptGenerator import PromptGenerator
from mods.modelLoader import ModelLoader
import outlines
from ...setup import Setup


def test_structured_outputs():
    report_index = 20 # row to pick in df_reports
    prompt_method = 'A'

    # Load device and env variables
    env = Setup()
    dh = DataHandler()
    ml = ModelLoader(model_id="gpt2", device=env.device, torch_dtype=env.torch_)
    model, tokenizer = ml.load_model(hf_token = env.config["HF_TOKEN"])
    device = env.device
    df_reports = dh.import_reports()
    row = df_reports.loc[report_index, 'what':'contingency_actions']
    prompt = PromptGenerator(**row.to_dict()).create_prompt(prompt_method)


    model_outlines = outlines.from_transformers(model, tokenizer)
    result = model_outlines(prompt, output_type=Report, max_new_tokens = 200)
    title = Report.model_validate_json(result).title
    report = Report.model_validate_json(result).report.strip()
    assert ( len(title) > 0 and len(report) > 0 )