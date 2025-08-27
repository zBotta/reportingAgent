"""
metricsEvaluator.py
"""

from evaluate import load
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
from app.conf.projectConfig import Config as cf
import torch

class MetricsEvaluator:

  """@brief: A class to calculate metrics like sentence similarity between texts, 
             ROUGE/BLEU scores, BERT score, etc.
  """

  def __init__(self,
               t_model_bert : str = cf.METRICS.BERT_MODEL,
               t_model_be : str = cf.METRICS.BE_MODEL, 
               t_model_ce :str = cf.METRICS.CE_MODEL):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.scores = {}
    self.bertscore_model = load("bertscore")
    self.rouge_model = load("rouge")
    self.bleu_model = load("bleu") 
    self.bert_type_model = t_model_bert
    self.be_model = SentenceTransformer(t_model_be) # all-MiniLM-L6-v2 model has 256 as seq length
    self.ce_model = CrossEncoder("cross-encoder/" + t_model_ce, max_length=cf.MODEL.MAX_NEW_TOKENS, device=self.device)

  def set_rouge_score(self, ref_text: str, pred_text_list: list):
     """
      Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
      between predicted texts and reference texts using Hugging Face's
      `evaluate` library.

      Args:
          predicitions (list): A list of generated text strings.
          references (list): A list of reference text strings. 
      """
     res = self.rouge_model.compute(predictions=pred_text_list, references=[ref_text])
     self.scores.update(res)

  def set_bleu_score(self, ref_text: str , pred_text_list: list):
    """
    Compute smoothed BLEU scores between predicted texts and reference
    texts using Hugging Face's `evaluate` library. Also expands and
    calculates precision values for 1–4 n-grams.

    Args:
        predicitions (list): A list of generated text strings.
        references (list): A list of reference text strings.

    Calculates:
        dict: Dictionary containing BLEU score, n-gram precisions,
              brevity penalty, and related statistics.
    """
    # avoid bleu.compute division by zero errors
    for idx, pred_text in enumerate(pred_text_list):
      if len(pred_text) == 0: # If the predicted text is empty, we set it to a dummy value
        not_na_text = "_" # Avoid empty predicted text
        pred_text_list[idx] = not_na_text
    res = self.bleu_model.compute(predictions=pred_text_list, references=[ref_text], smooth=True,) # Use smooth=True to avoid reporting score 0 when there is no high-order n-gram overlap (such as 4-grams)
    deleted_keys = ['brevity_penalty', 'length_ratio', 'translation_length', 'reference_length']
    for k in deleted_keys:
      res.pop(k)
    
    # save the n-grams list into dict keys
    precision_n_grams_scores = res.pop('precisions')
    for idx, item in enumerate(precision_n_grams_scores):
        res.update({f'b_{idx+1}_grams': item})
    self.scores.update(res)

  def set_bert_score(self, ref_text : str, pred_text_list : list):
    """ Takes a reference text and a list of predicted texts and returns a tuple with a the precision, recall and f1 score for each predicted text.
        Each precision, recall and f1 object  is a list
    """
    predictions = pred_text_list
    references = [ref_text]
    results = self.bertscore_model.compute(predictions=predictions, references=references, model_type=self.bert_type_model)
    self.scores[cf.METRICS.BS_PRECISION_KEY], self.scores[cf.METRICS.BS_RECALL_KEY], self.scores[cf.METRICS.BS_F1_KEY] = results["precision"], results["recall"], results["f1"]
    
  def get_bert_score(self) -> tuple:
    return self.scores[cf.METRICS.BS_PRECISION_KEY], self.scores[cf.METRICS.BS_RECALL_KEY], self.scores[cf.METRICS.BS_F1_KEY]

  def set_bi_encoder_score(self, ref_text : str, pred_text_list : list, compare_all_texts = False, is_test_bench = False):
    """
    compare_all_texts : If True, we are going to compare all the predicted texts between them (only for data validation purposes. Are the reports similar between them?).
                        If False, we take the first row of the similarity matrix to compare only wrt to the reference text
    is_test_bench: Is used for unifying the amount of scores between similarity methods
    """
    
    # The sentences to encode
    sentences = list(pred_text_list)
    sentences.insert(0, ref_text) # add the ref text to the beginning of the list
    # 2. Calculate embeddings by calling model.encode()
    embeddings = self.be_model.encode(sentences)

    # 3. Calculate the embedding similarities
    similarities = self.be_model.similarity(embeddings, embeddings)
    # Take the first row of the similarity matrix if we want to compare only wrt to the reference text. If not return all the similarity matrix
    be_scores = similarities.cpu().numpy() if compare_all_texts else similarities.cpu().numpy()[0]
    if is_test_bench:
      be_scores = np.delete(be_scores, 0)
    self.scores[cf.METRICS.BE_SIM_KEY] = be_scores 
    
  def get_bi_encoder_score(self) -> np.dtype:
    return self.scores[cf.METRICS.BE_SIM_KEY]

  def set_cross_encoder_score(self, ref_text: str, pred_text_list : list):
    # We want to compute the similarity between the query sentence and the corpus
    query = ref_text
    corpus = list(pred_text_list) # Make a copy of the list to avoid modifying the original list

    # 2) Batch score (ref, pred) pairs — no 'rank' over a big corpus
    #    Keep batches tiny to cap memory (adjust if you have more VRAM/RAM)
    batch_size = cf.METRICS.CE_BATCH_SIZE
    pairs = [(query, c) for c in corpus]
    ce_scores = []  # raw (ref,pred) scores
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            ce_scores.extend(self.ce_model.predict(pairs[i:i + batch_size]))
    ce_scores = np.asarray(ce_scores, dtype=np.float32)

    # 3) Normalization
    with torch.inference_mode():
        denom = float(self.ce_model.predict([(query, query)])[0]) + 1e-8

    # 4) Vectorized normalization (no np.append in a loop)
    ce_sim_score = ce_scores / denom

    # 5) Store result (same key/shape as before)
    self.scores[cf.METRICS.CE_SIM_KEY] = ce_sim_score

  def get_cross_encoder_score(self)  -> list:   
      return self.scores[cf.METRICS.CE_SIM_KEY]
    
  def proc_scores(self, 
                  ref_text : str, 
                  pred_text_list : list, 
                  is_test_bench = True):
    
    # IMPORTANT: The order of execution of the methods will determine the order of columns of the TestBench results
    self.set_bert_score(ref_text, pred_text_list) # BERT SCORE SHOULD ALWAYS BE CALLED FIRST
    self.set_rouge_score(ref_text, pred_text_list) 
    self.set_bleu_score(ref_text, pred_text_list)
    self.set_bi_encoder_score(ref_text, pred_text_list, compare_all_texts = False, is_test_bench=is_test_bench)
    self.set_cross_encoder_score(ref_text, pred_text_list)

  def get_scores(self) -> dict:
    return self.scores  
