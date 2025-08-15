"""
metricsEvaluator.py
"""

from evaluate import load
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
from app.conf.projectConfig import Config as cf

class MetricsEvaluator:

  """@brief: A class to calculate metrics like sentence similarity between texts, 
             ROUGE/BLEU scores, BERT score, etc.
  """

  def __init__(self,
               t_model_bert : str = cf.METRICS.BERT_MODEL,
               t_model_be : str = cf.METRICS.BE_MODEL, 
               t_model_ce :str = cf.METRICS.CE_MODEL):
    self.scores = {}
    self.bertscore_model = load("bertscore")
    self.bert_type_model = t_model_bert
    self.be_model = SentenceTransformer(t_model_be) # all-MiniLM-L6-v2 model has 256 as seq length
    self.ce_model = CrossEncoder("cross-encoder/" + t_model_ce)

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
    sentences = pred_text_list
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

  def set_cross_encoder_score(self, ref_text: str, pred_text_list : list, is_test_bench = False):
    # We want to compute the similarity between the query sentence...
    query = ref_text

    # ... and all sentences in the corpus
    corpus = pred_text_list
    #corpus.insert(0, ref_text) # add the ref text to the beginning of the corpus

    # 2. We rank all sentences in the corpus for the query
    ranks = self.ce_model.rank(query, corpus)

    # 3. calculate max score
    max_score = -100

    for rank in ranks:
      max_score = max(max_score, rank['score'])

    # 4. return scores in percentage vs max_score
    ce_sim_score = np.array([])
    for rank in ranks:
        score = rank['score']/max_score
        ce_sim_score = np.append(ce_sim_score,score)

    if is_test_bench:
      ce_sim_score = np.delete(ce_sim_score, 0)

    self.scores[cf.METRICS.CE_SIM_KEY] = ce_sim_score

  def get_cross_encoder_score(self)  -> list:   
      return self.scores[cf.METRICS.CE_SIM_KEY]
    
  def proc_scores(self, 
                  ref_text : str, 
                  pred_text_list : list, 
                  is_test_bench = True):
    self.set_bert_score(ref_text, pred_text_list)
    self.set_bi_encoder_score(ref_text, pred_text_list, compare_all_texts = False, is_test_bench=is_test_bench)
    self.set_cross_encoder_score(ref_text, pred_text_list, is_test_bench = is_test_bench)

  def get_scores(self) -> dict:
    return self.scores  
