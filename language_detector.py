import clean_text 
import helpers 

import pandas as pd 
import numpy as np
import pickle
import joblib 
import sentencepiece as spm


def language_detector(data, textcolname):
  # preset values
  VOCAB_SIZE = 50000
  ALPHA = 1e-2
  THRESHOLD = 0.8
  N_VALUE = 3
  N_FEATURE = 300
  MARGIN = 0.006
  with open("N_GRAMS.pickle", "rb") as handle:
    N_GRAMS = pickle.load(handle)

  DATA = data.copy()

  # standardise column name
  data = data[[textcolname]].copy()
  data.rename(columns={textcolname:"text"}, inplace=True)
  # clean text
  data["raw"] = data["text"]
  data = clean_text.lowercase(data, "text")
  data = clean_text.rm_multiplespace(data, "text")
  data = clean_text.rm_numbers(data, "text")
  data = clean_text.pre_tokenize(data, "text")
  # tokenize text
  data["tokens"] = data["text"]
  data = clean_text.tokenize(data, "tokens", spm.SentencePieceProcessor(model_file = "tokenizer.model"))
  # vectorize text
  X = clean_text.vectorize(data, "tokens", VOCAB_SIZE)
  # nb model
  nb_model = joblib.load("multinomialnb_model.joblib")
  data = helpers.nb_predict(nb_model, X, data, "text", t=THRESHOLD)
  # unicode rule
  data = helpers.unicode_predict(data, N_GRAMS, N_VALUE, N_FEATURE, MARGIN)
  # drop columns
  data = data[["unicode_predict"]].copy()
  data.rename(columns={"unicode_predict":"predicted_language"}, inplace=True)
  
  DATA = pd.concat([DATA, data], axis=1)
  
  return DATA