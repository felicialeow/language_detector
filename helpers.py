# required libraries
import pandas as pd
import numpy as np
import iso639
import math
import random
import time
import re
import unicodedata
import pickle
import joblib
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sentencepiece as spm


# data processing functions 
def rm_duplicates(data):
  """
  rm_duplicates(data)
  Remove duplicated rows from dataset and print number of rows dropped.

  Parameters
  ----------
  data: pandas dataframe 

  Returns 
  -------
  dataframe
    Dataframe with duplicats removed
  """
  n1 = len(data)
  data.drop_duplicates(keep="first", inplace=True, ignore_index=True)
  n2 = len(data)
  print(n1-n2, "duplicates removed from dataset")
  return data

def train_test_split(data):
  """
  train_test_split(data)
  Split dataset into train (80%) and test (20%) set, print the size of each set.

  Parameters
  ----------
  data: pandas dataframe 

  Returns 
  -------
  dataframe
    A new column `split` created to label train vs test set
  """
  random.seed(1)
  n = len(data)
  train = round(n*0.8)
  test = len(data) - train
  test_indices = random.sample(range(n), test) 
  data["split"] = "train"
  data.iloc[test_indices, -1] = "test"
  print("Data split into", train, "train set,", test, "test set")
  return data

def add_source(data, sourcename):
  """
  add_source(data, sourcename)
  Add data source column.

  Parameters
  ----------
  data: pandas dataframe
  sourcename: str

  Returns
  -------
  dataframe
    Dataset with `source` column added and value equals `sourcename`
  """
  data["source"] = sourcename
  return data

def iso639_1_to_language(code):
  """ 
  iso639_1_to_language(code)
  Convert iso639-1 code to language name.

  Parameters
  ----------
  code: str

  Returns 
  -------
  str
    Name of language based on ISO 639 system, return `unknown` if code is not found
  """
  try:
    return iso639.Language.from_part1(code).name
  except:
    return "unknown"
    
def iso639_3_to_language(code):
  """ 
  iso639_3_to_language(code)
  Convert iso639-3 code to language name.

  Parameters
  ----------
  code: str

  Returns 
  -------
  str
    Name of language based on ISO 639 system, return `unknown` if code is not found
  """
  try:
    return iso639.Language.from_part3(code).name
  except:
    return "unknown"

def replace_lang(lang, original, new):
  """
  replace_lang(lang, original, new)
  Substitute language name.

  Parameters
  ----------
  lang: str
    language name 
  original: str
    language name to be replaced
  new: str
    language name to replace

  Returns
  -------
  str
    Substituted language name
  """
  if lang == original:
    return new
  else:
    return lang


# prediction accuracy metrics 
def fnr(true, predict):
  """
  Calculate False Negative Rate (FNR).
  FNR = number of false negatives/ number of true positives

  Parameters 
  ----------
  true: array 
    array of true classes 
  predict: array
    array of predicted classes
  
  Returns
  -------
  float 
  """
  n_pos = sum(true == "English")
  n_fn = sum((true == "English") & (predict != "English"))
  fnr = n_fn/n_pos*100
  return fnr

def fpr(true, predict):
  """
  Calculate False Positive Rate (FPR).
  FPR = number of false positives/ number of true negatives

  Parameters 
  ----------
  true: array 
    array of true classes 
  predict: array
    array of predicted classes
  
  Returns
  -------
  float 
  """
  n_neg = sum(true != "English")
  n_fp = sum((true != "English") & (predict == "English"))
  fpr = n_fp/n_neg*100
  return fpr

def print_score(score, prefix=""):
  """
  print_score(score, prefix)
  Print accuracy score of model.

  Parameters
  ----------
  score: float 
    accuracy score 
  prefix: str 
    prefix to label accuracy score type
  """
  print(prefix, " accuracy: %.2f" % score, "%", sep="")

def print_fnr(true, predict, prefix):
  """
  print_fnr(true, predict, prefix)
  Print false negative rate (FNR). 
  FNR = number of false negatives/number of true positives 

  Parameters 
  ----------
  true: array 
    array of true classes 
  predict: array
    array of predicted classes 
  prefix: str

  Returns 
  -------
  float
  """
  score = fnr(true, predict)
  print(prefix, " FNR: %.2f" % score, "%", sep="")
  return score

def print_fpr(true, predict, prefix):
  """
  print_fpr(true, predict, prefix)
  Print false positive rate (FPR). 
  FPR = number of false positives/ number of true negatives 

  Parameters 
  ----------
  true: array 
    array of true classes 
  predict: array
    array of predicted classes 
  prefix: str

  Returns 
  -------
  float
  """
  score = fpr(true, predict)
  print(prefix, " FPR: %.2f" % score, "%", sep="")
  return score

def threshold(charlen, t=0.9):
  """
  threshold(charlen, t)
  Calculate threshold probability to classify text as predicted class or `unknown`.
  threshold = exp(1/`charlen`) * t 

  Parameters
  ----------
  charlen: int
    length of string
  t: float
    probability confidence level
  
  Returns
  -------
  float
    threshold value
  """
  if charlen < 10:
    return t
  else:
    return math.exp(1/charlen)*t

def nb_predict(model, X, data, textcol, t=0.9):
  """
  nb_predict(model, X, data, textcol, t)
  Classify text as `model` predicted class if probability above threshold, else `unknown`.

  Parameters
  ----------
  model: sklearn model 
    trained model used for prediction
  X: array 
    input array for model
  data: pandas dataframe
  textcol: str 
  t: float
    probability confidence level
  
  Returns
  -------
  dataframe 
    Dataframe with new columns added:
    `best_predict`: model predicted class
    `best_prob`: probability of predicted class
    `threshold`: threshold probability 
    `nb_predict`: final predicted class
  """
  def prediction(row):
    if row["best_prob"] >= row["threshold"]:
      value = row["best_predict"]
    else:
      value = "UNKNOWN"
    return value

  data = data.copy()
  pred_class = model.predict(X)
  data["best_predict"] = pred_class
  pred_prob = model.predict_proba(X)
  pred_prob = list(map(max, pred_prob))
  data["best_prob"] = pred_prob
  data["threshold"] = data[textcol].apply(lambda text: threshold(len(text), t))
  data["nb_predict"] = data.apply(lambda row: prediction(row), axis=1)
  return data


# naive bayes model parameter tuning
def nb_tune_alpha(X, y, data, alphalist):
  """
  nb_tune_alpha(X, y, data, alphalist)
  Tune smoothing parameter (alpha) in multinomial naive bayes model 
    using 5 fold cross validation. 
  Class prediction using nb_predict(model, X, data, textcol, t=0.9) function.
  
  Parameters
  ----------
  X: array
    input array for model
  y: array
    output array for model
  data: pandas dataframe 
  alphalist: array
    list of alpha values to tune
  
  Returns 
  -------
  dict
    A dictionary with keys:
    `alpha`: list of alpha values 
    `train_score`: list of mean training accuracy
    `train_fnr`: list of mean training false negative rate 
    `train_fpr`: list of mean training false positive rate
    `test_score`: list of mean test accuracy
    `test_fnr`: list of mean test false negative rate 
    `test_fpr`: list of mean test false positive rate
  """
  N_FOLD = 5
  kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=1)
  kf_indices = {}
  for i, (train_index, test_index) in enumerate(kf.split(X)):
    kf_indices[i] = (train_index, test_index)

  cv_results = {"alpha":[], 
                "train_score":[], 
                "train_fnr":[], 
                "train_fpr":[],
                "test_score":[], 
                "test_fnr":[], 
                "test_fpr":[]}

  n_alpha = len(alphalist)
  print("Fitting", N_FOLD, "folds for each of", n_alpha, "candidates, totalling", N_FOLD*n_alpha, "fits")
  for a in alphalist:
    train_scores = []
    train_fnrs = []
    train_fprs = []
    test_scores = []
    test_fnrs = []
    test_fprs = []

    print("alpha =", a)
    nb_model = MultinomialNB(alpha=a)

    start_time = time.time()
    for fold in range(5):
      training_folds_X = X[kf_indices[fold][0]]
      training_folds_y = y[kf_indices[fold][0]]
      holdout_fold_X = X[kf_indices[fold][1]]
      holdout_fold_y = y[kf_indices[fold][1]]

      nb_model.fit(training_folds_X, training_folds_y)

      train_predict = nb_predict(nb_model, training_folds_X, data.iloc[kf_indices[fold][0],], "text")
      train_score = accuracy_score(train_predict["language"], train_predict["nb_predict"], normalize=True)*100
      train_fnr = fnr(train_predict["language"], train_predict["nb_predict"])
      train_fpr = fpr(train_predict["language"], train_predict["nb_predict"])
      
      test_predict = nb_predict(nb_model, holdout_fold_X, data.iloc[kf_indices[fold][1],], "text")
      test_score = accuracy_score(test_predict["language"], test_predict["nb_predict"], normalize=True)*100
      test_fnr = fnr(test_predict["language"], test_predict["nb_predict"])
      test_fpr = fpr(test_predict["language"], test_predict["nb_predict"])

      train_scores.append(train_score)
      train_fnrs.append(train_fnr)
      train_fprs.append(train_fpr)

      test_scores.append(test_score)
      test_fnrs.append(test_fnr)
      test_fprs.append(test_fpr)

    end_time = time.time()
    total_time = end_time - start_time
    mean_train_score = np.mean(train_scores)
    mean_train_fnr = np.mean(train_fnrs)
    mean_train_fpr = np.mean(train_fpr)
    mean_test_score = np.mean(test_scores)
    mean_test_fnr = np.mean(test_fnrs)
    mean_test_fpr = np.mean(test_fprs)

    cv_results["alpha"] += [a]
    cv_results["train_score"] += [mean_train_score]
    cv_results["train_fnr"] += [mean_train_fnr] 
    cv_results["train_fpr"] += [mean_train_fpr] 
    cv_results["test_score"] += [mean_test_score] 
    cv_results["test_fnr"] += [mean_test_fnr]
    cv_results["test_fpr"] += [mean_test_fpr]

    print("[CV 5/5] END ..... total time:", "{:0.2f}".format(total_time), "s, train score=", "{:0.2f}".format(mean_train_score), ", test score=", "{:0.2f}".format(mean_test_score), sep="")
  return cv_results

def nb_tune_threshold(X, y, data, alpha, thresholdlist):
  """
  nb_tune_threshold(X, y, data, alpha, thresholdlist)
  Tune probability confidence level (t) in the threshold formula using 5 fold cross validation. 
    threshold = exp(1/text length)*t
  
  Parameters
  ----------
  X: array
    input array for model
  y: array
    output array for model
  data: pandas dataframe 
  alpha: float
    smoothing parameter in the multinomialNB(alpha=alpha) function
  thresholdlist: list
    list of threshold level to tune (t betweens 0 and 0.9)
  
  Returns 
  -------
  dict
    A dictionary with keys:
    `threshold`: list of threshold values 
    `train_score`: list of mean training accuracy
    `train_fnr`: list of mean training false negative rate 
    `train_fpr`: list of mean training false positive rate
    `test_score`: list of mean test accuracy
    `test_fnr`: list of mean test false negative rate 
    `test_fpr`: list of mean test false positive rate
  """
  N_FOLD = 5
  kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=1)
  kf_indices = {}
  for i, (train_index, test_index) in enumerate(kf.split(X)):
    kf_indices[i] = (train_index, test_index)

  cv_results = {"threshold":[], 
                "train_score":[], 
                "train_fnr":[], 
                "train_fpr":[],
                "test_score":[], 
                "test_fnr":[], 
                "test_fpr":[]}

  nb_model = MultinomialNB(alpha=alpha)

  n_threshold = len(thresholdlist)
  print("Fitting", N_FOLD, "folds for each of", n_threshold, "candidates, totalling", N_FOLD*n_threshold, "fits")
  for t in thresholdlist:
    train_scores = []
    train_fnrs = []
    train_fprs = []
    test_scores = []
    test_fnrs = []
    test_fprs = []

    print("threshold =", t)
    start_time = time.time()
    for fold in range(5):
      training_folds_X = X[kf_indices[fold][0]]
      training_folds_y = y[kf_indices[fold][0]]
      holdout_fold_X = X[kf_indices[fold][1]]
      holdout_fold_y = y[kf_indices[fold][1]]

      nb_model.fit(training_folds_X, training_folds_y)

      train_predict = nb_predict(nb_model, training_folds_X, data.iloc[kf_indices[fold][0],], "text", t=t)
      train_score = accuracy_score(train_predict["language"], train_predict["nb_predict"], normalize=True)*100
      train_fnr = fnr(train_predict["language"], train_predict["nb_predict"])
      train_fpr = fpr(train_predict["language"], train_predict["nb_predict"])
      
      test_predict = nb_predict(nb_model, holdout_fold_X, data.iloc[kf_indices[fold][1],], "text", t=t)
      test_score = accuracy_score(test_predict["language"], test_predict["nb_predict"], normalize=True)*100
      test_fnr = fnr(test_predict["language"], test_predict["nb_predict"])
      test_fpr = fpr(test_predict["language"], test_predict["nb_predict"])

      train_scores.append(train_score)
      train_fnrs.append(train_fnr)
      train_fprs.append(train_fpr)

      test_scores.append(test_score)
      test_fnrs.append(test_fnr)
      test_fprs.append(test_fpr)

    end_time = time.time()
    total_time = end_time - start_time
    mean_train_score = np.mean(train_scores)
    mean_train_fnr = np.mean(train_fnrs)
    mean_train_fpr = np.mean(train_fpr)
    mean_test_score = np.mean(test_scores)
    mean_test_fnr = np.mean(test_fnrs)
    mean_test_fpr = np.mean(test_fprs)

    cv_results["threshold"] += [t]
    cv_results["train_score"] += [mean_train_score]
    cv_results["train_fnr"] += [mean_train_fnr] 
    cv_results["train_fpr"] += [mean_train_fpr] 
    cv_results["test_score"] += [mean_test_score] 
    cv_results["test_fnr"] += [mean_test_fnr]
    cv_results["test_fpr"] += [mean_test_fpr]

    print("[CV 5/5] END ..... total time:", "{:0.2f}".format(total_time), "s, train score=", "{:0.2f}".format(mean_train_score), ", test score=", "{:0.2f}".format(mean_test_score), sep="")
  return cv_results

def print_cv(cv_results, param, logaxis=False):
  """
  print_cv(cv_results, param, logaxis=False)
  Plot accuracy score, false negative rate, false positive rate for different values of parameter

  Parameters
  ----------
  cv_results: dict
    CV results from nb_tune_*() functions 
  param: str
  logaxis: boolean
  """
  df = pd.DataFrame(cv_results)
  if logaxis:
    df[param] = df[param].apply(lambda v: math.log10(v))
  fig, axes = plt.subplots(3, 1)
  df.plot.line(x=param, y=["train_score", "test_score"], ax=axes[0])
  df.plot.line(x=param, y=["train_fnr", "test_fnr"], ax=axes[1])
  df.plot.line(x=param, y=["train_fpr", "test_fpr"], ax=axes[2])


# unicode blocks 
def unicode_blocks():
  """
  unicode_block()
  Load unicode character database from https://unicode.org/Public/UNIDATA/Blocks.txt

  Returns
  -------
  dict: 
    (start_idx, end_idx): language name
  """
  block_exp = re.compile(r"^(....)\.\.(....); (.*)$")
  block_dict = {}
  for line in open("Data/Original/Blocks.txt"):
    if line.startswith('#'):
      continue
    line = line.strip()
    if not line:
      continue
    matchedobj = block_exp.match(line)
    if matchedobj:
      start = int(matchedobj.group(1),16)
      end = int(matchedobj.group(2),16)
      blockname = matchedobj.group(3)
      block_dict[(start, end)] = blockname
  return block_dict

BLOCKS = unicode_blocks()

def count_blocks(text):
  """
  count_blocks(text)
  Count the proportion of characters in each unicode block.

  Parameters
  ----------
  text: str

  Returns
  -------
  Counter
    block_name: proportion of character in `block_name`
  """
  if text:
    text = unicodedata.normalize("NFC", text)
    all_lang = []
    for c in text:
      if c.isalpha():
        all_lang.append([lang_name for (start_idx, end_idx), lang_name in BLOCKS.items() if ord(c) in range(start_idx, end_idx)][0])
    all_lang = Counter(all_lang)
    total_count = all_lang.total()
    all_lang = Counter({lang: count/total_count for lang, count in all_lang.items()})
    if all_lang:
      return all_lang
  return Counter()

def highest_block(blocks):
  """
  highest_block(blocks)
  Identify the block with highest proportion; proportion has to be at least 30%.

  Parameters
  ----------
  blocks: Counter

  Returns
  -------
  str
    block name with highest proportion, or "UNKNOWN" if all block is less than 30%
  """
  if blocks.most_common(1):
    lang, percent = blocks.most_common(1)[0]
    if percent >= 0.3:
      return lang
  return "UNKNOWN"


# script and language name mapping 
SCRIPT_TO_LANGUAGE = {
  "Malayalam":                         "Malayalam",
  "Tamil":                             "Tamil",
  "Thai":                              "Thai",
  "Devanagari":                        "Hindi",
  "Latin Extended Additional":         "Vietnamese",

  
  "Greek and Coptic":                  "Greek",
  "Greek Extended":                    "Greek",
  
  "Hiragana":                          "Japanese",
  "Katakana":                          "Japanese",
  "Katakana Phonetic Extensions":      "Japanese",
  
  "Hangul Syllables":                  "Korean",
  "Hangul Jamo":                       "Korean",
  "Hangul Compatibility Jamo":         "Korean",
  "Hangul Jamo Extended-A":            "Korean",
  'Hangul Jamo Extended-B':            "Korean",
  
  "CJK Radicals Supplement":           "Chinese",
  "CJK Symbols and Punctuation":       "Chinese",
  "CJK Strokes":                       "Chinese",
  "CJK Compatibility":                 "Chinese",
  "CJK Unified Ideographs Extension A":"Chinese",
  "CJK Unified Ideographs":            "Chinese",
  "CJK Compatibility Ideographs":      "Chinese",
  "CJK Compatibility Forms":           "Chinese",
  "Kangxi Radicals":                   "Chinese",
  "Bopomofo":                          "Chinese",
  "Bopomofo Extended":                 "Chinese",
      
  "Latin-1 Supplement":                "EXTENDED LATIN",
  "Latin Extended-A":                  "EXTENDED LATIN",
  "Latin Extended-B":                  "EXTENDED LATIN",
  "Latin Extended-C":                  "EXTENDED LATIN",
  "Latin Extended-D":                  "EXTENDED LATIN",
  "Latin Extended-E":                  "EXTENDED LATIN",
  
  "Arabic":                            "ARABIC",
  "Cyrillic":                          "CYRILLIC",
  "CJK Unified Ideographs":            "CJK",
  "Basic Latin":                       "BASIC LATIN"
}

SCRIPT_TO_MULTILANGUAGE = {
  "CJK":            ["Chinese", 
                      "Japanese", 
                      "Korean"],
  "CYRILLIC":       ["Russian", 
                      "Bulgarian"],
  "ARABIC":         ["Arabic", 
                      "Urdu", 
                      "Pushto", 
                      "Persian"],
  "BASIC LATIN":    ["English", 
                      "Latin", 
                      "Indonesian", 
                      "Swahili"],
  "EXTENDED LATIN": ["Portuguese", 
                      "French", 
                      "Dutch", 
                      "Spanish", 
                      "Danish", 
                      "Italian", 
                      "Turkish", 
                      "Swedish", 
                      "German", 
                      "Estonian", 
                      "Romanian", 
                      "Polish", 
                      "Vietnamese"]
}


# unicode prediction
def create_ngram(text, n_value=3, n_feature=300, limit=True):
  """
  create_ngram(text, n_value=3, n_feature=300, limit=True)
  Create n gram.

  Parameters
  ----------
  text: array
  n_value: int
  n_feature: int
  limit: boolean
    max_features = N_FEATURE if `True`
  
  Returns 
  -------
  dictionary
    n-gram: ranking
  """

  if limit:
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(n_value,n_value), max_features=n_feature)
  else:
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(n_value,n_value))
  term_matrix = vectorizer.fit_transform(text)
  term_matrix = term_matrix.toarray().sum(axis=0)
  term_matrix = term_matrix/term_matrix.sum()
  ngram_proportion = sorted(zip(vectorizer.get_feature_names_out(), term_matrix), key=lambda pair: pair[1], reverse=True)  
  if limit:
    ngram_rank = {ngram[0]:rank for ngram, rank in zip(ngram_proportion, range(1, n_feature+1))}
  else:
    ngram_rank = {ngram[0]:rank for ngram, rank in zip(ngram_proportion, range(1, len(ngram_proportion)+1))}
  return ngram_rank 

def train_ngrams(data, n_value=3, n_feature=300):
  """
  train_ngrams(data, n_value=3, n_feature=300)
  Create a dictionary of n grams for each language.

  Parameters
  ----------
  data: pandas dataframe
    contains `language` and `text` column
  n_value: int
  n_feature: int

  Returns
  -------
  dictionary
    language: dictionary of n-gram, ranking 
  """
  languages = data["language"].unique()
  ngrams = {}
  for lang in languages:
    texts = data.loc[data["language"]==lang, "text"]
    ngrams[lang] = create_ngram(texts, n_value, n_feature, limit=True)
  return ngrams

def cjk_rule(blocks):
  """
  cjk_rule(blocks)
  Predict language of text as Chinese, Japanese or Korean based on the unicode blocks present.

  Parameters
  ----------
  blocks: Counter 

  Returns
  -------
  str
  """
  if (("Hiragana" in blocks) or
      ("Katakana" in blocks) or
      ("Katakana Phonetic Extensions" in blocks)):
    return "Japanese"
  if (("Hangul Syllables" in blocks) or
      ("Hangul Jamo" in blocks) or
      ("Hangul Compatibility Jamo" in blocks) or
      ("Hangul Jamo Extended-A" in blocks) or
      ('Hangul Jamo Extended-B' in blocks)):
    return "Korean"
  return "Chinese"

def unique_chars(text, script):
  """
  unique_chars(text, script)
  Predict language of text based on unique characters found in text and type of script.

  Parameters
  ----------
  text: str
  script: str 

  Returns
  -------
  str or boolean
    False if no unique character found in text, else return the language
  """
  russian_chars = re.compile("[ёыэ]")
  urdu_chars = re.compile("[ٹڈڑںھہۂے]")
  pushto_chars = re.compile("[ټځڅډړږښګڼۍې]")
  arabic_chars = re.compile("[إكى]")
  vietnamese_chars = re.compile("[ắằẵẳấầẫẩảạặậđếềễểẽẻẹệĩỉịốồỗổỏơớờỡởợọộũủưứừữửựụýỳỹỷỵ]")
  turkish_chars = re.compile("[ğIİış]")
  romanian_chars = re.compile("[șț]")
  polish_chars = re.compile("[ąćęłńśźż]")
  french_chars = re.compile("[œûÿ]")
  estonian_chars = re.compile("[šž]")
  spanish_chars = re.compile("[ñ]")
  german_chars = re.compile("[ß]")
  danish_chars = re.compile("[ø]")
  if script == "CYRILLIC":
    if russian_chars.findall(text):
      return "Russian"
  elif script == "ARABIC":
    if urdu_chars.findall(text):
      return "Urdu"
    if pushto_chars.findall(text):
      return "Pushto"
    if arabic_chars.findall(text):
      return "Arabic"
  elif (script == "EXTENDED LATIN") or (script == "BASIC LATIN"):
    if vietnamese_chars.findall(text):
      return "Vietnamese"
    if turkish_chars.findall(text):
      return "Turkish"
    if romanian_chars.findall(text):
      return "Romanian"
    if polish_chars.findall(text):
      return "Polish"
    if french_chars.findall(text):
      return "French"
    if estonian_chars.findall(text):
      return "Estonian"
    if spanish_chars.findall(text):
      return "Spanish"
    if german_chars.findall(text):
      return "German"
    if danish_chars.findall(text):
      return "Danish"
  return False

def ngram_distance(text_ngram, language, all_ngrams, n_feature=300):
  """
  ngram_distance(text_ngram, language, all_ngrams, n_feature)
  Calculate the distance between text n-gram and `language` n-gram. 

  Parameters
  ----------
  text_ngram: dictionary 
  language: str
  all_ngrams: dictionary 
    language specific n-grams
  n_feature: int 

  Returns
  -------
  int 
  """
  template_ngram = all_ngrams[language].copy()
  distance = 0
  for ngram in template_ngram:
    if ngram in text_ngram:
      distance += abs(text_ngram[ngram] - template_ngram[ngram])
    else:
      distance += n_feature 
  return distance 

def biased_min_distance(distances, margin=0.01):
  """
  biased_min_distance(distances, margin)
  Find the language with minimum distance from text n-gram.
  If the difference between English's distance and minimum distance is less than `margin`, 
    return English
  Else, 
    return language with minimum distance 
  Return UNKNOWN if there's tie between non-English languages

  Parameters 
  ----------
  distances: dictionary
  margin: float 

  Returns
  -------
  str
  """
  min_value = min(distances.values())
  min_language = min(distances, key=distances.get)
  if "English" in distances:
    eng_value = distances["English"]
    if min_value == eng_value:
      return "English"
    else:
      difference = (eng_value - min_value)/min_value
      if difference < margin:
        return "English"
      else:
        return min_language
  else:    
    if list(distances.values()).count(min_value) == 1:
      return min_language
    else:
      return "UNKNOWN"

def closest_distance(text, script, blocks, all_ngrams, n_value=3, n_feature=300, margin=0.01):
  """
  closest_distance(text, script, blocks, all_ngrams, n_value, n_feature, margin)
  Predict language of text based on n-gram distances

  Parameters
  ----------
  text: str
  script: str 
    unicode block name 
  blocks: Counter 
    unicode blocks found in text 
  all_ngrams: dictionary 
    template of n-gram for each language 
  n_value: int
  n_feature: int
  margin: float 

  Returns 
  -------
  str
  """
  if len(text) < n_value:
    return "UNKNOWN"
  else:
    text_ngram = create_ngram([text], n_value, n_feature, limit=False)
    distances = {}
    languages = SCRIPT_TO_MULTILANGUAGE[script]
    if script == "BASIC LATIN" or script == "EXTENDED LATIN":
      languages = SCRIPT_TO_MULTILANGUAGE["EXTENDED LATIN"] + SCRIPT_TO_MULTILANGUAGE["BASIC LATIN"]        
    for language in languages:
      distance = ngram_distance(text_ngram, language, all_ngrams, n_feature)
      distances[language] = distance
    return biased_min_distance(distances, margin)

def unicode_predict(data, all_ngrams, n_value, n_feature, margin=0.01):
  """
  unicode_predict(data, all_ngrams, n_value, n_feature, margin)
  Predict the language class of text assigned UNKNOWN by Naive Bayes model.

  Parameters
  ----------
  data: pandas dataframe
  all_ngrams: dictionary
  n_value: int
    n-gram
  n_feature: int
    top nth counts in n-gram
  margin: float 

  Returns
  -------
  pandas dataframe 
  """
  def unicode_rule(row, all_ngrams, n_value, n_feature, margin):
    if row["nb_predict"] == "UNKNOWN":
      blocks = count_blocks(row["text"])
      script = highest_block(blocks)
      if script in SCRIPT_TO_LANGUAGE:
        lang = SCRIPT_TO_LANGUAGE[script]
        if lang == "CJK":
          return cjk_rule(blocks)
        elif lang in SCRIPT_TO_MULTILANGUAGE:
          new_lang = unique_chars(row["text"], lang)
          if not new_lang:
            return closest_distance(row["text"], lang, blocks, all_ngrams, n_value, n_feature, margin)
          else:
            return new_lang
        else:
          return lang
      return "UNKNOWN"
    else:
      return row["nb_predict"]

  dat = data.copy()
  dat["unicode_predict"] = dat.apply(lambda row: unicode_rule(row, all_ngrams, n_value, n_feature, margin), axis=1)
  return dat 


# tune margin in unicode rule
def unicode_tune_margin(data, marginlist, all_ngrams, n_value, n_feature):
  """
  unicode_tune_margin(data, marginlist, all_ngrams, n_value, n_feature)
  Tune margin (m) in the unicode rule using 5 fold cross validation. 
  
  Parameters
  ----------
  data: pandas dataframe 
  marginlist: list
    list of margin to tune (m betweens 0 and 0.01)
  all_ngrams: dictionary 
  n_value: int 
  n_feature: int
  
  Returns 
  -------
  dict
    A dictionary with keys:
    `margin`: list of margin values 
    `train_score`: list of mean training accuracy
    `train_fnr`: list of mean training false negative rate 
    `train_fpr`: list of mean training false positive rate
    `test_score`: list of mean test accuracy
    `test_fnr`: list of mean test false negative rate 
    `test_fpr`: list of mean test false positive rate
  """
  N_FOLD = 5
  kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=1)
  kf_indices = {}
  for i, (train_index, test_index) in enumerate(kf.split(data)):
    kf_indices[i] = (train_index, test_index)

  cv_results = {"margin":[], 
                "train_score":[], 
                "train_fnr":[], 
                "train_fpr":[],
                "test_score":[], 
                "test_fnr":[], 
                "test_fpr":[]}

  n_margin = len(marginlist)
  print("Fitting", N_FOLD, "folds for each of", n_margin, "candidates, totalling", N_FOLD*n_margin, "fits")
  for m in marginlist:
    train_scores = []
    train_fnrs = []
    train_fprs = []
    test_scores = []
    test_fnrs = []
    test_fprs = []

    print("margin =", m)
    start_time = time.time()
    for fold in range(5):
      training_folds_data = data.iloc[kf_indices[fold][0],].copy()
      holdout_fold_data = data.iloc[kf_indices[fold][1],].copy()

      train_predict = unicode_predict(training_folds_data, all_ngrams, n_value, n_feature, m)
      train_score = accuracy_score(train_predict["language"], train_predict["unicode_predict"], normalize=True)*100
      train_fnr = fnr(train_predict["language"], train_predict["unicode_predict"])
      train_fpr = fpr(train_predict["language"], train_predict["unicode_predict"])
      
      test_predict = unicode_predict(holdout_fold_data, all_ngrams, n_value, n_feature, m)
      test_score = accuracy_score(test_predict["language"], test_predict["unicode_predict"], normalize=True)*100
      test_fnr = fnr(test_predict["language"], test_predict["unicode_predict"])
      test_fpr = fpr(test_predict["language"], test_predict["unicode_predict"])

      train_scores.append(train_score)
      train_fnrs.append(train_fnr)
      train_fprs.append(train_fpr)

      test_scores.append(test_score)
      test_fnrs.append(test_fnr)
      test_fprs.append(test_fpr)

    end_time = time.time()
    total_time = end_time - start_time
    mean_train_score = np.mean(train_scores)
    mean_train_fnr = np.mean(train_fnrs)
    mean_train_fpr = np.mean(train_fpr)
    mean_test_score = np.mean(test_scores)
    mean_test_fnr = np.mean(test_fnrs)
    mean_test_fpr = np.mean(test_fprs)

    cv_results["margin"] += [m]
    cv_results["train_score"] += [mean_train_score]
    cv_results["train_fnr"] += [mean_train_fnr] 
    cv_results["train_fpr"] += [mean_train_fpr] 
    cv_results["test_score"] += [mean_test_score] 
    cv_results["test_fnr"] += [mean_test_fnr]
    cv_results["test_fpr"] += [mean_test_fpr]

    print("[CV 5/5] END ..... total time:", "{:0.2f}".format(total_time), "s, train score=", "{:0.2f}".format(mean_train_score), ", test score=", "{:0.2f}".format(mean_test_score), sep="")
  return cv_results


def engCheck(text):
  try:
    text.encode(encoding='utf-8').decode('ascii')
  except UnicodeDecodeError:
    return 'Not Eng Unicode'
  else:
    return 'Eng Unicode'
    
def removechar(text):
    return ''.join(
      char for char in text if ord(char) < 128
    )