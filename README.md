## Table of Contents 
- [About](#about)
- [Overview](#overview)
- [Installation](#installation)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## About <a name="about"></a>
A language detector that employs Naives Bayes model and unicode rule to classify text into 32 different languages. The model is biased towards English language, and tries to classify text as English whenever possible. If the model fails to classify the text into any of the 32 languages with confidence, the text will be classified as UNKNOWN. 
The model is part of a project that explores various Natural Language Processing methods. It is solely used for research purpose. 

## Overview <a name="overview"></a>

### Data Set Up 
The dataset employed in this project is a compilation of data retrieved from 4 different sources. The language labels have been standardised to follow the iso639 system. Any language label that have been identified as incorrect have been corrected to the best of ability. Duplicated text have been removed the dataset.  
The dataset is split into training (~79.8%) and test set (~20.2%) for the purpose of model building. 

### Text Processing 
**Text normalization**  
This includes three steps of text cleaning: 

1. Lowercase text
2. Replace multiple whitespaces/tab to a single whitespace
3. Remove numbers from text

**Tokenization**  
All punctuations in the normalized text will be replaced by whitespace.  
A tokenizer has been trained with a vocabulary size of 50000 words that split the text by unigram. This tokenizer can be accessed by [tokenizer.model](tokenizer.model).

**Vectorization**  
The list of tokens will be vectorized into a sparse matrix with number of columns = 50000. The language label will be converted into an one-hot encoding vector. 

### Modelling 
The language detector model is an ensemble of two methods:  

- Multinomial Naive Bayes 
- Unicode rule

There are three metrics used in the evaluation of the model: 

- Prediction accuracy
- False negative rate 
- False positive rate

**Multinomial Naive Bayes**  
The multinomial naive bayes model is the main method utilised in the model. The smoothing parameter, alpha, is tuned using a 5-fold cross validation (CV) method. Based on the CV results, alpha = 0.01 is determined to be the ideal value. 

The predicted language based on multinomial naive bayes has an associated probability confidence level. A threshold method is implemented to classify the language as UNKNOWN if the probability confidence level falls below threshold.  

The threshold is determined based on the length of text and an arbitrary t value (to be determined by CV):  
threshold 
= t; if length < 10  
= exp(1/length)*t; else length >= 10  

Based on CV results, the t = 0.8 has been selected.  

**Unicode Rule**  
The text that failed to be classified by multinomial naive bayes model will use unicode rule in an attempt to classify it.  
The basis of this rule is the unicode block defined by the [Unicode Consortium](https://www.unicode.org/charts/). Each character is assigned to one block, representing a script. 

Steps:  
1. Count the number of characters belonging to each unicode block 
2. Determine the block with highest count 
3. If at least 30% of the characters fall in the block, unicode rule can be applied. Else, the text will remain UNKNOWN

Rules:  
1. If the highest block = CJK, assign the language as Chinese, Korean or Japanese
2. If the highest block = ARABIC, CYRILLIC, LATIN, EXTENDED LATIN, find any unique characters associated to each language. E.g. "ё" is a character used in Russian
3. If steps 1 and 2 failed to assign a language, create a trigram for the text  
    a. Compare the trigram against the template trigrams  
    b. Find the language with minimum distance between its template and text trigram  
    c. Compare the minimum distance vs distance to English template trigram  
    d. If the distance is close (< m), assign text as English. Else, the language with minimum distance.

The margin, m, of closeness is tuned using CV; with m = 0.006 chosen based on the results. 

**Final Model**  
The final model with an ensemble of multinomial naive bayes (a = 0.01, t = 0.8) and unicode rule (m = 0.006) achieved the following results:

- Accuracy = 99.1% 
- False negative rate = 0.4% 
- False positive rate = 4%
 
## Installation <a name="installation"></a>

Create a local copy of the repository 
```
git clone https://github.com/felicialeow/language_detector.git
```

Install the required python libraries 
```
pip install -r requirements.txt
```

In the terminal, run the test.py file to see how the language_detector() function works.
```
python test.py
```

## Authors <a name="authors"></a>
- Felicia Leow
- Han Riffin

## Acknowledgements <a name="acknowledgements"></a>
Credits to the following sources for providing the dataset used in the project:  

- Basil Saji: [A Dataset for Language Detection]("https://www.kaggle.com/datasets/basilb2s/language-detection")
- Aman Kharwal: [Language Detection with Machine Learning]("https://thecleverprogrammer.com/2021/10/30/language-detection-with-machine-learning/")
- LucaPapariello: [Language Identification]("https://huggingface.co/datasets/papluca/language-identification")
- Chazzer: [Big Language Detection Dataset]("https://www.kaggle.com/datasets/chazzer/big-language-detection-dataset")

The language detector model built in this project has taken inspiration from the following projects; 

- Ritchie Ng: [Vectorization, Multinomial Naive Bayes Classifier and Evaluation]("https://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/")
- Kent37: [guess-language]("https://github.com/kent37/guess-language")