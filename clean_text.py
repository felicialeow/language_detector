import re
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer


space_regex = "\s+"
punctuation_regex = r"[^\w\d\s]"


def lowercase(data, textcol):
    """
    lowercase(data, textcol)
    Lowertext text found in `textcol`.

    Parameters
    ----------
    data: pandas dataframe
    textcol: str

    Return 
    ------
    dataframe
    """
    data[textcol] = data[textcol].apply(lambda text: text.lower())
    return data


def rm_multiplespace(data, textcol):
    """
    rm_multiplespace(data, textcol)
    Replace multiple whitespace and/or tab with single whitespace instance found in `textcol`.

    Parameters
    ----------
    data: pandas dataframe
    textcol: str

    Return 
    ------
    dataframe
    """
    data[textcol] = data[textcol].apply(lambda text: re.sub(space_regex, " ", text))
    data[textcol] = data[textcol].apply(lambda text: text.strip())
    return data 

def rm_numbers(data, textcol):
    """
    rm_numbers(data, textcol)
    Remove numbers from `textcol`.

    Parameters
    ----------
    data: pandas dataframe
    textcol: str

    Return
    ------
    dataframe 
    """
    data[textcol] = data[textcol].apply(lambda text: re.sub("\d", "", text))
    return data


def pre_tokenize(data, textcol):
    """
    pre_tokenize(data, textcol)
    Split `textcol` by whitespace and punctuation and join tokens back with whitespace.

    Parameters
    ----------
    data: pandas dataframe
    textcol: str

    Return 
    ------
    dataframe
    """
    def split_text(text):
      tokens = text.split()
      tokens = [re.split(punctuation_regex, token) for token in tokens]
      tokens = [subtoken for token in tokens for subtoken in token]
      tokens = [token.strip() for token in tokens if token]
      return tokens
    data[textcol] = data[textcol].apply(lambda text: split_text(text))
    data[textcol] = data[textcol].apply(lambda tokens: " ".join(tokens))
    return data

def tokenize(data, textcol, tokenizer):
    """
    tokenize(data, textcol, tokenizer)
    Tokenize `textcol` using `tokenizer`.

    Parameters
    ----------
    data: pandas dataframe
    textcol: str

    Return 
    ------
    dataframe
        `textcol` replaced with string of token index separated by whitespace
    """
    data[textcol] = data[textcol].apply(lambda text: tokenizer.encode(text))
    data[textcol] = data[textcol].apply(lambda tokens: " ".join(map(str, tokens)))
    return data

def vectorize(data, textcol, endidx):
    """
    vectorize(data, textcol, endidx)
    Create a count vectorizer with custom vocabulary of indices, and vectorize string of token indices. 

    Parameters
    ----------
    data: pandas dataframe
    textcol: str
    endidx: int 

    Return 
    ------
    sparse matrix
        count vector
    """
    def split_idx(text):
      return text.split()
    vectorizer = CountVectorizer(tokenizer=split_idx, 
                                vocabulary=map(str,range(endidx)))
    countvector = vectorizer.transform(data[textcol])
    return countvector
