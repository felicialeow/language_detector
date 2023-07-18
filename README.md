## Table of Contents 
- [About](#about)
- [Getting Started](#getting_started)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## About <a name="about"></a>
A language detector that employs Naives Bayes model and unicode rule to classify text into 32 different languages. The model is biased towards English language, and tries to classify text as English whenever possible. If the model fails to classify the text into any of the 32 languages with confidence, the text will be classified as UNKNOWN. 
The model is part of a project that explores various Natural Language Processing methods. It is solely used for research purpose. 

## Getting Started <a name="getting_started"></a>
Follow the instructions to use the model for your own purpose. 

### Prerequisites 
Create a local copy of the repository 
```
git clone https://github.com/felicialeow/language_detector.git
```

### Installing 
Install the required python libraries 
```
pip install -r requirements.txt
```

### Test 
In the terminal, run the test.py file to see how the language_detector() function works.
```
python test.py
```

## Authors <a name="authors"></a>
- Felicia Leow
- Han Riffin

## Acknowledgements <a name="acknowledgements"></a>
- Dataset sources
- Model inspiration
