import re
from bs4 import BeautifulSoup
from unidecode import unidecode
import contractions
from nltk.corpus import stopwords, words
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

def preprocessing(
    text, 
    tokenize=False,
    stem=False,
    lem=False,
    html=False,
    exist=False,
    remove_emails=True,
    remove_urls=True,
    remove_digits=True,
    remove_punct=True,
    expand_contractions=True,
    remove_special_chars=True,
    remove_stopwords=True,
    lst_stopwords=None,
    lst_regex=None
) -> str | list[str]:
    """
    Custom text preprocessing function.

    Parameters:
    - text (str): The input text to preprocess.
    - stem (bool): Apply stemming if True.
    - lem (bool): Apply lemmatization if True.
    - html (bool): Decode HTML content if True.
    - remove_emails (bool): Remove email addresses if True.
    - remove_urls (bool): Remove URLs if True.
    - remove_digits (bool): Remove digits if True.
    - remove_punct (bool): Remove punctuation if True.
    - expand_contractions (bool): Expand contractions if True.
    - remove_special_chars (bool): Remove special characters if True.
    - remove_stopwords (bool): Remove stopwords if True.
    - lst_stopwords (list): Custom list of stopwords (default: NLTK stopwords).

    Returns:
    - str: The preprocessed text.
    """
    
    # Lowercase conversion
    cleaned_text = text.lower()

    # HTML decoding
    if html:
        soup = BeautifulSoup(cleaned_text, "html.parser")
        cleaned_text = soup.get_text()
    
    # Remove Emails
    if remove_emails:
        cleaned_text = re.sub(r"([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)", " ", cleaned_text)
    
    # URL removal
    if remove_urls:
        cleaned_text = re.sub(r"(http|https|ftp|ssh)://[\w_-]+(?:\.[\w_-]+)+[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-]?", " ", cleaned_text)
    
    # Remove escape sequences and special characters
    if remove_special_chars:
        cleaned_text = re.sub(r"[^\x00-\x7f]", " ", cleaned_text)
        cleaned_text = unidecode(cleaned_text)
    
    # Remove multiple characters
    cleaned_text = re.sub(r"(.)\1{3,}", r"\1", cleaned_text)
    
    # Expand contractions
    if expand_contractions:
        cleaned_text = contractions.fix(cleaned_text)
        cleaned_text = re.sub("'(?=[Ss])", "", cleaned_text)
    
    # Remove digits
    if remove_digits:
        cleaned_text = re.sub(r"\d", " ", cleaned_text)
    
    # Punctuation removal
    if remove_punct:
        cleaned_text = re.sub("[!\"#$%&\\'()*+\,-./:;<=>?@\[\]\^_`{|}~]", " ", cleaned_text)
    
    # Line break and tab removal
    cleaned_text = re.sub(r"[\n\t]", " ", cleaned_text)
    
    # Excessive spacing removal
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    # Regex (in case, before cleaning)
    if lst_regex: 
        for regex in lst_regex:
            compiled_regex = re.compile(regex)
            cleaned_text = re.sub(compiled_regex, '', cleaned_text)

    # Tokenization (if tokenization, stemming, lemmatization or custom stopwords is required)
    if stem or lem or remove_stopwords or tokenize:
        if isinstance(cleaned_text, str):
            cleaned_text = cleaned_text.split()
        
        # Remove stopwords
        if remove_stopwords:
            if lst_stopwords is None:
                lst_stopwords = set(stopwords.words('english'))
            cleaned_text = [word for word in cleaned_text if word not in lst_stopwords]

        # Remove non-existent words
        if exist:
            english_words = set(words.words())
            cleaned_text = [word for word in cleaned_text if word in english_words]

        # Stemming
        if stem:
            stemmer = PorterStemmer()
            cleaned_text = [stemmer.stem(word) for word in cleaned_text]

        # Lemmatization
        if lem:
            lemmatizer = WordNetLemmatizer()
            cleaned_text = [lemmatizer.lemmatize(word) for word in cleaned_text]
        
        if not tokenize:
            cleaned_text = ' '.join(cleaned_text)

    return cleaned_text
