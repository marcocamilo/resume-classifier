import pandas as pd
from src.modules.nlp import preprocessing
from tqdm import tqdm

df = pd.read_parquet("./data/1-raw/postings_sample.parquet", columns=['description'])

sample_posting = df["description"].sample(1).item()
display(sample_posting)

#  ────────────────────────────────────────────────────────────────────
#   TEXT CLEANING                                                      
#  ────────────────────────────────────────────────────────────────────

## Posting common words
posting_stopwords = [
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "experience", "skills", "work", "job",
    "company", "position", "role", "team", "ability", "management", "time", "including",
    "business", "requirements", "required", "benefits", "support", "description"
]

## Function testing
lst_regex = [r"\b[a-zA-Z]\b"]
preprocessing(sample_posting, exist=True, 
              lst_regex=lst_regex, lst_stopwords=posting_stopwords)

## Resume Cleaning
tqdm.pandas(desc='Text Cleaning')
df["cleaned_descriptions"] = df['description'].progress_apply(preprocessing, lst_regex=lst_regex, 
                                                     exist=True, lst_stopwords=posting_stopwords)

## Preview results
display(df.sample(15)[["description", "cleaned_descriptions"]])

#  ────────────────────────────────────────────────────────────────────
#   EXPORT DATA                                                        
#  ────────────────────────────────────────────────────────────────────
df.to_parquet("./data/2-interim/postings-cleaned.parquet")
