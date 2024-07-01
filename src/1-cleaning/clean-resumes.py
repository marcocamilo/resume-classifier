import pandas as pd
from sklearn.utils import resample
from src.modules.nlp import preprocessing
from tqdm import tqdm

df = pd.read_parquet("./data/1-raw/resume-data.parquet")

#  ────────────────────────────────────────────────────────────────────
#   VARIABLE EXPLORATION                                               
#  ────────────────────────────────────────────────────────────────────

## Category
df["Category"].value_counts()

## Resume
sample_resume = df["Resume"][0]
# sample_resume = df["Resume"].sample(1).item()
# display(sample_resume)

#  ────────────────────────────────────────────────────────────────────
#   TEXT CLEANING                                                      
#  ────────────────────────────────────────────────────────────────────

## Resume common words
resume_stopwords = [
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "company", "address", "email", "phone",
    "number", "date", "year", "years", "experience", "education", "skill", "skills", "summary", 
    "objective", "responsibility", "responsibilities", "accomplishment", "accomplishments", 
    "project", "projects", "qualification", "qualifications", "certification", "certifications",
    "state", "city", "university", "college", "institute", "school", "name", "management"
]

## Function testing
lst_regex = [r"\b[a-zA-Z]\b"]
preprocessing(sample_resume, exist=True, 
              lst_regex=lst_regex, lst_stopwords=resume_stopwords)

## Resume Cleaning
tqdm.pandas(desc='Text Cleaning')
df["cleaned_resumes"] = df['Resume'].progress_apply(preprocessing, lst_regex=lst_regex, exist=True, lst_stopwords=resume_stopwords)

## Preview results
display(df.sample(5)[["Resume", "cleaned_resumes"]])

#  ────────────────────────────────────────────────────────────────────
#   SAVE UNIQUE RESUMES (BEFORE RESAMPLING)                            
#  ────────────────────────────────────────────────────────────────────
df.to_parquet("./data/2-interim/resume-cleaned-unique.parquet")

#  ────────────────────────────────────────────────────────────────────
#   DATA BALANCING                                                     
#  ────────────────────────────────────────────────────────────────────
max_samples = df['Category'].value_counts().max()

balanced_df = pd.DataFrame()
for category in df['Category'].unique():
    class_df = df[df['Category'] == category]
    upsampled_class_df = resample(class_df, replace=True, n_samples=max_samples, random_state=42)
    balanced_df = pd.concat([balanced_df, upsampled_class_df])
df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

df['Category'].value_counts()

#  ────────────────────────────────────────────────────────────────────
#   EXPORT DATA                                                        
#  ────────────────────────────────────────────────────────────────────
df.to_parquet("./data/2-interim/resume-cleaned.parquet")
