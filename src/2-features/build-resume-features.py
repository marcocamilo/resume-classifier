import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors

df = pd.read_parquet("./data/2-interim/resume-cleaned.parquet")

#  ────────────────────────────────────────────────────────────────────
#   CATEGORY LABEL ENCODER                                             
#  ────────────────────────────────────────────────────────────────────
le = LabelEncoder()
df['Category'] = le.fit_transform(df["Category"])
df['Category'].value_counts()
joblib.dump(le, "./models/le-resumes.gz")

#  ────────────────────────────────────────────────────────────────────
#   VECTORIZATION                                                      
#  ────────────────────────────────────────────────────────────────────
# Tfidf
resumes = df['cleaned_resumes'].values

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(resumes)
tfidf_vecs = tfidf_vectorizer.transform(resumes).toarray()
joblib.dump(tfidf_vectorizer, './models/tfidf.gz')

df['tfidf_vectors'] = list(tfidf_vecs)

#  ────────────────────────────────────────────────────────────────────
#   WORD2VEC                                                           
#  ────────────────────────────────────────────────────────────────────
# sentences = df['Resume'].str.split().to_list()
# df['tokenized_resumes'] = sentences
# model = Word2Vec(sentences, vector_size=100, window=5, alpha=0.01)
#
# model.wv.save("./models/word2vec.gz")

#  ────────────────────────────────────────────────────────────────────
#   SAVE DATA                                                          
#  ────────────────────────────────────────────────────────────────────
df.to_parquet("./data/3-processed/resume-features.parquet")
