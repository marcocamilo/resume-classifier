from collections import Counter

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords

resumes = pd.read_parquet('./data/3-processed/resume-features.parquet')
vectorizer = joblib.load('./models/tfidf.gz')
le = joblib.load('./models/le-resumes.gz')

#  ────────────────────────────────────────────────────────────────────
#   RESUME
#  ────────────────────────────────────────────────────────────────────

# Wordclouds Cleaned Resumes 
# resume_wordcounts = Counter(" ".join(resumes['cleaned_resumes']).split())
# wordcloud_resumes = WordCloud(
#     background_color="white",
#     max_words=100, 
# ).generate_from_frequencies(resume_wordcounts)
# plt.imshow(wordcloud_resumes, interpolation='bilinear')
# plt.show()

# Wordcounts by Resume Category
categories = resumes["Category"].unique()
vocabulary = vectorizer.vocabulary_
resume_col = "cleaned_resumes"
stopwords_list = stopwords.words('english')

fig, axes = plt.subplots(5, 5, figsize=(22, 22))
axes = axes.flatten()
plt.rcParams.update({'font.size': 20})

for i, category in enumerate(categories):
    word_list = " ".join(resumes[resumes["Category"] == category][resume_col]).split()
    word_list = [word for word in word_list if word not in stopwords_list]
    word_counts = Counter(word_list)
    wordcloud = WordCloud(
        background_color="white", 
        width=600,
        height=600,
        max_words=100,
        stopwords=stopwords_list,
    ).generate_from_frequencies(word_counts)
    axes[i].imshow(wordcloud, interpolation="bilinear")
    axes[i].set_title(le.inverse_transform([category])[0], fontsize=16)
    axes[i].axis("off")

for ax in axes[len(categories):]:
    ax.axis("off")
    ax.set_visible(False)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
plt.title('Word Cloouds per Resume Category')
plt.savefig('./output/eda.png')
plt.show()
