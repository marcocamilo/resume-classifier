import pandas as pd
from src.modules.analysis import df_overview

df = pd.read_csv('./data/0-external/livecareer-resume-data.csv')

df_overview(df)

df = df[['Category', 'Resume_str']]
df.columns = ['Category', 'Resume']
df.head()

df.to_parquet('./data/1-raw/resume-data.parquet')
