import pandas as pd


df = pd.read_excel("dataset/TED_Talk.xlsx")
df = df[df['transcript'].notna()]
df = df[df['transcript'].apply(lambda x: len(x.strip()) > 0)]
df = df[df['talk__description'].apply(lambda x: len(x.strip()) > 0)]
df = df.rename(columns={'transcript':'text',"talk__description": "summary", 'talk__name': "name", "talk__id": "id"})
df = df[['id', "name", "summary", 'text']]

df.to_excel('dataset/data.xlsx', encoding='utf-8', index=False)