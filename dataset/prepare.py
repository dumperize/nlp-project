import pandas as pd

stop_list = [43148,  9985,   196,  1156, 42819,  1323,  2028, 42461, 42548,
       23943, 37985, 26265,  2273, 42546,  1677,  2147, 39095, 15814,
        2611,   117,  1464,   115, 82299,   729,   109,   179, 70428,
         364,   995,    99, 42464,    81,   988,  2684,  2366]

df = pd.read_excel("dataset/TED_Talk.xlsx")
df = df[df['transcript'].notna()]
df = df[df['transcript'].apply(lambda x: len(x.strip()) > 0)]
df = df[df['talk__description'].apply(lambda x: len(x.strip()) > 0)]
df = df.rename(columns={'transcript':'text',"talk__description": "summary", 'talk__name': "name", "talk__id": "id"})
df = df[['id', "name", "summary", 'text']]
df = df.drop(df[df['id'].isin(stop_list)].index, axis=0)

df.to_excel('dataset/data.xlsx', encoding='utf-8', index=False)