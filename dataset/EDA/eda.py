from collections import Counter, namedtuple
import razdel
import pymorphy2
import pandas as pd


Stats = namedtuple("Stats", "vocabulary,lemma_vocabulary,words_count,unique_words_counts")

def collect_stats(records, lower=True, text_max_word =3000,summary_max_word=100):
    morph = pymorphy2.MorphAnalyzer()

    text_stats = Stats(Counter(), Counter(), list(), list())
    summary_stats = Stats(Counter(), Counter(), list(), list())

    def upgrade_record_field_stats(field, stats,max_words):
        words = [word.text for word in razdel.tokenize(field)][:max_words]
        lemmas = [morph.parse(word)[0].normal_form for word in words]
        stats.vocabulary.update(words)
        stats.lemma_vocabulary.update(lemmas)
        stats.words_count.append(len(words))
        stats.unique_words_counts.append(len(set(words)))

    for i, record in records.iterrows():
            text= record['text']
            text= text if not lower else text.lower()
            upgrade_record_field_stats(text,text_stats,text_max_word)

            summary= record['summary']
            summary= summary if not lower else summary.lower()
            summary_words = [word.text for word in razdel.tokenize(summary)]
            upgrade_record_field_stats(summary,summary_stats,summary_max_word)
    return text_stats, summary_stats

df = pd.read_excel("dataset/TED_Talk.xlsx")
df = df[df['transcript'].notna()]
df = df.rename(columns={'transcript':'text',"talk__description": "summary"})
# print(df)
text_stats, summary_stats = collect_stats(df)
print(f"text vocabulary size: {len(text_stats.vocabulary)}")
print(f"text lemma vocabulary size: {len(text_stats.lemma_vocabulary)}")
print(f"summary vocabulary size: {len(summary_stats.vocabulary)}")
print(f"summary lemma vocabulary size: {len(summary_stats.lemma_vocabulary)}")
print(f"common lemmas summary vs text: {len(set(text_stats.lemma_vocabulary.keys()) & set(summary_stats.lemma_vocabulary.keys()))}")


