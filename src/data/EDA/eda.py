from collections import Counter, namedtuple
import nltk
import pymorphy2
import pandas as pd
import click
import json
import os


Stats = namedtuple(
    "Stats", "vocabulary,lemma_vocabulary,words_count,unique_words_counts")


@click.command()
@click.argument("input_file_path", type=click.Path(exists=True))
@click.argument("output_file_path")
def collect_stats(input_file_path, output_file_path, text_max_word=3000, summary_max_word=100):
    records = pd.read_excel(input_file_path)

    morph = pymorphy2.MorphAnalyzer()

    text_stats = Stats(Counter(), Counter(), list(), list())
    summary_stats = Stats(Counter(), Counter(), list(), list())

    def upgrade_record_field_stats(field, stats, max_words):
        words = nltk.word_tokenize(field)[:max_words]
        lemmas = [morph.parse(word)[0].normal_form for word in words]
        stats.vocabulary.update(words)
        stats.lemma_vocabulary.update(lemmas)
        stats.words_count.append(len(words))
        stats.unique_words_counts.append(len(set(words)))

    for i, record in records.iterrows():
        text = record['text']
        upgrade_record_field_stats(text, text_stats, text_max_word)

        summary = record['summary']
        upgrade_record_field_stats(summary, summary_stats, summary_max_word)

    json_obj = json.dumps({
        "text_stats_vocabulary": len(text_stats.vocabulary),
        "text_stats_lemma_vocabulary": len(text_stats.lemma_vocabulary),
        "summary_stats": len(summary_stats.vocabulary),
        "summary_lemma_vocabulary": len(summary_stats.lemma_vocabulary),
        "common_stats": len(set(text_stats.lemma_vocabulary.keys()) & set(summary_stats.lemma_vocabulary.keys())),
    }, indent=4)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as outfile:
        outfile.write(json_obj)


if __name__ == "__main__":
    collect_stats()
