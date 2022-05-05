from rouge import Rouge
import nltk
import click
import pandas as pd
from tqdm import tqdm
import json

from src.models.summarunner.extractive import build_summary_greedy


@click.command()
@click.argument("input_file_path", type=click.Path(exists=True))
@click.argument("output_file_path", type=click.Path())
def add_info_in_records(input_file_path: str, output_file_path: str) -> None:
    rouge = Rouge()
    records = pd.read_excel(input_file_path)

    col_sentences = []
    col_greedy_summary = []
    col_greedy_summary_sentences = []

    for i, record in tqdm(records.iterrows(), total=records.shape[0]):
        text, summary = record['text'], record['summary']

        sentences = nltk.sent_tokenize(text)
        greedy_summary, greedy_summary_sentences = build_summary_greedy(
            text, summary, calc_score=lambda x, y: rouge.get_scores([x], [y], avg=True)['rouge-2']['f'])

        col_sentences.append(json.dumps(sentences))
        col_greedy_summary.append(greedy_summary)
        col_greedy_summary_sentences.append(json.dumps(list(greedy_summary_sentences)))

    records['sentences'] = col_sentences
    records['greedy_summary'] = col_greedy_summary
    records['greedy_summary_sentences'] = col_greedy_summary_sentences

    records.to_excel(output_file_path, encoding='utf-8', index=False)


if __name__ == "__main__":
    add_info_in_records()
