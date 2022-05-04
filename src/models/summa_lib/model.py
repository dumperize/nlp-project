import click
import pandas as pd
import json
import os

from summa.summarizer import summarize
from src.models.calc_scores import calc_scores


PART = 0.1


@click.command()
@click.argument("input_file_path", type=click.Path(exists=True))
@click.argument("output_file_path")
def calc_summa_score(input_file_path: str, output_file_path: str):
    records = pd.read_excel(input_file_path)

    originals = []
    predictions = []

    for i, record in records.iterrows():
        originals.append(record['summary'])

        predicted_summary = summarize(record['text'], ratio=PART)
        predictions.append(predicted_summary)

    scores = calc_scores(originals, predictions)
    json_obj = json.dumps(scores, indent=4)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as outfile:
        outfile.write(json_obj)


if __name__ == "__main__":
    calc_summa_score()
