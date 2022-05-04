import pandas as pd
import click
import json
import os

from src.models.text_rank.get_text_rank import get_text_rank
from src.models.calc_scores import calc_scores


@click.command()
@click.argument("input_file_path", type=click.Path(exists=True))
@click.argument("output_file_path")
def calc_text_rank_score(input_file_path: str, output_file_path: str):
    records = pd.read_excel(input_file_path)

    originals = []
    predictions = []

    for i, resord in records.iterrows():
        originals.append(resord['summary'])

        predicted_summary = get_text_rank(resord['text'])
        predictions.append(predicted_summary)

    scores = calc_scores(originals, predictions)
    json_obj = json.dumps(scores, indent=4)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as outfile:
        outfile.write(json_obj)


if __name__ == "__main__":
    calc_text_rank_score()
