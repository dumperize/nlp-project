import nltk
import click
import pandas as pd
import json
import os

from src.models.calc_scores import calc_scores


@click.command()
@click.argument("input_file_path", type=click.Path(exists=True))
@click.argument("output_file_path")
def calc_lead_n_score(input_file_path: str, output_file_path: str, n: int = 3):
    records = pd.read_excel(input_file_path)

    origins = []
    predictions = []

    for i, record in records.iterrows():
        summary, text = record['summary'], record['text']

        origins.append(summary)
        sentences = nltk.tokenize.sent_tokenize(text)

        prediction = " ".join(sentences[:n])
        predictions.append(prediction)

    scores = calc_scores(origins, predictions)
    json_obj = json.dumps(scores, indent=4)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as outfile:
        outfile.write(json_obj)


if __name__ == "__main__":
    calc_lead_n_score()
