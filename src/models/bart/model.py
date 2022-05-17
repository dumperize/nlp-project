import torch
import pandas as pd
import json
import os
import click
from transformers import BartTokenizer, BartForConditionalGeneration

from src.models.calc_scores import calc_scores


def bart_summarize(text, tokenizer, model, device):

    text = text.replace("\n", "")
    text_input_ids = tokenizer.batch_encode_plus(
        [text], return_tensors="pt", max_length=1024
    )["input_ids"].to(device)
    summary_ids = model.generate(
        text_input_ids,
        num_beams=4,
        length_penalty=2.0,
        max_length=400,
        min_length=120,
        no_repeat_ngram_size=3,
    )
    summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary_txt

@click.command()
@click.argument("imput_file_path", type=click.Path(exists=True))
@click.argument("output_file_path", type=click.Path())
def calc_bart_without_train_score(imput_file_path, output_file_path):
    bart = torch.hub.load("pytorch/fairseq", "bart.large")
    bart.eval()  # disable dropout (or leave in train mode to finetune)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_excel(imput_file_path)

    originals = []
    predictions = []

    for i, record in df.iterrows():
        print(i)
        summary = bart_summarize(record['text'], tokenizer, model, device)
        originals.append(record['summary'])
        predictions.append(summary)

    scores = calc_scores(originals, predictions)
    json_obj = json.dumps(scores, indent=4)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as outfile:
            outfile.write(json_obj)


if __name__ == "__main__":
    calc_bart_without_train_score()