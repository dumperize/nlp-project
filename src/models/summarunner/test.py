import torch
import click
import json
import os
import pandas as pd
import youtokentome as yttm

from src.models.summarunner.detokenize import postprocess
from src.models.calc_scores import calc_scores
from src.models.summarunner.iterator import BatchIterator
from src.models.summarunner.model import SentenceTaggerRNN
from src.models.summarunner.utils import json_convert



@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("test_file_path", type=click.Path(exists=True))
@click.argument("bpe_file_path", type=click.Path(exists=True))
@click.argument("output_file_path")
def calc_summarunner_score(
    model_path, test_file_path, bpe_file_path, output_file_path, top_k=3
):
    test_records = pd.read_excel(test_file_path)
    test_records["sentences"] = test_records["sentences"].apply(json_convert)
    test_records["greedy_summary_sentences"] = test_records[ "greedy_summary_sentences" ].apply(json_convert)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bpe_processor = yttm.BPE(bpe_file_path)
    vocabulary = bpe_processor.vocab()
    test_iterator = BatchIterator(test_records, vocabulary, 32, bpe_processor, device=device)

    origins = []
    predictions = []

    model = SentenceTaggerRNN(len(vocabulary))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for batch in test_iterator:
        logits = model(batch["inputs"])
        sum_in = torch.argsort(logits, dim=1)[:, -top_k:]

        for i in range(len(batch["records"])):
            summary = batch["records"][i]["summary"]
            pred_summary = " ".join(
                    batch["records"][i]["sentences"][ind]
                    for ind in sum_in.sort(dim=1)[0][i]
                )

            summary, pred_summary = postprocess(summary, pred_summary)

            origins.append(summary)
            predictions.append(pred_summary)

    scores = calc_scores(origins, predictions)
    json_obj = json.dumps(scores, indent=4)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as outfile:
        outfile.write(json_obj)


if __name__ == "__main__":
    calc_summarunner_score()
