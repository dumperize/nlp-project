import pandas as pd
import json
import os
import click
from fastai.text.all import *
from transformers import *
from blurr.text.data.all import *
from blurr.text.modeling.all import *
from src.models.calc_scores import calc_scores


@click.command()
@click.argument("imput_file_path", type=click.Path(exists=True))
@click.argument("output_file_path", type=click.Path())
def clac_train_bart_score(imput_file_path, output_file_path):
    df = pd.read_excel(imput_file_path)
    articles = df.head(1500)

    # Import the pretrained model
    pretrained_model_name = "facebook/bart-large-cnn"
    hf_arch, hf_config, hf_tokenizer, hf_model = get_hf_objects(
        pretrained_model_name, model_cls=BartForConditionalGeneration
    )

    # Create mini-batch and define parameters
    hf_batch_tfm = Seq2SeqBatchTokenizeTransform(
        hf_arch,
        hf_config,
        hf_tokenizer,
        hf_model,
        task="summarization",
        text_gen_kwargs={
            "max_length": 120,
            "min_length": 30,
            "do_sample": False,
            "early_stopping": True,
            "num_beams": 4,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "bad_words_ids": None,
            "bos_token_id": 0,
            "pad_token_id": 1,
            "eos_token_id": 2,
            "length_penalty": 2.0,
            "no_repeat_ngram_size": 3,
            "encoder_no_repeat_ngram_size": 0,
            "num_return_sequences": 1,
            "decoder_start_token_id": 2,
            "use_cache": True,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "return_dict_in_generate": False,
            "forced_bos_token_id": 0,
            "forced_eos_token_id": 2,
            "remove_invalid_values": False,
        },
    )

    # Prepare data for training
    blocks = (Seq2SeqTextBlock(batch_tokenize_tfm=hf_batch_tfm), noop)
    dblock = DataBlock(
        blocks=blocks,
        get_x=ColReader("text"),
        get_y=ColReader("summary"),
        splitter=RandomSplitter(),
    )
    dls = dblock.dataloaders(articles, batch_size=2)

    # Define performance metrics
    seq2seq_metrics = {
        "bleu": {"returns": "bleu"},
        "rouge": {
            "compute_kwargs": {
                "rouge_types": ["rouge1", "rouge2", "rougeL"],
                "use_stemmer": True,
            },
            "returns": ["rouge1", "rouge2", "rougeL"],
        },
        "bertscore": {
            "compute_kwargs": {"lang": "fr"},
            "returns": ["precision", "recall", "f1"],
        },
    }

    # Model
    model = BaseModelWrapper(hf_model)
    learn_cbs = [BaseModelCallback]
    fit_cbs = [Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

    # Specify training
    learn = Learner(
        dls,
        model,
        opt_func=ranger,
        loss_func=CrossEntropyLossFlat(),
        cbs=learn_cbs,
        splitter=partial(seq2seq_splitter, arch=hf_arch),
    ).to_fp16()

    # Create optimizer with default hyper-parameters
    learn.create_opt()
    learn.freeze()

    # Training
    learn.fit_one_cycle(3, lr_max=3e-5, cbs=fit_cbs)

    predictions = [
        learn.blurr_generate(text_to_generate, early_stopping=False, num_return_sequences=1)
        for text_to_generate in enumerate(articles["text"])
    ]
    originals = articles["summary"].values

    scores = calc_scores(originals, predictions)
    json_obj = json.dumps(scores, indent=4)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as outfile:
        outfile.write(json_obj)


if __name__ == "__main__":
    clac_train_bart_score()
