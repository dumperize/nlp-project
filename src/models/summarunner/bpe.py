import click
import pandas as pd
import youtokentome as yttm


DEFAULT_VOCAB_SIZE = 10000


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('model_path')
@click.argument('vocab_size', default=DEFAULT_VOCAB_SIZE)
def train_bpe(data_path: str, model_path: str, vocab_size: int = DEFAULT_VOCAB_SIZE):
    records = pd.read_excel(data_path)

    temp_file_name = 'temp.txt'
    with open(temp_file_name, "w") as temp:
        for i, record in records.iterrows():
            text, summary = record['text'].lower(), record['summary'].lower()
            temp.write(text+'\n')
            temp.write(summary+'\n')
    yttm.BPE.train(data=temp_file_name,
                   vocab_size=vocab_size, model=model_path)
    test(model_path)


def test(model_path: str):
    print('-------------')
    print('Test:')
    bpe_proc = yttm.BPE(model_path)
    result = bpe_proc.encode(
        ['Corner stone seq2seq with attention (using bidirectional ltsm )'], output_type=yttm.OutputType.SUBWORD)
    print(result)


if __name__ == '__main__':
    train_bpe()
