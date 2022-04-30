from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge


def calc_scores(originals, prediction):
    blue_score = corpus_bleu([[x] for x in originals], prediction)
    rouge = Rouge()
    rouge_score = rouge.get_scores(prediction, originals, avg=True)

    print(f"BLUE: {blue_score}")
    print(f"ROUGE - 1: {rouge_score['rouge-1']}")
    print(f"ROUGE - 2: {rouge_score['rouge-2']}")
    print(f"ROUGE - L: {rouge_score['rouge-l']}")
