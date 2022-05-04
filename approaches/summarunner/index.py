from rouge import Rouge
from approaches.summarunner.extractive import build_summary_greedy

def calc_summarunner_score(records):
    originals = []
    predictions = []

    rouge = Rouge()

    for i, record in records.iterrows():
        summary = record['summary']
        text = record['text']

        predicted_summary = build_summary_greedy(
            text, 
            summary, 
            calc_score=lambda x,y: rouge.get_scores([x], [y], avg=True)['rouge-2']['f'])

        predictions.append(predicted_summary)
        originals.append(summary.lower())
    return originals, predictions