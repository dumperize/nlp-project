from summa.summarizer import summarize


PART = 0.1

def calc_summa_score(records):
    originals = []
    predictions = []

    for i, record in records.iterrows():
        originals.append(record['summary'].lower())

        predicted_summary = summarize(record['text'].lower(), ratio=PART)
        predictions.append(predicted_summary)

    return originals, predictions
