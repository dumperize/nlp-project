import nltk
nltk.download('punkt')


def calc_lead_n_score(records, n=3):
    origins = []
    predictions = []

    for i, record in records.iterrows():
        summary = record['summary'].lower()
        origins.append(summary)

        text = record['text'].lower()
        sentences = nltk.tokenize.sent_tokenize(text)
        if (len(sentences) == 0):
            print("sdf", text)
        prediction = " ".join(sentences[:n])
        predictions.append(prediction)

    return origins, predictions
