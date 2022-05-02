import copy
import nltk


def build_summary_greedy(text, original_summary, calc_score):
    original_summary = original_summary.lower()

    text = text.lower()
    sentences = nltk.tokenize.sent_tokenize(text)
    n_sentences = len(sentences)
    
    summary_sentences = set()


    score = -1.0
    summaries = []
    for _ in range(n_sentences):
        for i in range(n_sentences):
            if i in summary_sentences: 
                continue

            current_summary_sentences = copy.copy(summary_sentences)
            current_summary_sentences.add(i)
            current_summary_sentences_sort = sorted(list(current_summary_sentences))
            current_summary = " ".join([sentences[index] for index in  current_summary_sentences_sort])
            
            try:
                current_score = calc_score(current_summary, original_summary)
            except:
                print(current_summary)
                print(i)
                print(text)
                print(sentences)
                print(current_summary_sentences_sort)
            else:
                current_score = -1.0
            summaries.append([current_score, current_summary_sentences])

        best_score, best_summary_sentences = max(summaries)
        if best_score <= score:
            break
        summary_sentences = best_summary_sentences
        score = best_score

    summary_sentences_sort = sorted(list(summary_sentences))
    summary = " ".join([sentences[index] for index in  summary_sentences_sort])

    return summary, summary_sentences



    