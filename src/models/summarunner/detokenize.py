import re
import nltk
from tensorboard import summary


def punct_detokenize(text):
    text = text.strip()
    punctuation = ",.!?:;%"
    closing_punctuation = ")]}"
    opening_punctuation = "([{"

    for ch in punctuation + closing_punctuation:
        text = text.replace(" " + ch, ch)
    for ch in opening_punctuation:
        text = text.replace(ch + " ", ch)

    res = [r'"\s[^*]+\s"', r"'\s[^*]+\s'"]
    for r in res:
        for f in re.findall(r, text, re.U):
            text = text.replace(f, f[0] + f[2:-2] + f[-1])
    text = text.replace("' s", "'s").replace(" 's", "'s")
    text = text.strip()
    return text


def postprocess(
    ref, hyp, is_multiple_ref=False, detokenize_after=False, tokenize_after=True
):
    if is_multiple_ref:
        reference_sents = ref.split(" s_s ")
        decoded_sents = hyp.split("s_s")
        hyp = [
            w.replace("<", "&lt;").replace(">", "&qt;").strip() for w in decoded_sents
        ]
        ref = [
            w.replace("<", "&lt;").replace(">", "&qt;").strip() for w in reference_sents
        ]

        hyp = " ".join(hyp)
        ref = " ".join(ref)

    ref = ref.strip()
    hyp = hyp.strip()

    if detokenize_after:
        hyp = punct_detokenize(hyp)
        ref = punct_detokenize(ref)
    if tokenize_after:
        hyp = hyp.replace("@@UNKNOWN@@", "<unk>")
        hyp = " ".join(nltk.word_tokenize(hyp))
        ref = " ".join(nltk.word_tokenize(ref))

    return ref, hyp
