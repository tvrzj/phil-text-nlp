import pandas as pd
import stanza

f = open("./zar-final.txt", "r")
zarathustra = f.read()
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')

def batch_process(text, nlp, batch_size=50):
    paragraphs = text.split('\n\n')
    batches = [paragraphs[i:i + batch_size] for i in range(0, len(paragraphs), batch_size)]

    all_lemmas = []

    for batch in batches:
        batch_text = '\n\n'.join(batch)
        doc = nlp(batch_text)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.lemma is not None:
                    all_lemmas.append(word.to_dict())

    return all_lemmas

if __name__ == "__main__":
    lemmas = batch_process(zarathustra, nlp)
    lemmas_df = pd.DataFrame(lemmas)
    lemmas_df.to_json('../outputs/zarathustra-lemmas.json')