import pandas as pd
import numpy as np
import Stemmer
import bm25s
import json

with open('webhelp dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

if 'data' in data:
    df = pd.DataFrame(data['data'])

df = df[df.description.str.len() > 4]

df['text'] = df['title'] + ' ' + df['description']

texts_urls = df[['text', 'url']].drop_duplicates(subset='text', keep='first').reset_index(drop=True)

corpus = texts_urls.text.values
urls = texts_urls.url.values

stemmer = Stemmer.Stemmer("russian")
corpus_tokens = bm25s.tokenize(corpus, stemmer=stemmer, stopwords="ru")
retriever = bm25s.BM25(method="bm25+", delta=1.5)
retriever.index(corpus_tokens)


def get_top_k(question: str, k: int) -> tuple[str, str]:
    """
    Returns top-K results and a url.

    Args:
        question (str): A search question.
        k (int): The number of results.

    Returns:
        tuple[str, str]: A tuple of a string with results and a url.
    """
    questions_tokens = bm25s.tokenize(question, stemmer=stemmer, stopwords="ru")
    result_indexes, scores = retriever.retrieve(questions_tokens, k=k)

    result_indexes = result_indexes[0]
    url_index = result_indexes[np.argmax(scores[0])]

    formatted_results = [f"Текст {i+1}:\n{corpus[idx]}\n" for i, idx in enumerate(result_indexes)]
    texts = ''.join(formatted_results)

    return texts, urls[url_index]
