from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
import textwrap
from sentence_transformers import SentenceTransformer
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
import torch
impoer numpy as np
from FlagEmbedding import FlagReranker

path_corpus = ''

def split_sentences(text):
    sentence_endings = r'(?<=[.!?]) +'
    sentences = re.split(sentence_endings, text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def split_into_chunks(doc, splitter):
    return splitter.split_text(doc)

def get_splitted(documents, splitter):
    output_with_indices = []
    for index, doc in enumerate(documents):
        chunks = split_into_chunks(doc, splitter)
        for chunk in chunks:
            output_with_indices.append((index, chunk))
    return output_with_indices

def rate(indexes, texts, ind, true_label):
    if indexes[ind] == true_label:
        return 1
    return 0



def call_pipe(indexes, texts, q, k1=50, k2=5, p1=embedder_p, p2=get_top_k_chuks_rerank, llm=None, tokenizer=None, generation_config=None, embedder=None, reranker=None, corpus=None):
    ranked = new_pipe(indexes, 
                      texts, q, k1=k1, 
                      k2=k2, p1=p1, p2=p2, llm=llm, 
                      tokenizer=tokenizer, generation_config=generation_config, embedder=embedder, reranker=reranker)
    texts = [corpus[rank_i] for rank_i in check_real(ranked, indexes)]
    return texts

def get_top_k_chuks_rerank(question, chunks_e, k, reranker):
    scores = torch.tensor(reranker.compute_score([[question, chnk] for chnk in chunks_e], normalize=True) )
    reranged = reversed(scores.argsort())[:k]
    return reranged, scores[reranged] #индексы доков по убыванию релев, скоры
    
def embedder_p(q, chunks, k, embedder_model, embeddings):
    res = retrieve_context_topk(embedder_model, chunks, q, embeddings, k)
    return res, []
    
def retrieve_context_topk(embedder, texts, question, embeddings, k=1, get_indices=False):
    question_embedding = embedder([question])[0]
    scores = np.dot(embeddings, question_embedding) / \
        (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding))
    top_k_indices = np.argsort(scores)[-k:]
    if get_indices:
        return [i for i in reversed(top_k_indices)]
    return [i for i in reversed(top_k_indices)]

def get_embedder(model_name):
    embedder = SentenceTransformer(model_name)

    def f(texts, **args):
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]  # If a single string is provided, convert to list
        return embedder.encode(texts, normalize_embeddings=True, **args)
    
    return f

import re
def parse_questions(output):
    output = output.lower()
    question_pattern = r'вопрос \d+ *:.+'
    questions = re.findall(question_pattern, output)
    questions = [q[q.find(':')+1:].strip() for q in questions]
    return questions


def create_queries(llm, tokenizer, generation_config, question):
    prompt_desired = f'''На основе вопроса: {question} создай 2-3 абсолютно вопроса схожих по смыслу, которые могли бы быть заданы в том же контексте но переписаны другими словами\n
                         Формат: Вопрос 1: ...\n Вопрос 2: ...\n Вопрос 3: ...'''
    prompt = tokenizer.apply_chat_template([{
        "role": "system",
        "content": ""
    }, {
        "role": "user",
        "content": prompt_desired
    }], tokenize=False, add_generation_prompt=True)
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(llm.device) for k, v in data.items()}
    output_ids = llm.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    try:
        questions_cands = parse_questions(output)
        questions_cands[-1] = question  
    except (Exception) as e:
        print(e)
        return [question]
    # print('Вопросы сгенерированы')
    return questions_cands

def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for rank_list in ranked_lists:
        for rank, item in enumerate(rank_list):
            if item not in scores:
                scores[item] = 0
            scores[item] += 1 / (rank + 1 + k)
    # print('Ранжирование пройдено')
    # print(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def check_real(arr, indexes):
    return [indexes[el] for el in arr]

def new_pipe(inds, texts, q, k1, k2, p1, p2, llm, tokenizer, generation_config, embedder, reranker, embeddings):
    qs = create_queries(
        llm,
        tokenizer, 
        generation_config,
        q
    )
    list_of_ranks = []
    for cand_q in qs:
        indexes_f, _ = p1(cand_q, texts, k1, embedder, embeddings)
        template = texts[indexes_f]
        ind2, _ = p2(cand_q, template, k2, reranker)
        relev_chunks = template[ind2]
        final_inds = [np.where(texts==rr)[0][0] for rr in relev_chunks]
        list_of_ranks.append(final_inds)
    #потому что (ind, score)
    result_inds = [f.item() for f, _ in reciprocal_rank_fusion(list_of_ranks)]
    return result_inds