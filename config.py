from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import transformers
import numpy as np
import preproc
import torch
import copy
import re

transformers.logging.set_verbosity_error()

TOKEN = 'TOKEN'


# Searches the FAISS index for the top-k most similar entries to a query
def search_faiss(query, index, embedder_model, k=5):
    query_embedding = embedder_model([query])
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]


# Splits a single document into chunks using a specified splitter
def split_into_chunks(doc, splitter):
    return splitter.split_text(doc)


# Splits a list of documents into chunks and keeps track of original indices
def get_split(documents, splitter):
    output_with_indices = []
    for index, doc in enumerate(documents):
        chunks = split_into_chunks(doc, splitter)
        for chunk in chunks:
            output_with_indices.append((index, chunk))
    return output_with_indices


# Reranks document chunks based on relevance scores computed with the reranker
def get_top_k_chuks_rerank(question, chunks_e, k=100, reranker=None):
    scores = torch.tensor(reranker.compute_score([[question, chnk] for chnk in chunks_e], normalize=True))
    reranged = reversed(scores.argsort())[:k]
    return reranged, scores[reranged]


# Embeds question and chunks, then retrieves top-k chunks based on similarity
def embedder_p(q, chunks, k, embeddings=None, embedder_model=None):
    res = retrieve_context_topk(embedder_model, chunks, q, k, embeddings=embeddings)
    return res, []


# Retrieves the top-k most similar contexts to a question based on embeddings
def retrieve_context_topk(embedder, texts, question, k=1, embeddings=None):
    question_embedding = embedder([question])[0]
    scores = np.dot(embeddings, question_embedding) / \
             (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding))
    top_k_indices = np.argsort(scores)[-k:]
    return [i for i in reversed(top_k_indices)]


# Initializes the embedding model and provides a callable interface
def get_embedder(model_name):
    embedder = SentenceTransformer(model_name)

    def f(texts, **args):
        if isinstance(texts, str):
            texts = [texts]
        return embedder.encode(texts, normalize_embeddings=True, **args)

    return f


# Maps ranked indices to their original indices in the dataset
def check_real(arr):
    return [indexes[el] for el in arr]


# Executes the main ranking pipeline with embedding and reranking steps
def new_pipe(inds, texts, q, k1, k2, p1, p2, embeddings, embedder_model, reranker):
    qs = [q]
    list_of_ranks = []
    for cand_q in qs:
        top_k_indices, _ = search_faiss(cand_q, preproc.index, embedder_model, k=k1)
        template = texts[top_k_indices]
        ind2, _ = p2(cand_q, template, k2, reranker)

        relev_chunks = template[ind2]
        final_inds = [np.where(texts == rr)[0][0] for rr in relev_chunks]
        list_of_ranks.append(final_inds)
    return list_of_ranks[0]


# Combines ranking, reranking, and final corpus retrieval
def call_pipe(indexes, texts, q, k1=50, k2=5, p1=embedder_p,
              p2=get_top_k_chuks_rerank, corpus=None,
              embeddings=None, embedder_model=None,
              reranker=None):
    ranked_inds = new_pipe(indexes, texts, q, k1=k1, k2=k2, p1=p1, p2=p2,
                           embeddings=embeddings,
                           embedder_model=embedder_model,
                           reranker=reranker)

    ranked = [corpus[ind] for ind in check_real(ranked_inds)]
    return ranked, [ind for ind in check_real(ranked_inds)]


# Generates text based on a prompt and model configuration
def generate(model, tokenizer, generation_config, prompt_desired):
    prompt = tokenizer.apply_chat_template([{
        "role": "system",
        "content": ""
    }, {
        "role": "user",
        "content": prompt_desired
    }], tokenize=False, add_generation_prompt=True)

    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config, max_new_tokens=1024)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return output


# Executes the entire question-answering pipeline, combining retrieval and generation
def end2end_pipe(user_question,
                 prompt_reasoning=preproc.REASONING_PROMPT,
                 prompt=preproc.RAG_PROMPT):
    texts_orig, inds_orig = call_pipe(indexes, texts, user_question,
                                      k1=50, k2=10, embeddings=embeddings,
                                      embedder_model=embedder_model,
                                      reranker=reranker,
                                      corpus=preproc.corpus)

    texts_rer = copy.deepcopy(texts_orig)
    for j in range(len(texts_rer)):
        texts_rer[j] = f'Чанк {j + 1}: {texts_rer[j]}'

    prompt_desired = prompt_reasoning.format(
        context_str='\n'.join(texts_rer),
        query_str=user_question
    )
    output_chunks_related = generate(preproc.main_model, preproc.main_tokenizer,
                                     preproc.main_generation_config,
                                     prompt_desired)

    prompt_desired_rag = prompt.format(
        context_str='\n'.join(texts_rer),
        query_str=user_question
    )
    output_rag = generate(preproc.main_model, preproc.main_tokenizer, preproc.main_generation_config,
                          prompt_desired_rag)
    return output_chunks_related, output_rag, inds_orig


# Parses relevant chunk indices based on GPT output
def parse_chunks_true(string, relev_inds):
    splitted = [text.lower() for text in string.split('.')[::-1]]
    try:
        for text in splitted:
            relevs = list(map(int, re.findall(r'\d+', text)))
            if len(relevs) > 0:
                return list(set([relev_inds[relev_by_gpt - 1] for relev_by_gpt in relevs]))
    except:
        return [relev_inds[0]]
    return [relev_inds[0]]


# Executes the pipeline and returns final generated text and associated URLs
def result(q):
    voutput_chunks_related, \
    output_rag, \
    inds_orig = end2end_pipe(q,
                             prompt_reasoning=preproc.REASONING_PROMPT,
                             prompt=preproc.RAG_PROMPT)

    parsed_urls_ids = parse_chunks_true(output_chunks_related, inds_orig)
    urls_parsed = [preproc.urls[url_id] for url_id in parsed_urls_ids]
    return output_rag, urls_parsed


# Prepares the split corpus and indexes for further processing
corpus_split = get_split(preproc.corpus, preproc.splitter_classic)
indexes, texts = np.array([f for f, _ in corpus_split]), np.array([s for _, s in corpus_split])

# Initialize the reranker and embedder models
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
embedder_model = get_embedder('deepvk/USER-bge-m3')
embeddings = embedder_model(texts, batch_size=32)

# Run a test of the end-to-end pipeline
output_chunks_related, output_rag, inds_orig = end2end_pipe(
    'Тестовый прогон',
    prompt_reasoning=preproc.REASONING_PROMPT,
    prompt=preproc.RAG_PROMPT)
