from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
import faiss
import torch
import json

# Load the model name and FAISS index
main_model_name = "t-tech/T-pro-it-1.0"
index = faiss.read_index("path/to/faiss")
path_to_data = 'path/to/data'

# Define RAG (Retrieval-Augmented Generation) and reasoning prompts
RAG_PROMPT = (
    "Контекстная информация снизу. (набор документов чанков)\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Ты ассистент, который отвечает на вопрос пользователя, исходя из текстов данных тебе. "
    "Пользователь не видит тексты, которые я тебе присылаю, поэтому не упоминай их."
    "Если нет ответа на вопрос в этих текстах или вопрос пользователя совсем не относится к банкам или финансам, напиши только одну фразу: нет ответа на ваш вопрос."
    "Вопрос: {query_str}\n"
    "Твой ответ: "
)
REASONING_PROMPT = (
    "Контекстная информация снизу. (набор документов чанков)\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Используя только данную информацию контекст и вопрос пользователя, "
    "Порассуждай сперва какие из этих чанков точно содержут ответ. В конце выдели (номеры) чанков которые точно содержат ответ. "
    "Если чанков, в которых дается ответ на вопрос несколько, выбери с наименьшим номером. т.к. чанки идут по релевантности по убыванию."
    "Если вопрос подразумевает использование нескольких чанков, также выдели их. В конце дай номер релевантных чанков. Ответ на вопрос не надо, нужно чтобы ты выделил чанки\n"
    "Вопрос: {query_str}\n"
    "Твое рассуждение: "
)

# Load tokenizer for the main model
main_tokenizer = AutoTokenizer.from_pretrained(main_model_name)

# Load the main model with optimized configurations
main_model = AutoModelForCausalLM.from_pretrained(
    main_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
).eval()

# Load the generation configuration for the main model
main_generation_config = GenerationConfig.from_pretrained(main_model_name)

# Initialize a sentence splitter for chunking long texts
splitter_classic = SentenceSplitter(
    chunk_size=1000,  # Define the size of each chunk in characters
    chunk_overlap=250  # Overlap between chunks to ensure context preservation
)

# Load the dataset and prepare it
with open(path_to_data, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract the 'data' field if present in the loaded JSON
if 'data' in data:
    df = pd.DataFrame(data['data'])

# Filter out rows with very short descriptions
df = df[df.description.str.len() > 4]

# Combine 'title' and 'description' fields into a single text field
df['text'] = df['title'] + ' ' + df['description']

# Remove duplicate texts, keeping only the first occurrence
texts_urls = df[['text', 'url']].drop_duplicates(subset='text', keep='first').reset_index(drop=True)

# Extract the corpus of texts and corresponding URLs
corpus = texts_urls.text.values
urls = texts_urls.url.values
