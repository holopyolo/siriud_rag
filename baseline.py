from retriever import get_top_k
from tlight import generate

OUTPUT_PATH = "result.txt"
K = 5

user_question = "как мне загрузить рекламу в чеки, если она не печатается, и что делать с базой товаров?"
system_promt = f"Ты ассистент, который отвечает на вопрос пользователя, исходя из текстов данных тебе. Если нет ответа на вопрос в этих текстах, отказывайся отвечать.\nВопрос: {user_question}\n"

texts, url = get_top_k(question=user_question, k=K)
output = generate(prompt=texts, system_promt=system_promt)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(texts)
    f.write("\n\n\n")
    f.write(output)
    f.write("\n\n\n")
    f.write(url)
