from benchmark import benchmark

OUTPUT_PATH = "result.txt"
K = 5

accuracy, avg_bert_score, avg_time = benchmark(k=K)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Accuracy of BM25: {accuracy:.2%}")
    f.write("\n\n")
    f.write(f"BERTScore of LLM: {avg_bert_score:.2%}")
    f.write("\n\n")
    f.write(f"avg_time: {avg_time}")
    f.write("\n\n")


# user_question = "как мне загрузить рекламу в чеки, если она не печатается, и что делать с базой товаров?"
# system_promt = f"Ты ассистент, который отвечает на вопрос пользователя, исходя из текстов данных тебе. Если нет ответа на вопрос в этих текстах, отказывайся отвечать.\nВопрос: {user_question}\n"
#
# texts, url = get_top_k(question=user_question, k=K)
# output = generate(prompt=texts, system_promt=system_promt)
#
# with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
#     f.write(texts)
#     f.write("\n\n\n")
#     f.write(output)
#     f.write("\n\n\n")
#     f.write(url)
