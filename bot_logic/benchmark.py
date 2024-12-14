from retriever import get_top_k
from bert_score import score
from tlight import generate
import pandas as pd
import time

df = pd.read_csv("benchmark.csv")


def benchmark(k: int) -> tuple[float, float, float]:
    """
    Evaluates the model on accuracy, BERTScore.
    Additionally, measures the average execution time for the process
    from obtaining `texts` to generating `outputs`.

    Args:
        k (int): The number of top-k retriever.

    Returns:
        tuple:
            accuracy (float): The proportion of questions where the target description is in the retrieved texts.
            avg_bert_score (float): The average BERTScore over all samples.
            avg_time (float): The average time (in seconds) for processing from `texts` to `outputs`.
    """
    system_promt = (
        "Ты ассистент, который отвечает на вопрос пользователя, исходя из текстов данных тебе. "
        "Если нет ответа на вопрос в этих текстах, отказывайся отвечать. Не генерируй маркдаун. "
        "Старайся писать короче.\nВопрос: "
    )
    bert_scores = []
    timings = []
    hits = 0

    for _, row in df.iterrows():
        user_question = row["question"]
        target_description = row["description"]
        target = row["target"]

        start_time = time.time()

        texts, url = get_top_k(question=user_question, k=k)
        output = generate(prompt=texts, system_promt=system_promt + f"{user_question}\n")

        elapsed_time = time.time() - start_time
        timings.append(elapsed_time)

        if target_description in texts:
            hits += 1

        _, _, f1 = score([output], [target], lang="ru", verbose=False)
        bert_scores.append(f1.item())

    accuracy = hits / len(df)
    avg_bert_score = sum(bert_scores) / len(bert_scores)
    avg_time = sum(timings) / len(timings)

    return accuracy, avg_bert_score, avg_time
