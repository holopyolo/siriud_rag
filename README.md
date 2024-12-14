# b2b rag что успели почистить


## Установка
Инструкции по установке необходимых зависимостей и запуску проекта.

```bash
# Клонируйте репозиторий
git clone https://github.com/ваш_проект.git

# Перейдите в директорию проекта
cd ваш_проект

# Установите зависимости
pip install -r requirements.txt
```

## Логика
Описание логики работы проекта и его компонентов.

### Структура проекта
```plaintext
bot_logic/
    bot.py
    config.py
    preproc.py
data/
    benchmark.csv
    benchmark100.csv
    formatted.csv
jsones/
    ideal_prompt_100_t_lite.json
    ideal_prompt_100_t_pro_8bit.json
    ideal_prompt_100_vikhr (1).json
    ideal_prompt_t_pro_8bit_contextual.json
    ragas_t_lite_100.json
    ragas_t_pro_8bit_100.json
    ragas_vikhr_100.json
    t-pro_contextual_ragas.json
legacy/
    baseline.py
    benchmark.py
    nemo.py
    retriever_pipe.py
    retriever.py
    tlight.py
notebooks/
    One_of_the_work.ipynb
    ...
README.md
requirements.txt
```

### часть файлов, что осталось и успели почистить
- `bot_logic/bot.py`: Основной файл логики бота. Запускать
- `bot_logic/config.py`: Конфигурационный файл.
- `bot_logic/preproc.py`: Файл для предварительной обработки данных.
- `data/`: Директория с данными.
- `jsones/`: Директория с JSON файлами.
- `legacy/`: Директория с устаревшими скриптами.
- `notebooks/`: Директория с Jupyter ноутбуками, подсчет метрики и исследование
- `README.md`: Этот файл.
- `requirements.txt`: Файл с зависимостями проекта.

## Использование
Запускать bot.py

## Вклад
Вложили много токенов 