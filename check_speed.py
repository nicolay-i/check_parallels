# Цель в проверке скорости обработки запросов при заданном параллелизме
# т.е. какое время занимает обработка 1, 2, 4, 8, 16, 32 однинаковых запросов к LLM в режиме параллельности

import time
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

from openAi import ask_ai
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess

user_query = "Как улучшить качество ответов нейросетей?"

system_prompt_keywoards = """
Ты — ИИ-ассистент, специализирующийся на анализе текстов и генерации ключевых слов для поисковых систем. Твоя задача — на основе запроса пользователя сгенерировать три списка ключевых слов/фраз, отражающих разную степень релевантности к запросу.

**Входные данные:** Текстовый запрос от пользователя.

**Задача:**
1.  Проанализируй семантику и основные сущности в запросе пользователя.
2.  Сгенерируй три списка ключевых слов/фраз (каждый по 5-10 элементов):
    *   **Уровень 1 (Высокая релевантность):** Слова/фразы, наиболее точно и полно отражающие суть запроса. Это могут быть прямые синонимы, основные термины, конкретные технологии или проблемы, упомянутые в запросе.
    *   **Уровень 2 (Средняя релевантность):** Слова/фразы, семантически связанные с запросом или представляющие его важные аспекты, контекст, смежные области.
    *   **Уровень 3 (Низкая релевантность):** Слова/фразы, представляющие более широкие категории, косвенно связанные темы или альтернативные подходы.
3.  Старайся генерировать не только отдельные слова, но и значимые словосочетания (n-граммы), где это уместно.
4.  Можешь использовать синонимы и связанные понятия для расширения списка.

**Формат вывода:** Предоставь результат в простом текстовом формате, четко разделяя уровни. Каждый уровень должен начинаться с метки "Уровень X:" и содержать список ключевых слов/фраз, разделенными точкой с запятой.

**Пример формата вывода:** 1

Уровень 1 (высокая релевантность, наиболее точные совпадения): слово1; фраза 1; ...
Уровень 2 (средняя релевантность, семантически связанные понятия): слово2; фраза 2; ...
Уровень 3 (низкая релевантность, более широкие категории): слово3; фраза 3; ...

**Ограничения:**
*   Каждый список должен содержать от 5 до 10 уникальных элементов.
*   Ключевые слова/фразы должны быть на русском языке.
*   Не включай в списки сам исходный запрос пользователя целиком.
"""

user_prompt_keywoards = """
Сгенерируй список ключевых слов для следующего запроса:

**Запрос:** 

"""

# загрузка переменных окружения
load_dotenv()

async def process_query(query):
    """Отправка запроса к API OpenAI"""
    try:
        answer, tokens_answer, model_answer = await ask_ai(system_prompt=system_prompt_keywoards, prompt=user_prompt_keywoards + query)
        return answer, tokens_answer
    except Exception as e:
        print(f"Ошибка при запросе к API: {e}")
        return None

async def run_batch(batch_size):
    """Запуск пакета запросов заданного размера"""
    start_time = time.time()
    tasks = [process_query(user_query) for _ in range(batch_size)]
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Вычисление временных метрик и добавление расчета токенов
    total_time = end_time - start_time
    avg_time = total_time / batch_size if batch_size > 0 else 0
    total_tokens = sum(tokens for result in results if result is not None for tokens in [result[1]])  # Суммируем tokens_answer, игнорируя ошибки
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    return {
        "batch_size": batch_size,
        "total_time": total_time,
        "avg_time": avg_time,
        "results": results,
        "tokens_per_second": tokens_per_second,
        "total_tokens": total_tokens
    }

async def main(parallels:int=8, max_requests:int=10):
    # запуск ollama в отдельном процессе с параметром OLLAMA_NUM_PARALLEL
    ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, env={**os.environ, "OLLAMA_NUM_PARALLEL": str(parallels)})
    
    start_time = time.time()

    batch_sizes = []
    current_size = 1
    while current_size <= max_requests:
        batch_sizes.append(current_size)
        if current_size < 16:
            current_size *= 2
        else:
            current_size += 8

    results = []
    
    for size in tqdm(batch_sizes, desc="Обработка пакетов"):
        print(f"\nЗапуск пакета размером {size}...")
        result = await run_batch(size)
        results.append(result)
        print(f"Завершено за {result['total_time']:.2f} секунд")
        await asyncio.sleep(2)
    
    # Обновление DataFrame для включения токенов в секунду и суммы токенов
    df = pd.DataFrame([(r["batch_size"], r["total_time"], r["avg_time"], r["tokens_per_second"], r["total_tokens"]) 
                       for r in results],
                      columns=["Размер пакета", "Общее время (с)", "Среднее время на запрос (с)", "Токены в секунду", "Сумма токенов"])
    
    # Сохранение в CSV
    df.to_csv(f"results/results_{parallels}.csv", index=False)
    
    # Сохранение журнала из процесса ollama 
    try:
        log_output = ollama_process.stdout.read1().decode("utf-8", errors="replace")
        with open(f"results/ollama_log_{parallels}.txt", "w", encoding="utf-8") as f:
            f.write(log_output)
    except Exception as e:
        print(f"Ошибка при сохранении лога: {e}")
    
    # Завершение процесса ollama
    ollama_process.terminate()
    ollama_process.wait(timeout=5)
    
    
    
    # Построение графика с добавлением нового подграфика
    plt.figure(figsize=(12, 9))  # Увеличим размер для дополнительного подграфика
    
    plt.subplot(2, 2, 1)
    plt.plot(df["Размер пакета"], df["Общее время (с)"], marker='o')
    plt.title("Общее время обработки")
    plt.xlabel("Размер пакета")
    plt.ylabel("Время (с)")
    
    plt.subplot(2, 2, 2)
    plt.plot(df["Размер пакета"], df["Среднее время на запрос (с)"], marker='o')
    plt.title("Среднее время на запрос")
    plt.xlabel("Размер пакета")
    plt.ylabel("Время (с)")
    
    plt.subplot(2, 2, 3)
    plt.plot(df["Размер пакета"], df["Токены в секунду"], marker='o')
    plt.title("Токены в секунду")
    plt.xlabel("Размер пакета")
    plt.ylabel("Токены/с")
    
    plt.subplot(2, 2, 4)
    plt.plot(df["Размер пакета"], df["Сумма токенов"], marker='o')
    plt.title("Сумма токенов по размеру батча")
    plt.xlabel("Размер пакета")
    plt.ylabel("Сумма токенов")
    
    plt.tight_layout()
    plt.savefig(f"results/performance_metrics_{parallels}.png")
    print(f"Результаты сохранены в results/results_{parallels}.csv и results/performance_metrics_{parallels}.png")
    end_time = time.time()
    print(f"Завершено за {(end_time - start_time):.2f} секунд")

if __name__ == "__main__":
    start_time = time.time()
    print("Запуск проверки скорости для 10 параллельных запросов (до 30 запросов)")
    asyncio.run(main(parallels=10, max_requests=30))
    end_time = time.time()
    print(f"Завершено за {(end_time - start_time):.2f} секунд")
    start_time = time.time()
    print("Запуск проверки скорости для 20 параллельных запросов (до 50 запросов)")
    asyncio.run(main(parallels=20, max_requests=50))
    end_time = time.time()
    print(f"Завершено за {(end_time - start_time):.2f} секунд")
    start_time = time.time()
    print("Запуск проверки скорости для 40 параллельных запросов (до 70 запросов)")
    asyncio.run(main(parallels=40, max_requests=70))
    end_time = time.time()
    print(f"Завершено за {(end_time - start_time):.2f} секунд")
    start_time = time.time()
    print("Запуск проверки скорости для 80 параллельных запросов (до 90 запросов)")
    asyncio.run(main(parallels=80, max_requests=90))
    end_time = time.time()
    print(f"Завершено за {(end_time - start_time):.2f} секунд")

