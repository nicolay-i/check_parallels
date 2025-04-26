# Цель в проверке скорости обработки запросов при заданном параллелизме
# т.е. какое время занимает обработка 1, 2, 4, 8, 16, 32 однинаковых запросов к LLM в режиме параллельности

import time
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

from generate_report import generate_report
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

async def process_query(query, index):
    """Отправка запроса к API OpenAI"""
    try:
        answer, tokens_answer, model_answer = await ask_ai(system_prompt=system_prompt_keywoards, prompt=user_prompt_keywoards + query, timeout=6000)
        
        print(f"Получил ответ {index} запроса ({tokens_answer} токенов)")
        
        return answer, tokens_answer
    except Exception as e:
        print(f"Ошибка при запросе к API: {e}")
        return None

async def run_batch(num_parallels):
    """Запуск пакета запросов заданного размера"""
    start_time = time.time()
    tasks = [process_query(user_query, i) for i in range(num_parallels)]
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Вычисление временных метрик и добавление расчета токенов
    total_time = end_time - start_time
    avg_time = total_time / num_parallels if num_parallels > 0 else 0
    total_tokens = sum(tokens for result in results if result is not None for tokens in [result[1]])  # Суммируем tokens_answer, игнорируя ошибки
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    return {
        "num_parallels": num_parallels,
        "total_time": total_time,
        "avg_time": avg_time,
        "results": results,
        "tokens_per_second": tokens_per_second,
        "total_tokens": total_tokens
    }

async def main(parallels:int=8, max_requests:int=10):
    # запуск ollama в отдельном процессе с параметром OLLAMA_NUM_PARALLEL
    ollama_path = os.getenv("OLLAMA_PATH")

    # Создаем минимальное окружение, чтобы не передавать лишнее из venv
    clean_env = os.environ.copy() 
    # Можно даже попробовать совсем чистое окружение, если предыдущее не сработает:
    # clean_env = {} 
    clean_env["OLLAMA_NUM_PARALLEL"] = str(parallels)
    # Возможно, потребуется добавить системный PATH, если ollama его ищет
    # clean_env["PATH"] = os.environ.get("SystemRoot", "") + r"\\System32;" + os.environ.get("PATH", "") # Пример для Windows

    # Запускаем ollama в новом окне консоли
    ollama_process = subprocess.Popen([ollama_path, "serve"], 
                                      env=clean_env, 
                                      creationflags=subprocess.CREATE_NEW_CONSOLE,
                                      shell=False)    
    
    print(f"Запущен ollama serve с OLLAMA_NUM_PARALLEL={parallels} в отдельном окне. Ожидание 7 секунд...")
    time.sleep(7) # Даем время на запуск сервера
    
    start_time = time.time()

    num_parallels = []
    current_size = 1
    while current_size <= max_requests:
        num_parallels.append(current_size)
        if current_size < 16:
            current_size *= 2
        else:
            current_size += 8

    results = []
    
    for size in tqdm(num_parallels, desc="Обработка пакетов"):
        print(f"\nЗапуск пакета размером {size}...")
        result = await run_batch(size)
        results.append(result)
        print(f"Завершено за {result['total_time']:.2f} секунд")
        await asyncio.sleep(2)
    
    # Обновление DataFrame для включения токенов в секунду и суммы токенов
    df = pd.DataFrame([(r["num_parallels"], r["total_time"], r["avg_time"], r["tokens_per_second"], r["total_tokens"]) 
                       for r in results],
                      columns=["Количество параллельных запросов", "Общее время (с)", "Среднее время на запрос (с)", "Токены в секунду", "Сумма токенов"])
    
    # Сохранение в CSV
    df.to_csv(f"results/results_{parallels}.csv", index=False)
    
    # Завершение процесса ollama
    try:
        print("Завершаю процесс ollama...")
        # Используем taskkill для принудительного завершения процесса ollama
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Процесс ollama завершен")
    except Exception as e:
        print(f"Ошибка при завершении процесса ollama: {e}")
    
    
    
    # Построение графика с добавлением нового подграфика
    plt.figure(figsize=(12, 9))  # Увеличим размер для дополнительного подграфика
    
    plt.subplot(2, 2, 1)
    plt.plot(df["Количество параллельных запросов"], df["Общее время (с)"], marker='o')
    plt.title("Общее время обработки")
    plt.xlabel("Количество параллельных запросов")
    plt.ylabel("Время (с)")
    
    plt.subplot(2, 2, 2)
    plt.plot(df["Количество параллельных запросов"], df["Среднее время на запрос (с)"], marker='o')
    plt.title("Среднее время на запрос")
    plt.xlabel("Количество параллельных запросов")
    plt.ylabel("Время (с)")
    
    plt.subplot(2, 2, 3)
    plt.plot(df["Количество параллельных запросов"], df["Токены в секунду"], marker='o')
    plt.title("Токены в секунду")
    plt.xlabel("Количество параллельных запросов")
    plt.ylabel("Токены/с")
    
    plt.subplot(2, 2, 4)
    plt.plot(df["Количество параллельных запросов"], df["Сумма токенов"], marker='o')
    plt.title("Сумма токенов по количеству параллельных запросов")
    plt.xlabel("Количество параллельных запросов")
    plt.ylabel("Сумма токенов")
    
    plt.tight_layout()
    plt.savefig(f"results/performance_metrics_{parallels}.png")
    print(f"Результаты сохранены в results/results_{parallels}.csv и results/performance_metrics_{parallels}.png")
    end_time = time.time()
    print(f"Завершено за {(end_time - start_time):.2f} секунд")

if __name__ == "__main__":
    start_time_all = time.time()
    
    # start_time = time.time()
    # print("Запуск проверки скорости для 4 параллельных запросов (до 30 запросов)")
    # asyncio.run(main(parallels=4, max_requests=30))
    # end_time = time.time()

    # print("Запуск проверки скорости для 6 параллельных запросов (до 30 запросов)")
    # start_time = time.time()
    # asyncio.run(main(parallels=6, max_requests=30))
    # end_time = time.time()

    print("Запуск проверки скорости для 8 параллельных запросов (до 30 запросов)")
    start_time = time.time()
    asyncio.run(main(parallels=8, max_requests=30))
    end_time = time.time()

    # print("Запуск проверки скорости для 10 параллельных запросов (до 30 запросов)")
    # start_time = time.time()
    # asyncio.run(main(parallels=10, max_requests=30))
    # end_time = time.time()

    # print("Запуск проверки скорости для 12 параллельных запросов (до 30 запросов)")
    # start_time = time.time()
    # asyncio.run(main(parallels=12, max_requests=30))
    # end_time = time.time()
    # print(f"Завершено за {(end_time - start_time):.2f} секунд")
    
    start_time = time.time()
    print("Запуск проверки скорости для 15 параллельных запросов (до 30 запросов)")
    asyncio.run(main(parallels=15, max_requests=30))
    end_time = time.time()
    print(f"Завершено за {(end_time - start_time):.2f} секунд")
    
    start_time = time.time()
    print("Запуск проверки скорости для 20 параллельных запросов (до 30 запросов)")
    asyncio.run(main(parallels=20, max_requests=30))
    end_time = time.time()
    print(f"Завершено за {(end_time - start_time):.2f} секунд")

    print(f"Полный прогон за {(end_time - start_time_all) / 60:.2f} минут")
    
    # Генерация отчета
    generate_report()
