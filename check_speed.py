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

system_prompt_keywoards = """
Переведи текст с комментариями на русский, сохраняя форматирование комментариев с ">" и оставляя без изменений названия полей "UserName" и "Text".  
Необходимо точно передать смысл комментариев, сохраняя структуру и стиль исходного текста.  
Учитывай контекст всего перевода, чтобы обеспечить плавную и корректную передачу информации.  
Результат должен быть точным и хорошо читаемым, идеально подходящим для использования в системе.  
При необходимости, используй дополнительные слова или фразы для улучшения качества перевода.
"""

user_prompt_keywoards = """
### Title
STT with multi-shot

### Text
Hey everyone,
I'm looking into selfhosting some STT for my apps, but I wanted to know if there's a way to do multi-shot ? Like with in context some .wav | transcription | .wav | transcription so that I can adapt it to my voice without re-training it.

Or is whisper.cpp still the only good way ?

### Comments
> UserName: t2_mkmia1m4g
> Text: If you are looking for optimized ways of using the whisper then you can try faster whisper.  We have already created a blog on this: https://docs.inferless.com/how-to-guides/deploy-whisper-large-v3-using-inferless#deploy-whisper-large-v3-using-inferless
"""

# загрузка переменных окружения
load_dotenv()

async def process_query(index):
    """Отправка запроса к API OpenAI"""
    try:
        answer, tokens_answer, model_answer = await ask_ai(system_prompt=system_prompt_keywoards, prompt=user_prompt_keywoards, timeout=6000)
        
        print(f"Получил ответ {index} запроса ({tokens_answer} токенов)")
        
        return answer, tokens_answer
    except Exception as e:
        print(f"Ошибка при запросе к API: {e}")
        return None

async def run_batch(num_parallels):
    """Запуск пакета запросов заданного размера"""
    start_time = time.time()
    tasks = [process_query(i) for i in range(num_parallels)]
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
        
        if current_size > 40:
            current_size *= 1.5
            continue

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
    
    max_requests = 50
    
    start_time = time.time()
    print("Запуск проверки скорости для 3 параллельных запросов (до 12 запросов)")
    asyncio.run(main(parallels=3, max_requests=12))
    end_time = time.time()

    start_time = time.time()
    print("Запуск проверки скорости для 5 параллельных запросов (до 30 запросов)")
    asyncio.run(main(parallels=5, max_requests=30))
    end_time = time.time()

    start_time = time.time()
    print("Запуск проверки скорости для 6 параллельных запросов (до 30 запросов)")
    asyncio.run(main(parallels=6, max_requests=30))
    end_time = time.time()

    # start_time = time.time()
    # print("Запуск проверки скорости для 4 параллельных запросов (до 50 запросов)")
    # asyncio.run(main(parallels=4, max_requests=max_requests))
    # end_time = time.time()

    # print("Запуск проверки скорости для 6 параллельных запросов (до 50 запросов)")
    # start_time = time.time()
    # asyncio.run(main(parallels=6, max_requests=max_requests))
    # end_time = time.time()

    # print("Запуск проверки скорости для 8 параллельных запросов (до 30 запросов)")
    # start_time = time.time()
    # asyncio.run(main(parallels=8, max_requests=max_requests))
    # end_time = time.time()

    # print("Запуск проверки скорости для 10 параллельных запросов (до 30 запросов)")
    # start_time = time.time()
    # asyncio.run(main(parallels=10, max_requests=max_requests))
    # end_time = time.time()

    # print("Запуск проверки скорости для 12 параллельных запросов (до 30 запросов)")
    # start_time = time.time()
    # asyncio.run(main(parallels=12, max_requests=max_requests))
    # end_time = time.time()
    # print(f"Завершено за {(end_time - start_time):.2f} секунд")
    
    # start_time = time.time()
    # print("Запуск проверки скорости для 15 параллельных запросов (до 30 запросов)")
    # asyncio.run(main(parallels=15, max_requests=max_requests))
    # end_time = time.time()
    # print(f"Завершено за {(end_time - start_time):.2f} секунд")
    
    # start_time = time.time()
    # print("Запуск проверки скорости для 20 параллельных запросов (до 30 запросов)")
    # asyncio.run(main(parallels=20, max_requests=max_requests))
    # end_time = time.time()
    # print(f"Завершено за {(end_time - start_time):.2f} секунд")

    # start_time = time.time()
    # print("Запуск проверки скорости для 18 параллельных запросов (до 30 запросов)")
    # asyncio.run(main(parallels=18, max_requests=max_requests))
    # end_time = time.time()
    # print(f"Завершено за {(end_time - start_time):.2f} секунд")


    # start_time = time.time()
    # print("Запуск проверки скорости для 22 параллельных запросов (до 30 запросов)")
    # asyncio.run(main(parallels=22, max_requests=max_requests))
    # end_time = time.time()
    # print(f"Завершено за {(end_time - start_time):.2f} секунд")


    # start_time = time.time()
    # print("Запуск проверки скорости для 24 параллельных запросов (до 30 запросов)")
    # asyncio.run(main(parallels=24, max_requests=max_requests))
    # end_time = time.time()
    # print(f"Завершено за {(end_time - start_time):.2f} секунд")



    print(f"Полный прогон за {(end_time - start_time_all) / 60:.2f} минут")
    
    # Генерация отчета
    generate_report()
