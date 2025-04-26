## Выявляение оптимального размера batch_size для ollama (или поиск оптимального количества запросов, которые обрабатывает ollama параллельно)

Выполняется несколько прогонов с разным размером batch (переменная OLLAMA_NUM_PARALLEL).

Фактически выполняется команда `setx OLLAMA_NUM_PARALLEL 10; ollama serve`.

В результате генерируются файлы `results/results_10.csv`, `results/performance_metrics_10.png`, ...

Задать свой набор контрольных значений можно в файле `check_speed.py` в разделе `if __name__ == "__main__":`.

Для каждого прогона указывается batch_size (сколько ollama обрабатывает параллельно) и max_requests (сколько всего запросов будет выполнено).

Расчет количества запросов выполняется по формуле:

Начальные значения:
1, 2, 4, 8, 16

Последующие значения добавляют по 8:

24, 32, 40, 48, 56, 64, 72, 80, 88, 96 ...

Это нужно для упрощения анализа.

## Отчет

[Отчет](report.md)

Отладка проводилась на GeForce 4060 Ti.


## Установка и настройка

0. Установите ollama

Скачайте ollama с https://ollama.com/download

или выполните команду:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

1. Клонируйте репозиторий:
```bash
git clone https://github.com/nicolay-i/check_parallels
cd check_parallels
```

2. Создайте и активируйте виртуальное окружение:
```bash
python -m venv venv
# Для Windows
venv\Scripts\activate
# Для Linux/Mac
source venv/bin/activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Запустите скрипт:
```bash
python check_speed.py
```


