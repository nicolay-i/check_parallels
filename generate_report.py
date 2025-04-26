# Генерация отчета
# 1. Считывание всех файлов с результатами из папки results
# 2. Генерация отчета в формате markdown
# 3. Сохранение отчета в файл report.md

# Формат отчета:
# Заголовок с указанием количества параллельных запросов (берется из названия файла)
# График (изображение performance_metrics_{n}.png)
# Таблица с результатами (берется из файла results_{n}.csv)
# Итоговые выводы по всем прогонам
# Какой размер batch_size оптимальный для данного оборудования с точки зрения максимальной скорости токенов / секунду


import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from glob import glob

def generate_report():
    # Поиск всех файлов с результатами
    results_files = sorted(glob('results/results_*.csv'), key=lambda x: int(re.search(r'results_(\d+)\.csv', x).group(1)))
    
    # Подготовка данных для итоговых выводов
    all_results = []
    for result_file in results_files:
        parallel_requests = re.search(r'results_(\d+)\.csv', result_file).group(1)
        df = pd.read_csv(result_file)
        best_result = df.loc[df['Токены в секунду'].idxmax()]
        all_results.append({
            'Количество параллельных запросов': int(parallel_requests),
            'Оптимальный размер пакета': int(best_result['Количество параллельных запросов']),
            'Максимальная скорость (токены/сек)': best_result['Токены в секунду']
        })
    
    # Создание DataFrame с итоговыми результатами
    all_results_df = pd.DataFrame(all_results)
    
    # Определение абсолютно лучшей конфигурации
    best_config = all_results_df.loc[all_results_df['Максимальная скорость (токены/сек)'].idxmax()]
    
    with open('report.md', 'w', encoding='utf-8') as report_file:
        report_file.write('# Отчет о производительности\n\n')
        
        report_file.write('## Итоговые выводы по всем прогонам\n\n')
        
        # Таблица с результатами всех прогонов
        report_file.write('### Оптимальные параметры для разного количества параллельных запросов\n\n')
        report_file.write(all_results_df.to_markdown(index=False) + '\n\n')
        
        # Лучшая конфигурация
        report_file.write('### Наилучшая конфигурация для данного оборудования\n\n')
        report_file.write(f'- **Количество параллельных запросов:** {int(best_config["Количество параллельных запросов"])}\n')
        report_file.write(f'- **Оптимальный размер пакета:** {int(best_config["Оптимальный размер пакета"])}\n')
        report_file.write(f'- **Скорость обработки:** {best_config["Максимальная скорость (токены/сек)"]:.2f} токенов/сек\n\n')
                
        report_file.write('---\n\n')
        
        # Детальные результаты по каждому прогону
        report_file.write('## Детальные результаты по каждому прогону\n\n')
        
        for result_file in results_files:
            # Извлечение числа параллельных запросов из имени файла
            parallel_requests = re.search(r'results_(\d+)\.csv', result_file).group(1)
            
            # Чтение данных из CSV
            df = pd.read_csv(result_file)
            
            # Добавление заголовка секции в отчет
            report_file.write(f'### Результаты для {parallel_requests} параллельных запросов\n\n')
            
            # Добавление графика
            image_path = f'results/performance_metrics_{parallel_requests}.png'
            if os.path.exists(image_path):
                report_file.write(f'![График производительности для {parallel_requests} параллельных запросов]({image_path})\n\n')
            
            # Добавление таблицы с результатами
            report_file.write('#### Таблица результатов\n\n')
            report_file.write(df.to_markdown(index=False) + '\n\n')
            
            # Находим оптимальный размер batch_size
            optimal_batch = df.loc[df['Токены в секунду'].idxmax()]
            report_file.write(f'**Оптимальный размер пакета:** {int(optimal_batch["Количество параллельных запросов"])}\n\n')
            report_file.write(f'**Максимальная скорость обработки:** {optimal_batch["Токены в секунду"]:.2f} токенов в секунду\n\n')
            
            report_file.write('---\n\n')

if __name__ == "__main__":
    generate_report()
    print("Отчет успешно сгенерирован в файле report.md")

