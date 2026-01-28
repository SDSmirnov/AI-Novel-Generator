#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для стилистической очистки текста (Версия 2.0 - Гибридная).
Детектирование ошибок - на уровне предложений.
Исправление - на уровне абзаца (для сохранения контекста).
Работает параллельно с использованием ThreadPoolExecutor.

Пример запуска:
python detect-and-fix-bad-style.py --input book.txt --output style-fixed-book.txt --logfile style-fixes.log
"""

import os
import sys
import json
import re
import logging
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from qdrant_client.models import PointStruct, VectorParams, Distance

try:
    import google.generativeai as genai
    from qdrant_client import QdrantClient, models
except ImportError:
    print("Ошибка: Необходимые библиотеки не установлены.", file=sys.stderr)
    print("Выполните: pip install google-generativeai qdrant-client tqdm", file=sys.stderr)
    sys.exit(1)

# --- Глобальные константы ---
QDRANT_COLLECTION = "bad_style_v12"
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-flash-latest"
BAD_EXAMPLES_JSON = "bad_style_examples.json"

# --- 1. Настройка Утилит ---

def setup_logging(logfile: str):
    """Настраивает два потока логирования: в файл (подробный) и в консоль (инфо)."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=logfile,
        filemode='w'
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(console_handler)

def load_bad_style_examples(json_path: str) -> list[str]:
    """Загружает список 'красных флагов' из JSON."""
    if not os.path.exists(json_path):
        logging.critical(f"Файл стилистических ошибок '{json_path}' не найден.")
        sys.exit(1)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        examples = data.get("bad_style_examples")
        if not examples or not isinstance(examples, list):
            raise ValueError("JSON не содержит ключ 'bad_style_examples' или он не является списком.")
        logging.info(f"Загружено {len(examples)} стилистических 'красных флагов' из {json_path}")
        return examples
    except Exception as e:
        logging.critical(f"Ошибка загрузки JSON '{json_path}': {e}")
        sys.exit(1)

def setup_qdrant(examples: list[str]) -> QdrantClient:
    """Инициализирует Qdrant в памяти и индексирует 'красные флаги'."""
    logging.info("Инициализация Qdrant в :memory:...")
    client = QdrantClient(location=":memory:")
    
    try:
        test_embedding = genai.embed_content(model=EMBEDDING_MODEL, content="test", task_type="retrieval_document")
        vector_size = len(test_embedding['embedding'])
    except Exception as e:
        logging.critical(f"Не удалось получить размер вектора от Gemini. Ошибка: {e}")
        sys.exit(1)
    
    logging.info(f"Размер векторов: {vector_size}. Создание коллекции '{QDRANT_COLLECTION}'...")
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )
    
    logging.info(f"Векторизация {len(examples)} 'красных флагов'...")
    try:
        vectors = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=examples,
            task_type="retrieval_document" # Индексируем как документы
        )['embedding']
    except Exception as e:
        logging.critical(f"Ошибка векторизации 'красных флагов' в Gemini: {e}")
        sys.exit(1)

    logging.info("Загрузка векторов в Qdrant...")
    points = []
    for i, ex in enumerate(examples):
        points.append(PointStruct(id=i, vector=vectors[i], payload={'example': ex}))
    client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    logging.info("Индексация 'красных флагов' в Qdrant завершена.")
    return client

def split_text_into_chunks(text: str) -> list[str]:
    """Разбивает текст на параграфы (по пустым строкам)."""
    chunks = re.split(r'\n+', text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    logging.info(f"Текст разделен на {len(chunks)} абзацев (контекстных блоков).")
    return chunks

def split_chunk_into_sentences(chunk: str) -> list[str]:
    """
    Разбивает абзац на предложения.
    Используем регулярное выражение, которое ищет конец предложения (.,!,?)
    и оставляет его частью предложения.
    """
    if not chunk:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    return [s.strip() for s in sentences if s.strip()]


# --- 2. Логика Работников (Workers) ---

def get_llm_fix(paragraph: str, problematic_sentences: list, llm: genai.GenerativeModel) -> str:
    """
    Отправляет полный абзац и список проблемных предложений в Gemini.
    """
    
    # Собираем список найденных проблем
    hits_summary = ""
    for sentence, hits in problematic_sentences:
        hits_summary += f"\n--- \nВ предложении: \"{sentence}\"\n"
        hits_summary += "Обнаружены семантические проблемы (клише):\n"
        hits_summary += "\n".join([
            f"- (Score: {hit.score:.2f}) Похоже на: '{hit.payload['example']}'"
            for hit in hits
        ])
    
    prompt = f"""
Ты — строгий редактор "Зануда". Твоя задача — стилистически улучшить текст,
исправляя клише, повторы и чрезмерные физиологизмы, СОХРАНЯЯ КОНТЕКСТ.

Жанр романа - современная проза, эротический триллер, dark romance.
Ниже приведен полный абзац и список конкретных предложений из него,
в которых система детекции нашла стилистические ошибки ("красные флаги").

---
ПОЛНЫЙ АБЗАЦ (для контекста):
"{paragraph}"
---
СПИСОК ПРОБЛЕМНЫХ МЕСТ:
{hits_summary}
---

ЗАДАЧA:
Аккуратно отредактируй **весь абзац**, уделяя ОСОБОЕ внимание исправлению стилистических ошибок в указанных предложениях.
Ты можешь переформулировать, объединить или сократить предложения, чтобы улучшить стиль и сохранить связность.

ПРАВИЛА:
1. СОХРАНИ 100% ПЕРВОНАЧАЛЬНОГО СМЫСЛА И СЮЖЕТА. Не меняй ход событий.
2. НЕ ДОБАВЛЯЙ НИЧЕГО ОТ СЕБЯ. Работай только с предоставленным текстом.
3. Верни ТОЛЬКО отредактированный абзац, БЕЗ КОММЕНТАРИЕВ, предисловий или извинений.
"""
    
    try:
        response = llm.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3 # Более строгое, "редакторское" следование
            )
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"Ошибка API Gemini при исправлении абзаца: {e}")
        return paragraph # Возвращаем оригинал в случае сбоя

def process_chunk(
    chunk_tuple: tuple[int, str],
    qdrant_client: QdrantClient,
    llm: genai.GenerativeModel,
    threshold: float
) -> tuple[int, str, str | None]:
    """
    Один рабочий цикл для ThreadPoolExecutor (обработка одного абзаца).
    Возвращает (index, original_text, fixed_text)
    """
    index, paragraph = chunk_tuple
    
    if not paragraph:
        return (index, paragraph, None)
        
    try:
        # 1. Разделяем абзац на предложения
        sentences = split_chunk_into_sentences(paragraph)
        if not sentences:
            return (index, paragraph, None)
            
        # 2. Векторизуем все предложения (как запросы)
        sentence_vectors = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=sentences,
            task_type="retrieval_query"
        )['embedding']
        
        # 3. Ищем 'красные флаги' в Qdrant (пакетный поиск)
        search_requests = []
        for vector in sentence_vectors:
            search_requests.append(
                models.SearchRequest(
                    vector=vector,
                    limit=5, # Ищем до 5 совпадений на предложение
                    score_threshold=threshold,
                    with_payload=True,
                )
            )
        
        search_results_list = qdrant_client.search_batch(
            collection_name=QDRANT_COLLECTION,
            requests=search_requests
        )

        # 4. Анализируем результаты
        problematic_sentences = []
        for i, hits in enumerate(search_results_list):
            if hits: # Если для i-го предложения найдены проблемы
                problematic_sentences.append((sentences[i], hits))

        # 5. Принимаем решение
        if not problematic_sentences:
            # Проблем в абзаце не найдено
            return (index, paragraph, None)
        
        # Проблемы найдены, отправляем на исправление
        logging.info(f"Абзац {index}: Найдено {len(problematic_sentences)} проблемных предложений. Отправка в LLM...")
        
        # 6. Исправляем (с полным контекстом абзаца)
        # Добавляем небольшую задержку, чтобы не превысить QPM (запросы в минуту)
        time.sleep(0.5) 
        fixed_paragraph = get_llm_fix(paragraph, problematic_sentences, llm)
        
        if fixed_paragraph == paragraph:
            logging.warning(f"Абзац {index}: LLM не внес изменений, несмотря на найденные флаги.")
            return (index, paragraph, None)
            
        return (index, paragraph, fixed_paragraph)

    except Exception as e:
        logging.error(f"Критическая ошибка при обработке абзаца {index}: {e}")
        import traceback
        print(traceback.format_exc())
        import sys
        sys.exit(1)
        return (index, paragraph, None)

# --- 3. Основной скрипт ---

def main():
    parser = argparse.ArgumentParser(
        description="Гибридная очистка стиля (Детекция: Предложения, Правка: Абзацы).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True, help="Входной файл .txt (книга).")
    parser.add_argument('--output', '-o', type=str, required=True, help="Выходной файл .txt (исправленная книга).")
    parser.add_argument('--logfile', '-l', type=str, required=True, help="Файл лога для записи всех изменений.")
    parser.add_argument('--examples', '-e', type=str, default=BAD_EXAMPLES_JSON, help=f"Путь к {BAD_EXAMPLES_JSON} с 'красными флагами'.")
    parser.add_argument('--max_workers', '-w', type=int, default=20, help="Количество параллельных потоков (max_workers).")
    parser.add_argument('--threshold', '-t', type=float, default=0.82, help="Порог срабатывания Qdrant (0.0 до 1.0).")
    args = parser.parse_args()

    # 1. Настройка
    setup_logging(args.logfile)
    logging.info("--- Запуск стилистического редактора 'Зануда' (v2.0 Гибридный) ---")
    
    try:
        api_key = os.environ.get("AI_API_KEY")
        if not api_key:
            logging.critical("Переменная окружения AI_API_KEY не установлена.")
            sys.exit(1)
        genai.configure(api_key=api_key)
    except Exception as e:
        logging.critical(f"Ошибка конфигурации Gemini: {e}")
        sys.exit(1)

    # 2. Загрузка и индексация
    bad_examples = load_bad_style_examples(args.examples)
    qdrant_client = setup_qdrant(bad_examples)
    
    # 3. Инициализация LLM
    llm = genai.GenerativeModel(
        LLM_MODEL,
        system_instruction="Ты — строгий редактор 'Зануда'. Твоя задача — не переписывать текст, а исправлять конкретные стилистические ошибки: клише, повторы, чрезмерные эпитеты. Ты возвращаешь ТОЛЬКО отредактированный текст без каких-либо комментариев."
    )

    # 4. Чтение и разделение входного файла
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        logging.critical(f"Входной файл '{args.input}' не найден.")
        sys.exit(1)
    
    chunks = split_text_into_chunks(full_text) # Chunks = Абзацы
    chunks_with_indices = list(enumerate(chunks))
    
    fixed_chunks_map = {} # Словарь для сборки в правильном порядке
    total_fixed = 0

    # 5. Запуск параллельной обработки
    logging.info(f"Запуск ThreadPoolExecutor с {args.max_workers} потоками...")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_chunk = {
            executor.submit(
                process_chunk, 
                chunk_tuple, 
                qdrant_client=qdrant_client, 
                llm=llm, 
                threshold=args.threshold
            ): chunk_tuple[0] 
            for chunk_tuple in chunks_with_indices
        }
        
        for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Анализ и правка абзацев"):
            try:
                index, original_text, fixed_text = future.result()
                
                if fixed_text:
                    total_fixed += 1
                    logging.debug(f"--- ИСПРАВЛЕНИЕ (Абзац {index}) ---\n[ДО]:\n{original_text}\n\n[ПОСЛЕ]:\n{fixed_text}\n----------------------------------")
                    fixed_chunks_map[index] = fixed_text
                else:
                    fixed_chunks_map[index] = original_text # Сохраняем оригинал
                    
            except Exception as e:
                chunk_index = future_to_chunk[future]
                logging.error(f"Ошибка в потоке для абзаца {chunk_index}: {e}")
                fixed_chunks_map[chunk_index] = chunks[chunk_index] # Сохраняем оригинал

    # 6. Сборка и сохранение результата
    logging.info("Сборка исправленного текста...")
    
    final_text_chunks = [fixed_chunks_map[i] for i in sorted(fixed_chunks_map.keys())]
    final_text = "\n\n".join(final_text_chunks)
    
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(final_text)
    except Exception as e:
        logging.critical(f"Не удалось записать результат в '{args.output}': {e}")
        sys.exit(1)

    logging.info("--- РАБОТА ЗАВЕРШЕНА ---")
    logging.info(f"Исправлено {total_fixed} из {len(chunks)} абзацев.")
    logging.info(f"Результат сохранен в: {args.output}")
    logging.info(f"Подробный лог сохранен в: {args.logfile}")

if __name__ == "__main__":
    main()
