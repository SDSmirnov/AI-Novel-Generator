#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import google.generativeai as genai
import json
import os
import argparse
import logging
from google.api_core import retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('final_fix_apply.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Настройки безопасности для API ---
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# --- Схема для ответа модели ---
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewrites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "Уникальный ID фрагмента, который был передан в запросе."},
                    "fixed_text": {"type": "string", "description": "Переписанный, исправленный текст абзаца."}
                },
                "required": ["chunk_id", "fixed_text"]
            }
        }
    },
    "required": ["rewrites"]
}

# --- Системный промпт ---
SYSTEM_PROMPT_REWRITER = """
Ты — элитный литературный редактор, специализирующий на современной русской прозе в жанрах мистики и психологических триллеров. Твоя задача — выполнить точечную стилистическую редактуру группы проблемных абзацев.
Жанр романа - современная проза, эротический триллер, dark romance.

ПРАВИЛА РАБОТЫ:
1.  **ГЛУБОКИЙ КОНТЕКСТ**: Для каждого проблемного абзаца тебе предоставлен ПОЛНЫЙ ТЕКСТ ГЛАВЫ, в которой он находится. Перед тем как вносить правку, ты обязан ознакомиться с главой, чтобы понять её ритм, тон, эмоциональный накал и место абзаца в повествовании.
2.  **ХОЛИСТИЧЕСКИЙ ПОДХОД К КЛАСТЕРУ**: Все абзацы, переданные в одном запросе, страдают от одной и той же стилистической проблемы (клише или самоповтора). Твоя задача — решить эту проблему для каждого абзаца, но сделать это РАЗНООБРАЗНО. Не заменяй одно клише другим, которое будет повторяться. Предложи уникальные, но стилистически согласованные решения.
3.  **ХИРУРГИЧЕСКАЯ ТОЧНОСТЬ**: Ты переписываешь ТОЛЬКО указанный «АБЗАЦ ДЛЯ ИСПРАВЛЕНИЯ». Сюжет, диалоги, действия персонажей и ключевые детали должны быть сохранены.
4.  **ГОЛОС АВТОРА**: Сохраняй авторский стиль, лексику и ритмику. Твоя правка должна быть незаметной, органично вписываясь в текст.
5.  **СТРОГИЙ JSON-ОТВЕТ**: Твой ответ должен быть единым JSON объектом, строго соответствующим предоставленной схеме. Не включай никаких комментариев или текста вне JSON.
"""

@retry.Retry(predicate=retry.if_transient_error, deadline=300)
def call_llm_chapter_context_rewrite(model, prompt: str) -> dict:
    """Вызывает LLM для переписывания кластера с полным контекстом главы."""
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA
    )
    response = model.generate_content(prompt, generation_config=generation_config)
    try:
        cleaned_response = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Не удалось распарсить JSON из ответа модели: {e}\nОтвет модели:\n{response.text}")
        return {"rewrites": []}


def process_cluster(cluster_data: dict, model, chapters: list, processed_locations: set, lock: Lock) -> list:
    """
    Обрабатывает один кластер проблем.
    Возвращает список применённых изменений в формате [(ch_idx, p_idx, fixed_text), ...]
    """
    result = cluster_data['result']
    i = cluster_data['index']
    total = cluster_data['total']
    
    cluster_id = result.get('cluster')
    logger.info(f"--- Обработка кластера #{cluster_id} ({i+1}/{total}): {result.get('motive')} ---")
    
    if not (result.get('status', '') == 'CRITICAL' or result.get('confidence', 0) > 0.85):
        logger.info(f"SKIPPED: {result.get('status')} confidence {result.get('confidence', 0)}")
        return []

    chunks_for_prompt = []
    all_chunks_in_cluster = [result.get('original')] + result.get('similar', [])

    for chunk_data in all_chunks_in_cluster:
        if not chunk_data or not chunk_data.get('text'):
            continue

        chapter_from_report = chunk_data.get('chapter')
        paragraph_from_report = chunk_data.get('paragraph_id') if 'paragraph_id' in chunk_data else chunk_data.get('paragraph')

        if chapter_from_report is None or paragraph_from_report is None:
            logger.warning(f"В данных отчета для кластера #{cluster_id} отсутствуют 'chapter' или 'paragraph'. Пропускаем фрагмент.")
            continue

        chapter_idx = chapter_from_report - 1
        paragraph_idx = paragraph_from_report
        location = (chapter_idx, paragraph_idx)

        # Потокобезопасная проверка и добавление локации
        with lock:
            if location in processed_locations:
                continue
            processed_locations.add(location)

        try:
            full_chapter_text = "\n\n".join(p.strip() for p in chapters[chapter_idx] if p.strip())
            text_to_fix = chapters[chapter_idx][paragraph_idx]

            if text_to_fix.strip() != chunk_data['text'].strip():
                logger.warning(f"Несоответствие текста в кластере #{cluster_id} по адресу (Глава {chapter_idx+1}, Абзац {paragraph_idx}).")

            chunk_id = f"loc_{chapter_idx}_{paragraph_idx}"

            chunks_for_prompt.append({
                "chunk_id": chunk_id,
                "location": location,
                "full_chapter_text": full_chapter_text,
                "text_to_fix": text_to_fix
            })

        except IndexError:
            logger.error(f"Некорректные индексы (Глава {chapter_idx+1}, Абзац {paragraph_idx}) для кластера #{cluster_id}. Пропускаем.")
            continue

    if not chunks_for_prompt:
        logger.info(f"Все фрагменты в кластере #{cluster_id} уже были обработаны или пропущены.")
        return []

    # Формирование промпта
    prompt_body = ""
    for chunk in chunks_for_prompt:
        prompt_body += (
            f'---\n'
            f'CHUNK_ID: "{chunk["chunk_id"]}"\n\n'
            f'ПОЛНЫЙ ТЕКСТ ГЛАВЫ (для контекста):\n'
            f'"""\n{chunk["full_chapter_text"]}\n"""\n\n'
            f'АБЗАЦ ДЛЯ ИСПРАВЛЕНИЯ (внутри этой главы):\n'
            f'"""\n{chunk["text_to_fix"]}\n"""\n'
            f'---\n\n'
        )

    final_prompt = (
        f"ОБЩИЙ ДИАГНОЗ ДЛЯ ГРУППЫ АБЗАЦЕВ:\n{result.get('diagnosis')}\n\n"
        f"ОБЩАЯ РЕКОМЕНДАЦИЯ:\n{result.get('recommendation')}\n\n"
        f"ТВОЯ ЗАДАЧА: Перепиши КАЖДЫЙ из следующих абзацев, используя полный текст их глав как контекст и следуя общей рекомендации. "
        f"Для каждого абзаца найди уникальное, но стилистически согласованное решение. Верни ответ в предписанном JSON формате.\n\n"
        f"{prompt_body}"
    )

    # Вызов модели
    try:
        response_data = call_llm_chapter_context_rewrite(model, final_prompt)
        rewrites = response_data.get("rewrites", [])

        if not rewrites:
            logger.warning(f"Модель вернула пустой или некорректный ответ для кластера #{cluster_id}.")
            return []

        # Сбор изменений
        changes = []
        for rewrite in rewrites:
            chunk_id = rewrite.get("chunk_id")
            fixed_text = rewrite.get("fixed_text")

            original_chunk = next((c for c in chunks_for_prompt if c["chunk_id"] == chunk_id), None)
            if original_chunk and fixed_text:
                ch_idx, p_idx = original_chunk["location"]
                logger.info(f"Правка для Главы {ch_idx+1}, Абзаца {p_idx}")
                changes.append((ch_idx, p_idx, fixed_text))
            else:
                logger.warning(f"Не удалось найти локацию для chunk_id из ответа модели: {chunk_id}")

        return changes

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке ответа модели для кластера #{cluster_id}: {e}")
        return []


def apply_chapter_context_fixes(input_json_path: str, output_txt_path: str, gemini_api_key: str, max_workers: int = 5):
    """Основная функция, реализующая финальный, контекстно-зависимый подход с многопоточностью."""
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(
        model_name="gemini-flash-latest",
        system_instruction=SYSTEM_PROMPT_REWRITER,
        safety_settings=SAFETY_SETTINGS
    )

    logger.info(f"Загрузка отчета из {input_json_path}...")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Не удалось прочитать JSON-файл: {e}")
        return

    chapters = data.get('chapters', [])
    analysis_results = data.get('analysis', [])
    if not chapters:
        logger.error("В JSON-файле не найден раздел 'chapters'.")
        return

    problems_to_fix = [res for res in analysis_results if res.get('status') in ['STRONG_CLICHE', 'SLIGHT_REPETITION']]
    logger.info(f"Найдено {len(problems_to_fix)} кластеров стилистических проблем для исправления.")
    problems_to_fix.sort(key=lambda x: (x['status'] != 'STRONG_CLICHE', x.get('confidence', 0)), reverse=False)

    chapters_copy = [list(ch) for ch in chapters]
    processed_locations = set()
    lock = Lock()

    # Подготовка данных для параллельной обработки
    cluster_tasks = [
        {
            'result': result,
            'index': i,
            'total': len(problems_to_fix)
        }
        for i, result in enumerate(problems_to_fix)
    ]

    # Параллельная обработка кластеров
    all_changes = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Запуск всех задач
        future_to_cluster = {
            executor.submit(
                process_cluster,
                cluster_data,
                model,
                chapters,
                processed_locations,
                lock
            ): cluster_data['result'].get('cluster')
            for cluster_data in cluster_tasks
        }

        # Сбор результатов по мере выполнения
        for future in as_completed(future_to_cluster):
            cluster_id = future_to_cluster[future]
            try:
                changes = future.result()
                all_changes.extend(changes)
                if changes:
                    logger.info(f"Кластер #{cluster_id}: применено {len(changes)} изменений")
            except Exception as e:
                logger.error(f"Ошибка при обработке кластера #{cluster_id}: {e}")

    # Применение всех изменений
    logger.info(f"Применение {len(all_changes)} изменений к тексту...")
    for ch_idx, p_idx, fixed_text in all_changes:
        chapters_copy[ch_idx][p_idx] = fixed_text

    # Сохранение результата
    logger.info(f"Сборка и сохранение исправленного текста в {output_txt_path}...")
    final_text_parts = []
    for index, chapter_content in enumerate(chapters_copy):
        clean_chapter = "\n\n".join(p for p in chapter_content if p.strip())
        final_text_parts.append(f"## Глава {index+1}\n\n{clean_chapter}")

    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n\n\n'.join(final_text_parts))
        logger.info(f"Финальная версия текста успешно сохранена в {output_txt_path}!")
    except Exception as e:
        logger.error(f"Не удалось сохранить итоговый файл: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Применяет стилистические исправления к тексту на основе JSON-отчета, с учетом полного контекста главы."
    )
    parser.add_argument("input_file", help="Путь к JSON-файлу с отчетом (analysis_report.md.json).")
    parser.add_argument("--output", default="b4d_m00d_final_version.txt", help="Путь к итоговому файлу.")
    parser.add_argument("--api-key", help="Ключ API Google Gemini (или переменная окружения AI_API_KEY).")
    parser.add_argument("--workers", type=int, default=15, help="Количество параллельных потоков (по умолчанию 5).")

    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("AI_API_KEY")
    if not api_key:
        logger.error("Необходимо указать ключ API Gemini.")
        return

    apply_chapter_context_fixes(args.input_file, args.output, api_key, args.workers)


if __name__ == "__main__":
    main()
