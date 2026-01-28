import google.generativeai as genai
import json
import numpy as np
import os
import random
import re
import sys
import time
import argparse

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Tuple, Dict, Any

# --- КОНФИГУРАЦИЯ ИЗ ОРИГИНАЛЬНОГО ФАЙЛА ---
# (Необходима для вызова API в этом скрипте)

API_KEY = os.getenv('AI_API_KEY', 'ВАШ_API_КЛЮЧ')
MODEL_NAME = "gemini-2.5-pro"

# Загружаем глобальные концепты и оверрайды из оригинальных файлов
DEFAULT_CONCEPTS = open('prompts/global_concepts.md', 'r').read() if os.path.isfile('prompts/global_concepts.md') else ""
CONCEPTS = open('concepts.txt', 'r').read() if os.path.isfile('concepts.txt') else DEFAULT_CONCEPTS

DEFAULT_OVERRIDE = open('prompts/global_override.md', 'r').read() if os.path.isfile('prompts/global_override.md') else ""
SYSTEM_OVERRIDE = open('override.txt', 'r').read() if os.path.isfile('override.txt') else DEFAULT_OVERRIDE

ANTI_PLEASING = open('prompts/global_anti_pleasing.md', 'r').read() if os.path.isfile('prompts/global_anti_pleasing.md') else ""
PLANNER_ROLE = "Твоя роль: архитектор-планировщик захватывающей художественной книги на русском языке."

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

STATS = {}

# --- КОНЕЦ КОНФИГУРАЦИИ ---

def load_prompt(name: str, **kwargs) -> str:
    """Загружает и форматирует промпт."""
    try:
        template = open(f'prompts/{name}.md', 'r', encoding='utf-8').read()
    except FileNotFoundError:
        print(f"[!] КРИТИЧЕСКАЯ ОШИБКА: Файл промпта 'prompts/{name}.md' не найден.")
        sys.exit(1)

    def evaluator(match):
        expression = match.group(1)
        try:
            # Используем locals() и globals() для более безопасного eval
            return str(eval(expression, globals(), kwargs))
        except Exception as e:
            print(f"[!] Ошибка вычисления в промпте '{name}': {expression} -> {e}")
            return f"{{EVAL_ERROR: {expression}}}"
    return re.sub(r"\{(.*?)\}", evaluator, template)

# Загружаем схемы и JSON-примеры из оригинальных файлов
PLAN_SCHEMA = json.loads(load_prompt('schema_plan'))
SCHEMA_ANALYSIS = json.loads(load_prompt('schema_analysis'))
JSON_OUT = """
{
    "chapters": [
        {
            "number": "1",
            "title": "Глава 1",
            "scenes": [
                "2025-10-17 11:30. Вася, измотанный после бессонной ночи, спускается в отмытую пивоварню. Цель — сварить простой лагер, вернуться к основам. Он не ищет прорыва, а пытается восстановить контроль через рутину. Он методично проверяет чистоту чана, пересчитывает мешки с солодом. Входит Матрёна с письмом от Пилснера. КЛЮЧЕВОЙ МОМЕНТ: Вася читает официальную жалобу на 'недопустимый уровень карбонизации'. Вместо гнева или страха он усмехается. ПОКАЗАТЬ: Это начало новой, 'холодной' войны, которая его не пугает, а забавляет. Он берет перо и пишет ироничный ответ. Сцена заканчивается на том, как Матрёна уносит его ответ, качая головой, но с тенью улыбки. Время: ~20 минут.",
                ...
            ]
        },
        ....
    ]
}
"""

import re
import json
import logging

from typing import Any, Dict, Optional

def robust_json_parser(llm_response: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to extract and parse a JSON object from a string that may contain
    markdown fences or other extraneous text.

    Args:
        llm_response: The string response from the LLM, expected to contain JSON.

    Returns:
        A dictionary if parsing is successful, otherwise None.
    """
    # 1. Look for JSON within markdown fences ` ``json ... ``` `
    llm_response = re.sub(r'\s+', ' ', llm_response)
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s```", llm_response, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.warning(f"JSON in markdown failed to parse: {e}. Content: '{json_str[:100]}...'")
            # Fall through to other methods in case of malformed JSON in markdown

    try:
        return json.loads(llm_response)
    except json.JSONDecodeError as e:
        logging.warning(f"JSON in markdown failed to parse: {e}. Content: '{llm_response[:1000]}...'")
        # Fall through to other methods in case of malformed JSON in markdown


    # 2. If no markdown, try to find the first '{' and last '}'
    try:
        start_index = llm_response.find('{')
        end_index = llm_response.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            potential_json = llm_response[start_index:end_index+1]
            return json.loads(potential_json)
    except json.JSONDecodeError:
        # Fall through if this substring is not valid JSON
        pass

    # 3. As a last resort, try parsing the whole string
    try:
        return json.loads(llm_response)
    except json.JSONDecodeError:
        pass

    logging.error(f"Could not parse JSON from LLM response after multiple attempts. Response: '{llm_response}'")
    return None

@dataclass
class Step:
    """
    Класс Step из оригинального novel_generator.py,
    нужен для создания "завершенных" шагов.
    """
    name: str
    handler_name: str
    status: str = 'planned'


class SequelPlanner:
    """
    Этот класс отвечает за подготовку файла состояния для сиквела.
    Он загружает состояние Книги 1, обновляет "Библию Мира" для Книги 2
    и создает новый план глав.
    """
    def __init__(self, api_key, previous_state):
        genai.configure(api_key=api_key)

        self.new_state = {}
        self.base_model = None
        self.world_model = None # Модель с контекстом Книги 1

        self._load_and_prepare_state(previous_state)
        self._initialize_models()

    def _load_and_prepare_state(self, previous_state):
        """Загружает и переносит ключевые данные из предыдущего состояния."""
        print("  > Загрузка состояния предыдущей книги...")

        # 1. Перенос Состояния Мира (Критично)
        self.new_state['world_state'] = previous_state.get('world_state', {})
        print(f"  ✓ Состояние мира (персонажи, инвентарь) на конец Книги 1 перенесено.")

        # 2. Перенос "Библии Мира" (Будет обновлена)
        self.new_state['world_bible'] = previous_state.get('world_bible', {})
        if not self.new_state['world_bible']:
            print("[!] ОШИБКА: 'world_bible' в предыдущем состоянии пуста. Невозможно создать сиквел.")
            sys.exit(1)
        print("  ✓ 'Библия Мира' Книги 1 загружена для обновления.")

        # 3. Перенос Qdrant (Критично для контроля повторов)
        self.new_state['qdrant_collection_name'] = previous_state.get('qdrant_collection_name')
        if not self.new_state['qdrant_collection_name']:
            print("[!] ВНИМАНИЕ: Не найдено имя коллекции Qdrant. Будет создана новая.")
            self.new_state['qdrant_collection_name'] = f"novel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            print(f"  ✓ Коллекция Qdrant '{self.new_state['qdrant_collection_name']}' будет переиспользована.")

        # 4. Перенос Резюме Глав (Критично для контекста)
        self.new_state['chapter_summaries'] = previous_state.get('chapter_summaries', [])
        print(f"  ✓ {len(self.new_state['chapter_summaries'])} резюме глав из Книги 1 перенесены.")

        # 5. Очистка
        self.new_state['final_chapters_text'] = []
        self.new_state['transient_data'] = {}
        self.new_state['steps'] = [] # Будут созданы в конце

        print("  ✓ Базовое состояние для сиквела подготовлено.")

    def _initialize_models(self):
        """Инициализирует модели, включая 'world_model' с контекстом Книги 1."""
        planner_context = f"{SYSTEM_OVERRIDE}\n{PLANNER_ROLE}\n{CONCEPTS}\n{ANTI_PLEASING}"

        self.base_model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=planner_context,
            safety_settings=SAFETY_SETTINGS,
        )

        # Сразу создаем world_model с "Библией" Книги 1
        self._create_world_model()

    def _create_world_model(self):
        """Создает модель с system instruction, содержащей всю 'Библию Мира'."""
        world_context = load_prompt(
            'global_world_model',
            world_bible=self.new_state['world_bible'],
            ANTI_PLEASING=ANTI_PLEASING,
            OVERRIDE=SYSTEM_OVERRIDE,
            CONCEPTS=CONCEPTS
        )
        self.world_model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=world_context,
            safety_settings=SAFETY_SETTINGS,
        )
        print("✓ Создана модель с контекстом 'Библии Мира' (Книга 1)")

    def _call_gemini(self, prompt_text, temperature=0.8, use_world_model=False, response_schema=None):
        """Надежный вызов API."""
        print(f"  > Отправка запроса в Gemini (model: {'world' if use_world_model else 'base'})...")

        model = self.world_model if use_world_model and self.world_model else self.base_model

        try:
            config_args = {
                "temperature": temperature,
                "max_output_tokens": 100000,
            }
            if response_schema:
                config_args['response_mime_type'] = 'application/json'
                config_args['response_schema'] = response_schema

            generation_config = genai.types.GenerationConfig(**config_args)
            print(prompt_text)
            response = model.generate_content(
                prompt_text,
                generation_config=generation_config,
                safety_settings=SAFETY_SETTINGS
            )
            print(response.text)

            if response.usage_metadata:
                model_key = model.model_name
                if model_key not in STATS:
                    STATS[model_key] = {'input_tokens': 0, 'output_tokens': 0, 'calls': 0}
                STATS[model_key]['input_tokens'] += response.usage_metadata.prompt_token_count
                STATS[model_key]['output_tokens'] += response.usage_metadata.candidates_token_count
                STATS[model_key]['calls'] += 1
                print(f"   [INFO] Токены (в/в/всего): {response.usage_metadata.prompt_token_count}/{response.usage_metadata.candidates_token_count}/{response.usage_metadata.total_token_count}")

            return response.text

        except Exception as e:
            print(f"   ! Ошибка API: {e}. Повторная попытка через 10 сек...")
            time.sleep(10)
            return self._call_gemini(prompt_text, temperature, use_world_model, response_schema)

    def run(self, sequel_synopsis, num_chapters):
        """Выполняет шаги по планированию сиквела."""
        wb = self.new_state['world_bible'] # ссылка для удобства

        old_summaries = self.new_state['chapter_summaries'] # Сохраняем полные саммари Книги 1
        old_state = self.new_state['world_state'] # Сохраняем полный стейт Книги 1

        # "ШАГ 0": КОМПРЕССИЯ КОНТЕКСТА ---
        print("\n--- [Шаг 0.1] Сжатие резюме Книги 1 ---")
        prompt_0_1 = load_prompt('sequel_0_1_book_summary', chapter_summaries=json.dumps(old_summaries, ensure_ascii=False))
        global_summary = self._call_gemini(prompt_0_1, temperature=0.5, use_world_model=False)
        # Заменяем список из 25 резюме ОДНИМ сжатым резюме.
        self.new_state['chapter_summaries'] = [global_summary] + [f"\nКОНЕЦ ПРЕДЫДУЩЕЙ КНИГИ: [LAST_CHAPTER]{old_summaries[-1]}[/LAST_CHAPTER]"]
        print("  ✓ Резюме глав Книги 1 сжато в единый 'Глобальный Контекст'.")

        print("\n--- [Шаг 0.2] Очистка (Pruning) Состояния Мира ---")
        prompt_0_2 = load_prompt('sequel_0_2_prune_state', world_state_json=json.dumps(old_state, ensure_ascii=False))
        pruned_state_json = self._call_gemini(prompt_0_2, temperature=0.3, use_world_model=False)
        # Заменяем массивный world_state на его "чистую" версию

        self.new_state['world_state'] = robust_json_parser(pruned_state_json)
        print("  ✓ 'Состояние Мира' очищено от временных и ненужных данных.")

        # --- ОБНОВЛЕНИЕ "БИБЛИИ МИРА" ДЛЯ СИКВЕЛА ---
        print("\n--- [Шаг 1.1] Обновление Анализа ---")
        prompt_1_1 = load_prompt('sequel_1_1_update_analysis', world_bible=wb, synopsis=sequel_synopsis, NUM_CHAPTERS=num_chapters)
        wb['analysis'] = json.loads(self._call_gemini(prompt_1_1, temperature=0.8, use_world_model=True, response_schema=SCHEMA_ANALYSIS))
        print("  ✓ Анализ сюжета Книги 2 создан.")

        print("\n--- [Шаг 1.2] Обновление Персонажей ---")
        prompt_1_2 = load_prompt('sequel_1_2_update_characters', world_bible=wb, synopsis=sequel_synopsis, NUM_CHAPTERS=num_chapters)
        wb['characters'] = self._call_gemini(prompt_1_2, temperature=0.9, use_world_model=True)
        print("  ✓ Анкеты персонажей обновлены/дополнены.")

        print("\n--- [Шаг 1.3] Обновление Сеттинга ---")
        prompt_1_3 = load_prompt('sequel_1_3_update_setting', world_bible=wb, synopsis=sequel_synopsis)
        wb['setting'] = wb['setting'] + '\n' + self._call_gemini(prompt_1_3, temperature=0.7, use_world_model=True)
        print("  ✓ Сеттинг обновлен (добавлены новые локации/изменения).")

        print("\n--- [Шаг 1.4] Обновление Речевых Профилей ---")
        prompt_1_4 = load_prompt('sequel_1_4_update_voice_profiles', world_bible=wb)
        wb['voice_profiles'] = self._call_gemini(prompt_1_4, temperature=0.9, use_world_model=True)
        print("  ✓ Речевые профили обновлены/дополнены.")

        # Шаги 1.5 и 1.6 (Стиль) намеренно пропускаем - стиль цикла должен быть единым.
        print("\n--- [Шаги 1.5, 1.6] Сохранение Стиля ---")
        print("  ✓ Стиль книги и тон повествования из Книги 1 сохранены для целостности цикла.")

        # --- ГЕНЕРАЦИЯ НОВОГО ПЛАНА (по оригинальной логике) ---

        # Перед генерацией плана нам нужно, чтобы 'world_model' знала об
        # обновленных персонажах, сеттинге и т.д.
        print("\n--- [Шаг 1.10*] Пересоздание Модели Мира (с новыми данными) ---")
        self._create_world_model()

        print("\n--- [Шаг 1.7] Генерация Плана Глав (Книга 2) ---")
        prompt_1_7 = load_prompt('prompt_1_7_plan', world_bible=wb, json_out=JSON_OUT, NUM_CHAPTERS=num_chapters)
        scene_plan = self._call_gemini(prompt_1_7, temperature=0.8, use_world_model=True, response_schema=PLAN_SCHEMA)
        wb['chapters'] = json.loads(scene_plan)['chapters']
        print(f"  ✓ План на {len(wb['chapters'])} глав для Книги 2 сгенерирован.")

        print("\n--- [Шаг 1.8] Критика Плана ---")
        full_plot_json = json.dumps(wb['chapters'], indent=2, ensure_ascii=False)
        prompt_1_8 = load_prompt('prompt_1_8_critique_plan', world_bible=wb, full_plot_json=full_plot_json, NUM_CHAPTERS=num_chapters)
        plot_critique = self._call_gemini(prompt_1_8, temperature=0.6, use_world_model=True)
        print("  ✓ Критика плана получена.")

        print("\n--- [Шаг 1.9] Правка Плана ---")
        prompt_1_9 = load_prompt('prompt_1_9_refactor_plan', world_bible=wb, plot_critique=plot_critique, full_plot_json=full_plot_json)
        edited_scene_plan_json = self._call_gemini(prompt_1_9, temperature=0.7, use_world_model=True, response_schema=PLAN_SCHEMA)
        wb['chapters'] = json.loads(edited_scene_plan_json)['chapters']
        print("  ✓ План глав Книги 2 исправлен и финализирован.")

        # --- ФОРМИРОВАНИЕ "ЗАВЕРШЕННЫХ" ШАГОВ ---
        print("\n--- [Финализация] Создание списка выполненных шагов 'Foundation' ---")
        steps = [
            Step(name="1.1 Анализ", handler_name="step_foundation_1_1_analysis", status='done'),
            Step(name="1.2 Персонажи", handler_name="step_foundation_1_2_characters", status='done'),
            Step(name="1.3 Мир", handler_name="step_foundation_1_3_setting", status='done'),
            Step(name="1.4 Речевые профили", handler_name="step_foundation_1_4_voice_profiles", status='done'),
            Step(name="1.5 Стиль книги", handler_name="step_foundation_1_5_book_style", status='done'),
            Step(name="1.6 Стиль и тон", handler_name="step_foundation_1_6_style", status='done'),
            Step(name="1.7 План", handler_name="step_foundation_1_7_plan", status='done'),
            Step(name="1.8 Критика плана", handler_name="step_foundation_1_8_critique_plan", status='done'),
            Step(name="1.9 Правка плана", handler_name="step_foundation_1_9_refactor_plan", status='done'),
            Step(name="1.10 Создание модели мира", handler_name="step_foundation_1_10_create_world_model", status='done'),
        ]
        self.new_state['steps'] = [asdict(s) for s in steps]
        print("  ✓ Все шаги 'Foundation' отмечены как 'done'.")

        return self.new_state


if __name__ == "__main__":
    if API_KEY == 'ВАШ_API_КЛЮЧ':
        print("Ошибка: Пожалуйста, вставьте ваш API ключ в переменную API_KEY.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Подготовка файла состояния для генерации сиквела романа.")
    parser.add_argument("--input", required=True, help="Файл состояния предыдущей книги (например, novel_project_state.json)")
    parser.add_argument("--output", required=True, help="Имя нового файла состояния для сиквела (например, sequel_project_state.json)")
    parser.add_argument("--synopsis", required=True, help="Текстовый файл с синопсисом для *новой* книги (сиквела).")
    parser.add_argument("--chapters", type=int, default=4, help="Желаемое количество глав в сиквеле (по умолчанию 4).")

    args = parser.parse_args()

    # --- 1. Загрузка данных ---
    print(f"Загрузка предыдущего состояния из: {args.input}")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            previous_state = json.load(f)
    except FileNotFoundError:
        print(f"[!] ОШИБКА: Файл '{args.input}' не найден.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[!] ОШИБКА: Файл '{args.input}' не является корректным JSON.")
        sys.exit(1)

    print(f"Загрузка синопсиса сиквела из: {args.synopsis}")
    try:
        with open(args.synopsis, 'r', encoding='utf-8') as f:
            synopsis_text = f.read()
    except FileNotFoundError:
        print(f"[!] ОШИБКА: Файл синопсиса '{args.synopsis}' не найден.")
        sys.exit(1)

    # --- 2. Инициализация и выполнение ---
    print("Запуск планировщика сиквела...")
    try:
        planner = SequelPlanner(api_key=API_KEY, previous_state=previous_state)
        new_project_state = planner.run(sequel_synopsis=synopsis_text, num_chapters=args.chapters)

        # --- 3. Сохранение результата ---
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(new_project_state, f, indent=2, ensure_ascii=False)

        print(f"\n🎉 УСПЕХ! Новый файл состояния '{args.output}' готов.")
        print("Теперь вы можете запустить ваш оригинальный 'novel_generator.py',")
        print(f"указав ему этот файл, чтобы сгенерировать Книгу 2.")
        print(f"\nИспользовано токенов (всего): {STATS}")

    except Exception as e:
        print(f"\n[!] КРИТИЧЕСКАЯ ГЛОБАЛЬНАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
