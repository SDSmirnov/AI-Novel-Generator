import google.generativeai as genai
import json
import numpy as np
import os
import random
import re
import sys
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from typing import List, Tuple, Dict, Any, Optional


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
    name: str                 # Человекочитаемое имя (e.g., "1.1 Анализ")
    handler_name: str         # Имя метода в NovelGenerator (e.g., "step_foundation_1_1_analysis")
    status: str = 'planned'   # 'planned', 'started', 'done'

def load_prompt(name: str, **kwargs) -> str:
    template = open(f'prompts/{name}.md', 'r').read()
    def evaluator(match):
        expression = match.group(1)
        try:
            return str(eval(expression, kwargs))
        except Exception as e:
            print(f"[!] Ошибка вычисления в промпте '{name}': {expression} -> {e}")
            return f"{{EVAL_ERROR: {expression}}}"
    return re.sub(r"\{(.*?)\}", evaluator, template)

def log_chapter(kind, number, text):
    if not os.path.exists('generated_chapters'):
        os.makedirs('generated_chapters')
    with open(f"generated_chapters/chapter_{kind}_{number:02}.md", "w", encoding="utf-8") as f:
        f.write(text)

API_KEY = os.getenv('AI_API_KEY', 'ВАШ_API_КЛЮЧ')

ALLOW_MATURE_LANGUAGE = True
MODEL_NAME = "gemini-2.5-pro" # "gemini-3-pro-preview"

DEFAULT_CONCEPTS = load_prompt('global_concepts')
CONCEPTS = open('concepts.txt', 'r').read() if os.path.isfile('concepts.txt') else DEFAULT_CONCEPTS

DEFAULT_OVERRIDE = load_prompt('global_override')
SYSTEM_OVERRIDE = open('override.txt', 'r').read() if os.path.isfile('override.txt') else DEFAULT_OVERRIDE

ANTI_PLEASING = load_prompt('global_anti_pleasing')

PLANNER_ROLE = "Твоя роль: архитектор-планировщик захватывающей художественной книги на русском языке."

SYNOPSIS = open('synopsis.txt', 'r').read()
START_INFO = open('start-info.txt', 'r').read() if os.path.isfile('start-info.txt') else ''

NUM_CHAPTERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

MAX_WORKERS = 5
MAX_CLUSTERS = 8

# --- КОНЕЦ КОНФИГУРАЦИИ ---

SAFETY_SETTINGS = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]

JSON_REPEATER = load_prompt('schema_repeater')
REPEATER_PROMPT = load_prompt('global_repeater', ANTI_PLEASING=ANTI_PLEASING, json_repeater=JSON_REPEATER)

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

JSON_STATE_STRUCT = '''
{
    "chapter_summary": "Перечисли ВСЕ основные события, произошедшие в ЭТОЙ главе с указанием места, даты и времени: кто, что, где, когда, как...",
    "world_state_updates": {
        "chapter_end_date_time": "YYYY-MM-DD HH:MM",
        "characters": {
            "character_name_1": {
                "location": "Новая локация",
                "physical_condition": "Детальное и безжалостно реалистичное описание состояния. Укажи не просто 'ранен', а характер раны, степень боли (1-10), ограничения в движении. Пример: 'Сквозное ранение стрелой в левом плече, края прижжены. Постоянная ноющая боль (7/10), усиливающаяся при движении. Рука почти не действует, может лишь придерживать предметы. Начальные признаки лихорадки.'",
                "health_issues": [
                    {"what": "выбил палец ", "when": "глава 3, дата YYYY-MM-DD", "estimated_recovery": "дата заживления, восстановления"},
                    ....
                ],
                "psychological_condition": ["новое состояние"],
                "inventory_changes": {"add": ["новый предмет"], "remove": ["старый предмет"], "keep":["имеющийся предмет"]},
                "knowledge_update": "Что нового и важного узнал персонаж",
                "familiar_people": [
                    {"person": "Иия знакомого", "since": "Дата знакомства", "relationships": "Описание отношений", "last_contact": "дата"},
                    {"person": "Иия нового знакомого", "since": "Дата знакомства", "relationships": "Описание отношений", "last_contact": "дата"},
                    // Для всех новых знакомств
                ]
            },
            "character_name_2": {
                // ... аналогичные изменения ...
            }
        },
        "items": {
            "item_name_1": {
                "location": "Новое местоположение или владелец",
                "status": "Новый статус (например, 'использован')"
            }
        },
        "locations": {
            "location_name_1": {
                "status": "Новый статус (например, 'частично разрушена')"
            }
        }
    }
}
'''

TECHNIQUES_POOL = [
    "Начни сцену с неожиданной детали или действия",
    "Включи диалог с подтекстом - персонажи говорят не то, что думают",
    "Покажи противоречие между мыслями и действиями героя",
    "Добавь сенсорную деталь, которая усилит атмосферу",
    "Дай персонажу совершить иррациональный поступок",
    "Включи момент неловкой паузы или недопонимания",
    "Покажи, как среда влияет на поведение персонажа",
    "Используй контраст (тепло/холод, свет/тень, тишина/шум)",
    "Покажи персонажа через его отношение к мелочам",
    "Включи момент, когда план персонажа срывается",
    "Покажи реакцию окружающих (восхищение, зависть, страх) на действия ГГ",
]

CHAOS_ELEMENTS = [
    "Помни: люди часто делают глупости без причины",
    "Не все детали должны работать на сюжет - иногда жизнь просто происходит",
    "Позволь персонажам отвлекаться на мелочи",
    "Пусть кто-то скажет что-то неуместное или забудет важное",
    "Добавь момент, когда план идет не так, как задумано",
    "Персонаж делает что-то нелогичное из-за эмоций",
    "Кто-то забывает важную деталь",
    "План срывается из-за человеческой ошибки под давлением (кто-то споткнулся, уронил ключевой предмет, неправильно понял команду)",
    "Персонаж принимает неверное решение, основанное на неполной или ложной информации",
    "Эмоциональная реакция (гнев, страх) заставляет персонажа отклониться от плана с негативными последствиями",
    "Второстепенный персонаж неожиданно влияет на события",
    "Важный предмет потерян в самый неподходящий момент",
    "Персонаж неправильно понял ключевую информацию",
    "Союзник случайно мешает плану героя",
    "Технология/оружие работает не так, как ожидалось",
]

STATS = {} # 'models/gemini-2.5-pro': {'input_tokens': 0, 'output_tokens': 0, 'calls': 0},

@dataclass
class TextChunk:
    chunk_id: int
    text: str
    chapter: int
    paragraph_id: int

@dataclass
class SimilarityCluster:
    cluster_id: int
    original: TextChunk
    similar_chunks: List[Tuple[TextChunk, float]]

@dataclass
class AnalysisResult:
    cluster: SimilarityCluster
    status: str
    motive: str
    diagnosis: str
    recommendation: str
    confidence: float

def generate_embedding(text: str):
    result = genai.embed_content(
        model='models/gemini-embedding-001',
        content=text,
        task_type="SEMANTIC_SIMILARITY",
        output_dimensionality=768
    )
    return result['embedding']

import random
import re

class RhythmEngine:
    def __init__(self):
        self.patterns = {
            "action": {
                "weights": ["S", "M", "L"],
                "probs": [0.7, 0.25, 0.05], # Много коротких, мало длинных
                "description": "ACTION STACCATO: Быстрый темп, рубленые фразы, минимум причастных оборотов."
            },
            "dialogue": {
                "weights": ["S", "M", "L"],
                "probs": [0.4, 0.5, 0.1],
                "description": "DIALOGUE FLOW: Обмен репликами, паузы, реакции."
            },
            "description": {
                "weights": ["S", "M", "L"],
                "probs": [0.1, 0.4, 0.5], # Длинные, текучие предложения
                "description": "ATMOSPHERIC FLOW: Плавный, обволакивающий ритм, сложные конструкции."
            },
            "tension": {
                "weights": ["S", "M", "L"],
                "probs": [0.5, 0.1, 0.4], # Контраст: очень короткие и очень длинные
                "description": "TENSION SPIKES: Резкие перепады от длинных нагнетаний к коротким ударам."
            },
            "balanced": {
                "weights": ["S", "M", "L"],
                "probs": [0.3, 0.5, 0.2],
                "description": "NARRATIVE BALANCE: Спокойное повествование."
            }
        }

        # Ключевые слова для авто-определения ритма (резервный вариант)
        self.triggers = {
            "action": ["бой", "удар", "бег", "выстрел", "кровь", "схватка", "взрыв", "крик", "погоня", "драка"],
            "dialogue": ["разговор", "беседа", "обсужд", "спор", "допрос", "признание", "шепт", "голос"],
            "description": ["осмотр", "вид", "пейзаж", "комната", "тишина", "атмосфера", "воспоминание", "мысли"],
            "tension": ["страх", "темнота", "шаги", "ожидание", "ужас", "красться", "слеж"]
        }

    def detect_mode(self, scene_description: str) -> str:
        """Резервный метод: определяет режим по ключевым словам."""
        scene_description = scene_description.lower()
        scores = {k: 0 for k in self.triggers.keys()}

        for mode, keywords in self.triggers.items():
            for word in keywords:
                if word in scene_description:
                    scores[mode] += 1

        if max(scores.values()) == 0:
            return "balanced"
        return max(scores, key=scores.get)

    def _get_pattern_sequence(self, mode, length=15):
        """Генерирует сырую последовательность [S]->[M]..."""
        config = self.patterns.get(mode, self.patterns["balanced"])
        sequence = random.choices(
            config["weights"],
            weights=config["probs"],
            k=length
        )
        return " -> ".join([f"[{x}]" for x in sequence])

    def generate_rhythm_block(self, scene_text="", mode=None, length=20):
        """Для совместимости со старым кодом (один блок на все)."""
        if not mode:
            mode = self.detect_mode(scene_text)

        config = self.patterns.get(mode, self.patterns["balanced"])
        rhythm_map = self._get_pattern_sequence(mode, length)

        return f"""
### 🎹 СИНТАКСИЧЕСКИЙ КОНТРОЛЛЕР (РЕЖИМ: {mode.upper()})
Твоя карта ритма: {rhythm_map}
Важно: {config['description']}
"""

    def generate_chapter_map(self, scenes_data: list) -> str:
        """
        Генерирует сложную инструкцию для всей главы на основе анализа flash-lite.
        scenes_data: список словарей [{"num": 1, "mode": "action"}, ...]
        """
        output = ["### 🎹 ДИНАМИЧЕСКАЯ ПАРТИТУРА ГЛАВЫ"]
        output.append("Ты обязан менять ритм повествования от сцены к сцене согласно этой карте:")

        output.append("\nЛЕГЕНДА ДЛИНЫ ПРЕДЛОЖЕНИЙ:")
        output.append("[S] = КОРОТКОЕ (3-6 слов). Удар. Факт.")
        output.append("[M] = СРЕДНЕЕ (7-15 слов). Обычное действие.")
        output.append("[L] = ДЛИННОЕ (16+ слов). Атмосфера, мысли.")

        for scene in scenes_data:
            mode = scene.get('mode', 'balanced').lower()
            # Если модель вернула что-то странное, откатываемся на balanced
            if mode not in self.patterns:
                mode = 'balanced'

            config = self.patterns[mode]
            # Генерируем уникальный паттерн для этой сцены
            pattern = self._get_pattern_sequence(mode, length=12)

            scene_block = f"""
**СЦЕНА {scene.get('num', '?')}: {scene.get('title', 'Сцена')}**
- Режим: {mode.upper()}
- Задача ритма: {config['description']}
- Твой паттерн: {pattern} ... (повторять динамику)
"""
            output.append(scene_block)

        output.append("\nВАЖНО: Следи за переключением ритма при переходе между сценами!")
        return "\n".join(output)


class StyleRepeatingChecker:
    def __init__(self):
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.novel_collection_name: str | None = None # Будет установлено извне
        self.vector_size = 768
        self.chunk_id_counter = 0
        self._collection_checked = False # Флаг, что мы проверили/создали коллекцию

    def set_collection_name(self, name: str):
        self.novel_collection_name = name
        print(f"Qdrant: Установлено имя коллекции: {self.novel_collection_name}")

    def _ensure_collection_exists(self):
        if self._collection_checked:
            return
        if not self.novel_collection_name:
            raise ValueError("Имя коллекции Qdrant не установлено. Вызовите set_collection_name().")

        try:
            # Проверяем, существует ли коллекция
            self.qdrant_client.get_collection(collection_name=self.novel_collection_name)
            print(f"✓ Qdrant: Успешно подключено к существующей коллекции: {self.novel_collection_name}")
        except Exception:
            # Не существует, создаем
            print(f"Qdrant: Коллекция {self.novel_collection_name} не найдена. Создание новой...")
            self._create_novel_collection()

        self._collection_checked = True

    def _create_novel_collection(self):
        try:
            self.qdrant_client.recreate_collection(
                collection_name=self.novel_collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            print(f"✓ Qdrant: Коллекция {self.novel_collection_name} успешно создана.")
        except Exception as e:
            print(f"Не удалось создать коллекцию в Qdrant: {e}")
            raise

    def _sent_tokenize_with_punctuation(self, text: str) -> List[str]:
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_text(self, text: str, chapter_num: int) -> List[TextChunk]:
        chunks = []
        paragraphs = re.split(r'\n+', text)
        para_id = 0
        joiner = ""
        for paragraph in paragraphs:
            paragraph = joiner + "\n" + paragraph if joiner else paragraph
            joiner = ""
            if not paragraph.strip():
                continue

            if len(paragraph.strip()) < 20 or len(paragraph.strip().split(' ')) < 3:
                joiner = paragraph.strip()
                continue

            chunks.append(TextChunk(
                chunk_id=self.chunk_id_counter,
                text=paragraph.strip(),
                chapter=chapter_num,
                paragraph_id=para_id
            ))
            self.chunk_id_counter += 1

            # --- 2. Чанки по принципу скользящего окна ---
            sentences = self._sent_tokenize_with_punctuation(paragraph.strip())

            window_size = 3
            if len(sentences) >= window_size:
                for i in range(len(sentences) - window_size + 1):
                    window = sentences[i : i + window_size]
                    chunk_text = " ".join(window)
                    chunks.append(TextChunk(
                        chunk_id=self.chunk_id_counter,
                        text=chunk_text,
                        chapter=chapter_num,
                        paragraph_id=para_id
                    ))
                    self.chunk_id_counter += 1
            para_id += 1
        print(f"\n---СОЗДАНО ЧАНКОВ {len(chunks)}")

        return chunks

    def _embed_chunks(self, chunks: List[TextChunk]) -> Dict[int, List[float]]:
        embeddings = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {executor.submit(generate_embedding, chunk.text): chunk for chunk in chunks}
            for future in as_completed(future_to_chunk):
                try:
                    chunk = future_to_chunk[future]
                    embeddings[chunk.chunk_id] = future.result()
                    print('*', end='')
                except Exception as e:
                    print(f"Критическая ошибка вне потока при обработке чанка {chunk.chunk_id}: {e}")
                    pass
        print(f'\nСозданно эмбеддингов: {len(embeddings)}')
        return embeddings

    def _add_chapter_to_db(self, chunks: List[TextChunk], chapter_number: int) -> Dict[int, List[float]]:
        """--- ИЗМЕНЕНО: Добавлена проверка коллекции ---"""
        self._ensure_collection_exists() # Гарантируем, что коллекция есть

        embeddings = self._embed_chunks(chunks)
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=embeddings[chunk.chunk_id],
                payload=asdict(chunk)
            ) for chunk in chunks if chunk.chunk_id in embeddings
        ]
        if points:
            self.qdrant_client.upsert(
                collection_name=self.novel_collection_name,
                points=points,
                wait=True
            )
        return embeddings

    def _analyze_one_repetition_cluster(self, cluster: SimilarityCluster) -> AnalysisResult | None:
        try:
            fragments_text = f"ОРИГИНАЛЬНЫЙ ФРАГМЕНТ (из текущего черновика, Глава {cluster.original.chapter}):\n"
            fragments_text += f'"{cluster.original.text}"\n\n'
            fragments_text += "ПОХОЖИЕ ФРАГМЕНТЫ ИЗ ПРЕДЫДУЩИХ ГЛАВ:\n"

            for similar_chunk, score in cluster.similar_chunks:
                fragments_text += f"- Глава {similar_chunk.chapter} (схожесть: {score:.2f}):\n"
                fragments_text += f'  "{similar_chunk.text}"\n\n'

            prompt = f"Проанализируйте следующие фрагменты на предмет стилистического самоповтора:\n\n{fragments_text}\n\nДай рекомендации только для текущей Главы {cluster.original.chapter}"

            analysis_model = genai.GenerativeModel(
                model_name="gemini-flash-lite-latest",
                system_instruction=REPEATER_PROMPT,
                safety_settings=SAFETY_SETTINGS,
            )

            RESPONSE_SCHEMA = {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["OK", "SLIGHT_REPETITION", "STRONG_CLICHE"]},
                    "motive": {"type": "string"},
                    "diagnosis": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["status", "motive", "diagnosis", "recommendation", "confidence"]
            }
            print(prompt)

            response = analysis_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    response_mime_type='application/json',
                    response_schema=RESPONSE_SCHEMA,
                    max_output_tokens=400000,
                )
            )

            analysis_data = robust_json_parser(response.text)
            print(response.text)
            result = AnalysisResult(
                cluster=cluster,
                status=analysis_data.get('status', 'OK'),
                motive=analysis_data.get('motive', 'N/A'),
                diagnosis=analysis_data.get('diagnosis', 'N/A'),
                recommendation=analysis_data.get('recommendation', 'N/A'),
                confidence=float(analysis_data.get('confidence', 0.0))
            )
            return result

        except Exception as e:
            print(f"[ОШИБКА] Не удалось проанализировать кластер {cluster.cluster_id}: {e}")
            return None

    def check_cross_chapter_repetitions(self, draft_text: str, current_chapter_num: int) -> str:
        """--- ИЗМЕНЕНО: Добавлена проверка коллекции ---"""
        self._ensure_collection_exists() # Гарантируем, что коллекция есть

        print(f"Запуск проверки на межглавные повторы для Главы {current_chapter_num}...")
        draft_chunks = self._chunk_text(draft_text, current_chapter_num)
        if not draft_chunks:
            print("В черновике не найдено чанков для анализа.")
            return ""

        draft_embeddings = self._add_chapter_to_db(draft_chunks, current_chapter_num)
        all_clusters = []
        processed_chunk_ids = set() # Чтобы не проверять один и тот же чанк дважды

        for chunk in draft_chunks:
            if chunk.chunk_id in processed_chunk_ids or chunk.chunk_id not in draft_embeddings:
                continue

            search_results = self.qdrant_client.search(
                collection_name=self.novel_collection_name,
                query_vector=draft_embeddings[chunk.chunk_id],
                limit=20,
                score_threshold=0.85
            )

            if search_results:
                similar_chunks_data = []
                used_para_chapters = set()
                for hit in search_results:
                    if hit.id == chunk.chunk_id:
                        continue

                    if len(similar_chunks_data) > 5:
                        break

                    if hit.id in processed_chunk_ids:
                        continue

                    hit_chapter = hit.payload.get('chapter')
                    hit_para = hit.payload.get('paragraph_id')

                    if hit_para == chunk.paragraph_id and hit_chapter == chunk.chapter:
                        continue

                    ch_p = f"{hit_chapter}:{hit_para}"
                    if ch_p in used_para_chapters:
                        continue

                    processed_chunk_ids.add(hit.id)
                    used_para_chapters.add(ch_p)
                    prev_chunk = TextChunk(**hit.payload)
                    similar_chunks_data.append((prev_chunk, hit.score))

                cluster = SimilarityCluster(
                    cluster_id=len(all_clusters),
                    original=chunk,
                    similar_chunks=similar_chunks_data
                )
                all_clusters.append(cluster)
                processed_chunk_ids.add(chunk.chunk_id)

        if not all_clusters:
            print("Критических межглавных повторов не найдено.")
            return ""

        analysis_results = []
        all_clusters.sort(key=lambda c: c.similar_chunks[0][1] if c.similar_chunks else 0, reverse=True)

        print(f"Найдено {len(all_clusters)} потенциальных кластеров повторов. Анализируем до {MAX_CLUSTERS}...")
        for cluster in all_clusters[:MAX_CLUSTERS]:
            result = self._analyze_one_repetition_cluster(cluster)
            if result:
                analysis_results.append(result)

        critique_text = ""
        # Фильтруем результаты, чтобы оставить только реальные проблемы
        problem_results = [r for r in analysis_results if r.status != 'OK']

        if problem_results:
            critique_text = "### 🧐 Критический анализ на межглавные повторы:\n\n"
            for result in problem_results:
                status_emoji = '🔴' if result.status == 'STRONG_CLICHE' else '🟡'
                critique_text += f"{status_emoji} **Проблема ({result.status}, уверенность {result.confidence:.0%}):** {result.motive}\n"
                critique_text += f"- **Диагноз:** {result.diagnosis}\n"
                critique_text += f"- **Рекомендация:** {result.recommendation}\n"
                critique_text += f"- **Пример в этой главе:** `...{result.cluster.original.text[:100]}...`\n\n"
            print(f"Найдено {len(problem_results)} стилистических проблем.")
        else:
            print("Анализ завершен, стилистических проблем не выявлено.")

        log_chapter('repeats', current_chapter_num, critique_text)
        return critique_text


class NovelGenerator:
    """
    --- ИЗМЕНЕНО ---
    Класс теперь не содержит монолитных методов create_foundation и generate_novel.
    Вместо этого он предоставляет набор методов-обработчиков (step_...)
    для управления из NovelGenerationProject.
    """
    def __init__(self, api_key, model_name=MODEL_NAME):
        genai.configure(api_key=api_key)
        planner_context = f"""
        {SYSTEM_OVERRIDE}

        {PLANNER_ROLE}

        {CONCEPTS}

        {ANTI_PLEASING}
        """
        self.base_model = genai.GenerativeModel(
            model_name='gemini-2.5-pro',
            system_instruction=planner_context,
            safety_settings=SAFETY_SETTINGS,
        )
        self.fast_model = genai.GenerativeModel(
            model_name='gemini-flash-lite-latest', # Используем актуальный ID
            system_instruction=planner_context,
            safety_settings=SAFETY_SETTINGS
        )
        self.world_model = None
        self.pro_model = None
        self.world_bible = {}  # Управляется NovelGenerationProject
        self.scenes = []       # Управляется NovelGenerationProject
        self.world_state = {}  # Управляется NovelGenerationProject
        self.clusterer = StyleRepeatingChecker()
        self.rhythm_engine = RhythmEngine()

    def _create_world_model(self):
        """Создает модель с system instruction, содержащей всю 'Библию Мира'."""
        wb = json.loads(json.dumps(self.world_bible))
        wb['analysis'].pop('detailed_plot_plan', '')
        world_context = load_prompt('global_world_model', world_bible=wb, ANTI_PLEASING=ANTI_PLEASING, OVERRIDE=SYSTEM_OVERRIDE, CONCEPTS=CONCEPTS)
        log_chapter('bible', 0, world_context)
        self.world_model = genai.GenerativeModel(
            model_name='gemini-2.5-pro',
            system_instruction=world_context,
            safety_settings=SAFETY_SETTINGS,
        )
        self.pro_model = genai.GenerativeModel(
            model_name='gemini-2.5-pro',
            system_instruction=world_context,
            safety_settings=SAFETY_SETTINGS,
        )
        print("✓ Создана модель с контекстом 'Библии Мира'")

    def _call_gemini(self, prompt_text, attempt_count=3, temperature=0.8, top_p=0.95, top_k=40, use_world_model=False, response_schema=None, use_pro=False):
        """Надежный вызов API с повторными попытками."""
        print(f"  > Отправка запроса в Gemini...")

        if use_pro:
            model = self.pro_model
        elif use_world_model:
            model = self.world_model
        else:
            model = self.base_model

        if not model:
             print("   ! ОШИБКА: Модель не инициализирована. (Возможно, 'world_model' была вызвана до 'create_foundation'?)")
             if use_world_model or use_pro:
                 print("   ! ОТКАТ: Используется 'base_model'.")
                 model = self.base_model
             else:
                 raise ValueError("Базовая модель не инициализирована.")
        try:
            print(prompt_text)
            if response_schema:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=100000,
                    response_mime_type='application/json',
                    response_schema=response_schema,
                )
            else:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=100000,
                )
            response = model.generate_content(prompt_text, generation_config=generation_config, safety_settings=SAFETY_SETTINGS,)
            print(response.text)

            # gather API stats
            if response.usage_metadata:
                if model.model_name not in STATS:
                    STATS[model.model_name] = {'input_tokens': 0, 'output_tokens': 0, 'calls': 0}

                STATS[model.model_name]['input_tokens'] += response.usage_metadata.prompt_token_count
                STATS[model.model_name]['output_tokens'] += response.usage_metadata.candidates_token_count
                STATS[model.model_name]['calls'] += 1

                print(f"   [INFO] Входные токены: {response.usage_metadata.prompt_token_count}")
                print(f"   [INFO] Выходные токены: {response.usage_metadata.candidates_token_count}")
                print(f"   [INFO] Всего токенов: {response.usage_metadata.total_token_count}")
                print(f"   [INFO] {STATS}")
            # --- КОНЕЦ БЛОКА ПОДСЧЕТА ---
            return response.text
        except Exception as e:
            print(f"   ! Ошибка API: {e}. Попытка {attempt_count}...")
            # Добавлена задержка, чтобы не превышать лимиты запросов
            time.sleep(1)
            if attempt_count > 0:
                time.sleep(45)
                return self._call_gemini(prompt_text, attempt_count - 1, temperature, top_p, top_k, use_world_model, response_schema, use_pro)
            else:
                print("   ! Не удалось выполнить запрос к API после нескольких попыток.")
                return None

    def _update_world_state(self, response_text, chapter_num):
        print("  > Обновление состояния мира...")
        try:
            json_match = re.search(r'---JSON_STATE_START---(.*)---JSON_STATE_END---', response_text, re.DOTALL)
            if not json_match:
                print("   ! Не найден JSON блок в ответе для обновления состояния.")
                return None # Возвращаем только текстовое резюме, если оно есть

            json_data = robust_json_parser(json_match.group(1).strip())
            updates = json_data.get('world_state_updates', {})

            def deep_update(source, overrides):
                for key, value in overrides.items():
                    dkey = key
                    if isinstance(value, dict) and dkey in source:
                        source[dkey] = deep_update(source.get(dkey, {}), value)
                    else:
                        source[dkey] = value
                return source

            self.world_state = deep_update(self.world_state, updates)
            log_chapter('state', chapter_num, json.dumps(self.world_state, indent=2, ensure_ascii=False))
            print("  ✓ Состояние мира успешно обновлено.")
            return json_data.get("chapter_summary", "")
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"   ! Ошибка при парсинге JSON состояния мира: {e}")
            return None

    def _analyze_scenes_dynamics(self, chapter_plan_text):
        """Использует flash-lite для разметки динамики сцен."""
        print("  > ⚡ Анализ динамики сцен через Flash-Lite...")
        prompt = f"""
        Проанализируй план главы и определи тип динамики для КАЖДОЙ сцены.
        ПЛАН ГЛАВЫ:

        {chapter_plan_text}
        Доступные типы (mode):
        - action (драка, погоня, активное действие, стресс)
        - dialogue (разговор, спор, допрос, обсуждение)
        - description (описание места, размышления, спокойствие, наблюдение)
        - tension (саспенс, страх, ожидание, скрытное проникновение)
        - balanced (обычное повествование)

        Верни ТОЛЬКО JSON.
        """
        schema = {
            "type": "object",
            "properties": {
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "num": {"type": "integer"},
                            "title": {"type": "string"},
                            "mode": {"type": "string", "enum": ["action", "dialogue", "description", "tension", "balanced"]}
                        },
                        "required": ["num", "mode"]
                    }
                }
            }
        }

        try:
            response = self.world_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type='application/json',
                    response_schema=schema,
                    temperature=0.3 # Понижаем температуру для точности классификации
                ),
                safety_settings=SAFETY_SETTINGS,
            )
            print(f"SCE: {response.text}")
            return robust_json_parser(response.text).get('scenes', [])
        except Exception as e:
            print(f"   [!] Ошибка анализа ритма: {e}")
            return [] # Возвращаем пустой список, сработает фолбэк

    def step_foundation_1_1_analysis(self, project_state: 'NovelGenerationProject'):
        prompt_1_1 = load_prompt('prompt_1_1_analysis', synopsis=project_state.synopsis, NUM_CHAPTERS=NUM_CHAPTERS)
        self.world_bible['analysis'] = robust_json_parser(self._call_gemini(prompt_1_1, temperature=0.8, top_p=0.9, top_k=60, response_schema=SCHEMA_ANALYSIS))

    def step_foundation_1_2_characters(self, project_state: 'NovelGenerationProject'):
        prompt_1_2 = load_prompt('prompt_1_2_characters', NUM_CHAPTERS=NUM_CHAPTERS, synopsis=project_state.synopsis, world_bible=self.world_bible)
        self.world_bible['characters'] = self._call_gemini(prompt_1_2, temperature=0.9, top_p=0.9, top_k=50)

    def step_foundation_1_3_setting(self, project_state: 'NovelGenerationProject'):
        prompt_1_3 = load_prompt('prompt_1_3_setting', world_bible=self.world_bible)
        self.world_bible['setting'] = self._call_gemini(prompt_1_3, temperature=0.7, top_p=0.85, top_k=30)

    def step_foundation_1_4_voice_profiles(self, project_state: 'NovelGenerationProject'):
        prompt_1_2b = load_prompt('prompt_1_4_voice_profiles', world_bible=self.world_bible)
        self.world_bible['voice_profiles'] = self._call_gemini(prompt_1_2b, temperature=0.9, top_p=0.9, top_k=50)

    def step_foundation_1_5_book_style(self, project_state: 'NovelGenerationProject'):
        prompt_1_3a = load_prompt('prompt_1_5_book_style', world_bible=self.world_bible)
        self.world_bible['book_style'] = self._call_gemini(prompt_1_3a, temperature=0.6, top_p=0.8, top_k=25)

    def step_foundation_1_6_style(self, project_state: 'NovelGenerationProject'):
        prompt_1_4 = load_prompt('prompt_1_6_style', world_bible=self.world_bible, ALLOW_MATURE_LANGUAGE=ALLOW_MATURE_LANGUAGE)
        self.world_bible['style'] = self._call_gemini(prompt_1_4, temperature=0.6, top_p=0.8, top_k=25)

    def step_foundation_1_7_plan(self, project_state: 'NovelGenerationProject'):
        prompt_1_5 = load_prompt('prompt_1_7_plan', world_bible=self.world_bible, json_out=JSON_OUT, NUM_CHAPTERS=NUM_CHAPTERS)
        scene_plan = self._call_gemini(prompt_1_5, temperature=0.8, top_p=0.85, top_k=35, response_schema=PLAN_SCHEMA)
        self.world_bible['chapters'] = robust_json_parser(scene_plan)['chapters']
        log_chapter('scenes', 0, json.dumps(self.world_bible['chapters'], indent=2))

    def step_foundation_1_8_critique_plan(self, project_state: 'NovelGenerationProject'):
        full_plot_json = json.dumps(self.world_bible['chapters'], indent=2, ensure_ascii=False)
        wb = json.loads(json.dumps(self.world_bible))
        wb['analysis']['sequel_rules'] = START_INFO
        prompt_1_6 = load_prompt('prompt_1_8_critique_plan', world_bible=wb, full_plot_json=full_plot_json, NUM_CHAPTERS=NUM_CHAPTERS)
        plot_critique = self._call_gemini(prompt_1_6, temperature=0.6, top_p=0.8, top_k=30)
        project_state.transient_data['plot_critique'] = plot_critique # Сохраняем для следующего шага
        log_chapter('plot_critique', 0, plot_critique)

    def step_foundation_1_9_refactor_plan(self, project_state: 'NovelGenerationProject'):
        plot_critique = project_state.transient_data.pop('plot_critique', '') # Читаем и удаляем
        full_plot_json = json.dumps(self.world_bible['chapters'], indent=2, ensure_ascii=False)
        wb = json.loads(json.dumps(self.world_bible))
        wb['analysis']['sequel_rules'] = START_INFO
        prompt_1_7 = load_prompt('prompt_1_9_refactor_plan', world_bible=wb, plot_critique=plot_critique, full_plot_json=full_plot_json)
        edited_scene_plan_json = self._call_gemini(prompt_1_7, temperature=0.4, top_p=0.85, top_k=40, response_schema=PLAN_SCHEMA)
        self.world_bible['chapters'] = robust_json_parser(edited_scene_plan_json)['chapters']
        log_chapter('scenes_edited', 0, json.dumps(self.world_bible['chapters'], indent=2, ensure_ascii=False))

    def step_foundation_1_9_refactor_plan_2(self, project_state: 'NovelGenerationProject'):
        self.step_foundation_1_9_refactor_plan(project_state)

    def step_foundation_1_10_create_world_model(self, project_state: 'NovelGenerationProject'):
        self._create_world_model()

    # --- Этапы 'Chapter Generation' ---

    def _get_chapter_helpers(self, chapter_num: int, project_state: 'NovelGenerationProject'):
        previous_context = ""
        if chapter_num > 1:
            recent_summaries = project_state.chapter_summaries
            previous_context = f"""

                КОНТЕКСТ ПРЕДЫДУЩИХ ГЛАВ (для согласованности):
                [PREVIOUS_CONTEXT]
                {chr(10).join(recent_summaries)}
                [/PREVIOUS_CONTEXT]

                ВАЖНО: Сохраняй преемственность сюжета, развитие персонажей и упомянутые детали.
                """
        world_state_json = json.dumps(self.world_state, indent=2, ensure_ascii=False)
        return previous_context, world_state_json

    def step_chapter_X_1_plan(self, chapter_num: int, project_state: 'NovelGenerationProject'):
        chapter_scenes = self.world_bible['chapters'][chapter_num - 1]['scenes']
        chapter_scenes_str = "\n".join([str(x) for x in chapter_scenes])
        previous_context, world_state_json = self._get_chapter_helpers(chapter_num, project_state)
        full_text_previous_chapter = project_state.final_chapters_text[-1] if project_state.final_chapters_text else ""

        X_NUM_CHAPTERS = len(self.world_bible['chapters'])
        next_chapters = self.world_bible['chapters'][chapter_num:]
        next_chapters_summary = json.dumps({'FUTURE_KNOWLEDGE_TO_AVOID_BREAK': next_chapters}, ensure_ascii=False)
        prompt_2 = load_prompt('prompt_2_1_chapter_plan', X_NUM_CHAPTERS=X_NUM_CHAPTERS, chapter_scenes_str=chapter_scenes_str, chapter_num=chapter_num, world_state_json=world_state_json, previous_context=previous_context, full_text_previous_chapter=full_text_previous_chapter, next_chapters_summary=next_chapters_summary)
        chapter_plan = self._call_gemini(prompt_2, temperature=0.8, top_p=0.85, top_k=40, use_world_model=True)

        project_state.transient_data[f'ch_{chapter_num}_plan'] = chapter_plan
        log_chapter('plan', chapter_num, chapter_plan)

    def step_chapter_X_2_critique_plan(self, chapter_num: int, project_state: 'NovelGenerationProject'):
        chapter_plan = project_state.transient_data[f'ch_{chapter_num}_plan']
        previous_context, world_state_json = self._get_chapter_helpers(chapter_num, project_state)
        next_chapters = self.world_bible['chapters'][chapter_num:]
        next_chapters_summary = json.dumps({'FUTURE_KNOWLEDGE_TO_AVOID_BREAK': next_chapters}, ensure_ascii=False)

        prompt_2_5 = load_prompt('prompt_2_2_chapter_critique_plan', chapter_plan=chapter_plan, world_state_json=world_state_json, previous_context=previous_context, next_chapters_summary=next_chapters_summary)
        plan_critique = self._call_gemini(prompt_2_5, temperature=0.6, top_p=0.8, top_k=30, use_world_model=True)

        project_state.transient_data[f'ch_{chapter_num}_plan_critique'] = plan_critique
        log_chapter('plan_critique', chapter_num, plan_critique)

    def step_chapter_X_3_rewrite_plan(self, chapter_num: int, project_state: 'NovelGenerationProject'):
        chapter_plan = project_state.transient_data[f'ch_{chapter_num}_plan']
        plan_critique = project_state.transient_data[f'ch_{chapter_num}_plan_critique']

        prompt_2_75 = load_prompt('prompt_2_3_chapter_rewrite_plan', chapter_plan=chapter_plan, plan_critique=plan_critique)
        edited_plan = self._call_gemini(prompt_2_75, temperature=0.7, top_p=0.85, top_k=35, use_world_model=True)

        project_state.transient_data[f'ch_{chapter_num}_plan_edited'] = edited_plan
        log_chapter('plan_edited', chapter_num, edited_plan)
        print(f"✓ План Главы {chapter_num} успешно прошёл критику и был исправлен.")

    def step_chapter_X_4_draft(self, chapter_num: int, project_state: 'NovelGenerationProject'):
        chapter_plan = project_state.transient_data[f'ch_{chapter_num}_plan_edited']
        previous_context, world_state_json = self._get_chapter_helpers(chapter_num, project_state)

        scenes_dynamics = self._analyze_scenes_dynamics(chapter_plan)
        if scenes_dynamics:
            rhythm_instruction = self.rhythm_engine.generate_chapter_map(scenes_dynamics)
            print(f"   [RHYTHM] Сгенерирована карта для {len(scenes_dynamics)} сцен.")
        else:
            rhythm_instruction = self.rhythm_engine.generate_rhythm_block(scene_text=chapter_plan)
        print(f"   [RHYTHM] Сгенерирован ритм: {rhythm_instruction}") # Лог для отладки
        project_state.transient_data[f'ch_{chapter_num}_rhythm'] = rhythm_instruction

        num_techniques = random.choices([1, 2, 3, 4], weights=[10, 40, 40, 10])[0]
        selected_techniques = random.sample(TECHNIQUES_POOL, num_techniques)
        techniques_text = "\n".join([f"- {tech}" for tech in selected_techniques])
        random_chaos = ", ".join(random.sample(CHAOS_ELEMENTS, 2))
        mature_language_instruction = ""
        if ALLOW_MATURE_LANGUAGE:
            mature_language_instruction = """
            ВАЖНО: Используй аутентичную лексику, включая сильные выражения,
            если это соответствует персонажам и ситуации.
            """

        prompt_3 = load_prompt('prompt_3_chapter_draft', previous_context=previous_context, world_state_json=world_state_json, chapter_plan=chapter_plan, techniques_text=techniques_text, random_chaos=random_chaos, mature_language_instruction=mature_language_instruction, rhythm_instruction=rhythm_instruction)
        draft = self._call_gemini(prompt_3, temperature=0.85, top_p=0.98, top_k=80, use_world_model=True)

        project_state.transient_data[f'ch_{chapter_num}_draft'] = draft
        log_chapter('draft', chapter_num, draft)

    def step_chapter_X_5_critique_plot(self, chapter_num: int, project_state: 'NovelGenerationProject'):
        draft = project_state.transient_data[f'ch_{chapter_num}_draft']
        previous_context, world_state_json = self._get_chapter_helpers(chapter_num, project_state)
        mature_critique_instruction = ""
        if ALLOW_MATURE_LANGUAGE:
            mature_critique_instruction = """

        12. **АУТЕНТИЧНОСТЬ ЯЗЫКА:**
            - Соответствует ли лексика персонажей их социальному положению, профессии, эмоциональному состоянию?
            - Не слишком ли "причесана" речь для данной ситуации?
            - Используются ли естественные для персонажей выражения, включая сильную лексику где уместно?
            """

        prompt_4_1 = load_prompt('prompt_4_1_chapter_zanuda_1', previous_context=previous_context, world_state_json=world_state_json, draft=draft, mature_critique_instruction=mature_critique_instruction)
        critique = self._call_gemini(prompt_4_1, temperature=0.5, top_p=0.8, top_k=25, use_pro=True)

        project_state.transient_data[f'ch_{chapter_num}_critique_plot'] = critique
        log_chapter('critique', chapter_num, critique)

    def step_chapter_X_7_edit(self, chapter_num: int, project_state: 'NovelGenerationProject'):
        draft = project_state.transient_data[f'ch_{chapter_num}_draft']
        critique = project_state.transient_data[f'ch_{chapter_num}_critique_plot']
        critique_style = project_state.transient_data.get(f'ch_{chapter_num}_critique_style', '')
        mature_edit_instruction = ""
        if ALLOW_MATURE_LANGUAGE:
            mature_edit_instruction = """
            ЛЕКСИКА: При исправлениях сохраняй аутентичность языка персонажей.
            Не цензурируй их речь, если сильные выражения делают персонажей более правдоподобными.
            """

        prompt_4_2 = load_prompt('prompt_4_3_editor_tech', mature_edit_instruction=mature_edit_instruction, critique=critique, critique_style=critique_style, draft=draft, chapter_num=chapter_num)
        edited_chapter = self._call_gemini(prompt_4_2, temperature=0.7, top_p=0.85, top_k=35, use_world_model=True)

        project_state.transient_data[f'ch_{chapter_num}_edited'] = edited_chapter
        log_chapter('edited', chapter_num, edited_chapter)

    def step_chapter_X_8_check_repetitions(self, chapter_num: int, project_state: 'NovelGenerationProject'):
        edited_chapter = project_state.transient_data[f'ch_{chapter_num}_edited']

        repetition_critique = self.clusterer.check_cross_chapter_repetitions(edited_chapter, chapter_num)

        project_state.transient_data[f'ch_{chapter_num}_repetition_critique'] = repetition_critique
        log_chapter('critique_repetition', chapter_num, repetition_critique)

    def step_chapter_X_9_critique_stylistic(self, chapter_num: int, project_state: 'NovelGenerationProject'):
        edited_chapter = project_state.transient_data[f'ch_{chapter_num}_edited']
        repetition_critique = project_state.transient_data[f'ch_{chapter_num}_repetition_critique']
        mature_language_instruction = ""
        if ALLOW_MATURE_LANGUAGE:
            mature_language_instruction = """
            ВАЖНО: Используй аутентичную лексику, включая сильные выражения,
            если это соответствует персонажам и ситуации.
            """

        prompt_stylistic = load_prompt('prompt_4_4_reader', mature_language_instruction=mature_language_instruction, edited_chapter=edited_chapter, repetition_critique=repetition_critique)
        stylistic_critique = self._call_gemini(prompt_stylistic, temperature=0.75, top_p=0.8, top_k=35, use_world_model=True)

        project_state.transient_data[f'ch_{chapter_num}_stylistic_critique'] = stylistic_critique
        log_chapter('critique_stylistic', chapter_num, stylistic_critique)

    def step_chapter_X_10_finalize_and_state(self, chapter_num: int, project_state: 'NovelGenerationProject'):
        edited_chapter = project_state.transient_data.get(f'ch_{chapter_num}_edited')
        stylistic_critique = project_state.transient_data.get(f'ch_{chapter_num}_stylistic_critique')
        _, world_state_json = self._get_chapter_helpers(chapter_num, project_state)
        rhythm_instruction = project_state.transient_data.get(f'ch_{chapter_num}_rhythm')
        prompt_text = load_prompt('prompt_5_styler', stylistic_critique=stylistic_critique, edited_chapter=edited_chapter, rhythm_instruction=rhythm_instruction)
        response = self._call_gemini(prompt_text, temperature=0.5, top_p=0.8, top_k=20, use_world_model=True)
        final_chapter = response.replace("ОТРЕДАКТИРОВАННАЯ ГЛАВА:", "").strip()
        project_state.transient_data[f'ch_{chapter_num}_final_text'] = final_chapter
        log_chapter('final', chapter_num, final_chapter)
        print(f"✓ Текст Главы {chapter_num} успешно финализирован (Шаг X_10).")

    def step_chapter_X_11_extract_state(self, chapter_num: int, project_state: 'NovelGenerationProject'):
        final_chapter_text = project_state.transient_data.get(f'ch_{chapter_num}_final_text')
        _, world_state_json = self._get_chapter_helpers(chapter_num, project_state)
        prompt_text = load_prompt('prompt_6_extract_state', final_chapter_text=final_chapter_text, world_state_json=world_state_json, json_state_struct=JSON_STATE_STRUCT)
        response_text_with_tags = self._call_gemini(prompt_text, temperature=0.1, top_p=0.8, top_k=20, use_world_model=True)
        summary_text = self._update_world_state(response_text_with_tags, chapter_num)
        project_state.transient_data[f'ch_{chapter_num}_summary'] = summary_text
        log_chapter('summary', chapter_num, summary_text)
        print(f"✓ Состояние и резюме для Главы {chapter_num} успешно извлечены.")


class NovelGenerationProject:
    def __init__(self, state_file_name: str, api_key: str, synopsis: str):
        self.state_file_name = state_file_name
        self.synopsis = synopsis
        self.generator = NovelGenerator(api_key=api_key)

        self.steps: List[Step] = []
        self.qdrant_collection_name: str = f"novel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.chapter_summaries: List[str] = []
        self.final_chapters_text: List[str] = []
        self.transient_data: Dict[str, Any] = {}

        self._load_state()

        self.generator.clusterer.set_collection_name(self.qdrant_collection_name)

    def _get_state_data(self) -> dict:
        return {
            "steps": [asdict(s) for s in self.steps],
            "qdrant_collection_name": self.qdrant_collection_name,
            "chapter_summaries": self.chapter_summaries,
            "final_chapters_text": self.final_chapters_text,
            "transient_data": self.transient_data,
            "world_bible": self.generator.world_bible,
            "world_state": self.generator.world_state,
        }

    def _load_state(self):
        if not os.path.exists(self.state_file_name):
            print("Файл состояния не найден. Инициализация нового проекта.")
            self._initialize_steps()
            self.save_point()
            return

        print(f"Загрузка состояния из {self.state_file_name}...")
        try:
            with open(self.state_file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.steps = [Step(**s) for s in data.get('steps', [])]
            self.qdrant_collection_name = data.get('qdrant_collection_name', self.qdrant_collection_name)
            self.chapter_summaries = data.get('chapter_summaries', [])
            self.final_chapters_text = data.get('final_chapters_text', [])
            self.transient_data = data.get('transient_data', {})

            self.generator.world_bible = data.get('world_bible', {})
            self.generator.world_state = data.get('world_state', {})

            foundation_steps = [s for s in self.steps if s.handler_name.startswith('step_foundation_')]
            if foundation_steps and all(s.status == 'done' for s in foundation_steps):
                if self.generator.world_bible:
                    print("Восстановление 'Библии Мира' в модель...")
                    self.generator._create_world_model()
                else:
                    print("[!] Ошибка: шаги 'foundation' пройдены, но 'world_bible' пуста.")

            chapter_steps_exist = any(s.handler_name.startswith('step_chapter_') for s in self.steps)
            if not chapter_steps_exist and foundation_steps and all(s.status == 'done' for s in foundation_steps):
                print("Фаза 'Foundation' завершена, инициализация шагов глав...")
                self._initialize_chapter_steps()
                self.save_point()

            print("✓ Состояние успешно загружено.")

        except Exception as e:
            print(f"Ошибка загрузки состояния: {e}. Начинаем новый проект с нуля.")
            self._initialize_steps()
            self.save_point()

    def save_point(self):
        print(f"  ...Сохранение точки восстановления в {self.state_file_name}...")
        data_to_save = self._get_state_data()

        with open(self.state_file_name, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        print("  ✓ Точка восстановления сохранена.")

    def _initialize_steps(self):
        self.steps = [
            Step(name="1.1 Анализ", handler_name="step_foundation_1_1_analysis"),
            Step(name="1.2 Персонажи", handler_name="step_foundation_1_2_characters"),
            Step(name="1.3 Мир", handler_name="step_foundation_1_3_setting"),
            Step(name="1.4 Речевые профили", handler_name="step_foundation_1_4_voice_profiles"),
            Step(name="1.5 Стиль книги", handler_name="step_foundation_1_5_book_style"),
            Step(name="1.6 Стиль и тон", handler_name="step_foundation_1_6_style"),
            Step(name="1.7 План", handler_name="step_foundation_1_7_plan"),
            Step(name="1.8 Критика плана", handler_name="step_foundation_1_8_critique_plan"),
            Step(name="1.9 Правка плана", handler_name="step_foundation_1_9_refactor_plan"),
            Step(name="1.8-2 Критика плана", handler_name="step_foundation_1_8_critique_plan"),
            Step(name="1.9-2 Правка плана", handler_name="step_foundation_1_9_refactor_plan_2"),
            Step(name="1.10 Создание модели мира", handler_name="step_foundation_1_10_create_world_model"),
        ]

    def _initialize_chapter_steps(self):
        num_chapters_from_bible = len(self.generator.world_bible.get('chapters', []))
        num_chapters = num_chapters_from_bible if num_chapters_from_bible > 0 else int(NUM_CHAPTERS)

        if num_chapters == 0:
            print("[!] КРИТИЧЕСКАЯ ОШИБКА: Не удалось определить количество глав.")
            return

        print(f"Инициализация {num_chapters} глав...")

        for i in range(num_chapters):
            chapter_num = i + 1
            self.steps.extend([
                Step(name=f"Глава {chapter_num}: 2.1 План", handler_name=f"step_chapter_X_1_plan"),
                Step(name=f"Глава {chapter_num}: 2.2 Критика плана", handler_name=f"step_chapter_X_2_critique_plan"),
                Step(name=f"Глава {chapter_num}: 2.3 Правка плана", handler_name=f"step_chapter_X_3_rewrite_plan"),
                Step(name=f"Глава {chapter_num}: 3.1 Черновик", handler_name=f"step_chapter_X_4_draft"),
                Step(name=f"Глава {chapter_num}: 4.1 Критика (Сюжет)", handler_name=f"step_chapter_X_5_critique_plot"),
                Step(name=f"Глава {chapter_num}: 4.3 Редактура", handler_name=f"step_chapter_X_7_edit"),
                Step(name=f"Глава {chapter_num}: 4.4 Контроль повторов", handler_name=f"step_chapter_X_8_check_repetitions"),
                Step(name=f"Глава {chapter_num}: 4.5 Критика (Стилист)", handler_name=f"step_chapter_X_9_critique_stylistic"),
                Step(name=f"Глава {chapter_num}: 5.0 Финализация", handler_name=f"step_chapter_X_10_finalize_and_state"),
                Step(name=f"Глава {chapter_num}: 6.0 JSON", handler_name=f"step_chapter_X_11_extract_state"),
            ])

    def execute_step(self, step: Step):
        handler_name = step.handler_name
        kwargs = {'project_state': self}

        if handler_name.startswith('step_chapter_X_'):
            match = re.search(r"Глава (\d+):", step.name)
            if not match:
                raise ValueError(f"Не удалось извлечь номер главы из имени шага: {step.name}")

            chapter_num = int(match.group(1))
            handler_name = step.handler_name.replace('_X_', f'_{chapter_num}_')
            kwargs['chapter_num'] = chapter_num
            handler_name = step.handler_name

        handler = getattr(self.generator, handler_name, None)
        if not handler:
            raise AttributeError(f"Обработчик {handler_name} не найден в NovelGenerator")

        print(f"\n--- Выполнение шага: {step.name} ---")

        handler(**kwargs)

        if step.handler_name == 'step_foundation_1_9_refactor_plan_2':
            self._initialize_chapter_steps()

        if step.handler_name.endswith('_11_extract_state'):
            chapter_num = kwargs['chapter_num']

            final_text = self.transient_data.pop(f'ch_{chapter_num}_final_text', '')
            summary = self.transient_data.pop(f'ch_{chapter_num}_summary', f'Глава {chapter_num} - Ошибка резюме')

            self.final_chapters_text.append(f"# Глава {chapter_num}\n\n{final_text}")
            self.chapter_summaries.append(f"Глава {chapter_num}: {summary}")
            keys_to_delete = [k for k in self.transient_data if k.startswith(f'ch_{chapter_num}_')]
            for k in keys_to_delete:
                del self.transient_data[k]
            print(f"  ✓ Очищены временные данные для Главы {chapter_num}.")

    def run(self):
        """Главный цикл выполнения проекта. Проходит по шагам и сохраняется."""
        for step in self.steps:
            if step.status != 'done':
                step.status = 'started'
                self.save_point()
                try:
                    self.execute_step(step)
                    step.status = 'done'
                    self.save_point()
                except Exception as e:
                    print(f"\n[!] КРИТИЧЕСКАЯ ОШИБКА на шаге '{step.name}': {e}")
                    print("Процесс остановлен. Запустите скрипт снова для возобновления.")
                    step.status = 'planned'
                    self.save_point()
                    raise

        print("\n🎉 Все шаги выполнены. Генерация романа завершена.")
        output_filename = "my_novel.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(self.final_chapters_text))
        print(f"Роман сохранен в файл: {output_filename}")

if __name__ == "__main__":
    if API_KEY == 'ВАШ_API_КЛЮЧ':
        print("Ошибка: Пожалуйста, вставьте ваш API ключ в переменную API_KEY в скрипте.")
    else:
        STATE_FILE = "novel_project_state.json"
        print("Запуск менеджера проекта генерации романа...")
        try:
            project = NovelGenerationProject(
                state_file_name=STATE_FILE,
                api_key=API_KEY,
                synopsis=SYNOPSIS
            )
            project.run()
        except Exception as e:
            print(f"\n[!] ГЛОБАЛЬНАЯ ОШИБКА: {e}")
            raise e
            print("Процесс был аварийно завершен. Точка восстановления (если была) сохранена.")
