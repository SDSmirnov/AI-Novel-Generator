#!/usr-bin/env python
# coding: utf-8

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import wraps
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

import hashlib
import json
import os
import re
import threading
import time

# ============================================================================
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ============================================================================

DEFAULT_SMART_MODEL = 'openai/gpt-5.1'
DEFAULT_FAST_FACTS_MODEL = 'openai/gpt-5.1' # 'google/gemini-3-pro-preview'

COUNTERS = {'prompt': 0, 'response': 0, 'completion': 0}

def log_me(kind, text):
    Path('request_logs').mkdir(exist_ok=True)
    COUNTERS[kind] += 1
    number = COUNTERS[kind]
    f = open(f'request_logs/{kind}_{number:02}.md', 'a')
    f.write(text)
    f.close()

class AiClient:
    def __init__(self, model):
        self.model = model
        self.client = client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OR_AI_API_KEY')
        )

    def generate_content(self, prompt, *args, **kwargs):
        log_me('prompt', prompt)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=15000,
            extra_body={
                'reasoning': {
                    "max_tokens": 12000,
                }
            }
        )
        log_me('completion', str(completion))
        response = completion.choices[0].message.content or completion.choices[0].message.reasoning
        log_me('response', response)
        return response

@dataclass
class AnalyzerConfig:
    """Конфигурация анализатора"""
    smart_model: str = DEFAULT_SMART_MODEL
    fast_model: str = DEFAULT_FAST_FACTS_MODEL
    api_delay_seconds: float = 5.0
    chunk_target_words: int = 10000
    max_retries: int = 3
    retry_delay: float = 10.0
    temperature: float = 0.3
    max_output_tokens: int = 32000
    genre: Optional[str] = None
    single_book_mode: bool = False

    def __post_init__(self):
        if self.chunk_target_words < 1000:
            raise ValueError("chunk_target_words должен быть >= 1000")
        if self.api_delay_seconds < 1:
            print("Предупреждение: api_delay_seconds < 1. Рекомендуется > 1с.")

# ============================================================================
# УНИВЕРСАЛЬНЫЕ КРИТЕРИИ И ШКАЛА ОЦЕНОК
# ============================================================================

UNIVERSAL_CRITERIA_WITH_RATINGS = """
КРИТЕРИИ ОЦЕНКИ (УНИВЕРСАЛЬНЫЕ):

1. ЯЗЫК И СТИЛЬ:
   - Качество прозы, богатство языка, метафоры, ритм.
   - Атмосферность, соответствие стиля тону повествования.
   - Отсутствие стилистических ошибок, канцеляризмов, штампов.

2. КОМПОЗИЦИЯ И СТРУКТУРА:
   - Логика построения сюжета, связность глав.
   - Наличие экспозиции, развития, кульминации, развязки.
   - Отсутствие провисаний, логических дыр.

3. ПЕРСОНАЖИ И РАЗВИТИЕ:
   - Глубина проработки (психологизм, мотивация).
   - Развитие персонажей (арки), их изменение под влиянием событий.
   - Избегание "картонности" или "Мэри Сью".

4. ДИАЛОГИ:
   - Естественность речи, соответствие персонажам.
   - Индивидуальность голосов (персонажи говорят по-разному).
   - Функциональность (двигают сюжет / раскрывают характер, а не только служат экспозицией).

5. МИР И СЕТТИНГ (Worldbuilding):
   - Убедительность и глубина проработки мира.
   - Внутренняя непротиворечивость правил (физических, магических, социальных).
   - Влияние сеттинга на сюжет и персонажей.

6. ТЕМЫ И ИДЕИ:
   - Наличие и глубина проработки центральных тем.
   - Оригинальность и нетривиальность поднимаемых вопросов.
   - Отсутствие дидактизма (прямого "вдалбливания" морали).

7. ОРИГИНАЛЬНОСТЬ:
   - Новизна идей, сеттинга, персонажей.
   - Качество работы с жанровыми канонами (если применимо) – 'качественно, но ожидаемо' vs 'новый взгляд'.

8. ТЕМП И ДРАМАТУРГИЯ:
   - Управление вниманием читателя, саспенс.
   - Баланс между экшеном и рефлексией.
   - Эмоциональный отклик, катарсис.

9. ЦЕЛОСТНОСТЬ ТОНА:
   - Выдержанность единой интонации (юмор, драма, трагедия).
   - Отсутствие неуместных "выпадений" из тона.

10. ОБЩЕЕ ВПЕЧАТЛЕНИЕ:
    - Субъективный итог, насколько произведение захватывает и оставляет след.

ФОРМАТ ТАБЛИЦЫ ОЦЕНОК (для включения в рецензию):

| Критерий | Оценка (1–10) | Комментарий |
| :--- | :--- | :--- |
| Язык и стиль | [1-10] | [Краткий комментарий] |
| Композиция и структура | [1-10] | [Краткий комментарий] |
| Персонажи и развитие | [1-10] | [Краткий комментарий] |
| Диалоги | [1-10] | [Краткий комментарий] |
| Мир и сеттинг | [1-10] | [Краткий комментарий] |
| Темы и идеи | [1-10] | [Краткий комментарий] |
| Оригинальность | [1-10] | [Краткий комментарий] |
| Темп и драматургия | [1-10] | [Краткий комментарий] |
| Целостность тона | [1-10] | [Краткий комментарий] |
| **Общее впечатление** | **[1-10]** | **[Итоговый комментарий]** |
"""

RATING_SCALE = """
ШКАЛА ОЦЕНОК (из 10):

10 - Шедевр, эталонное произведение
9 - Выдающееся качество, минимум недостатков
8 - Отличная работа с незначительными огрехами
7 - Хорошее качество, но есть заметные недочеты
6 - Средний уровень, больше достоинств чем недостатков
5 - Посредственно, достоинства = недостатки
4. - Ниже среднего, недостатки перевешивают
3 - Слабая работа с серьезными проблемами
2 - Очень слабо, трудно читать
1 - Крайне низкое качество
"""

# ============================================================================
# УТИЛИТЫ
# ============================================================================

class Logger:
    """Улучшенная система логирования в папку сессии"""
    def __init__(self, base_dir: str = 'request_logs'):
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.counters = {'prompt': 0, 'response': 0}

    def log(self, kind: str, text: str, metadata: Optional[Dict] = None):
        """Логирование с метаданными"""
        self.counters[kind] += 1
        timestamp = datetime.now().strftime('%H:%M:%S')
        filename = f"{self.counters[kind]:03d}_{kind}_{timestamp.replace(':', '')}.md"
        filepath = self.session_dir / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {kind.upper()} #{self.counters[kind]}\n")
                f.write(f"**Время:** {timestamp}\n\n")
                if metadata:
                    f.write("## Метаданные\n")
                    for key, value in metadata.items():
                        f.write(f"- **{key}:** {value}\n")
                    f.write("\n")
                f.write("## Содержимое\n\n")
                f.write(text)
        except Exception as e:
            print(f"  ! Ошибка записи лога: {e}")

class CacheManager:
    """Управление кешем результатов"""
    def __init__(self, cache_dir: str = 'analysis_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_key(self, text: str, prefix: str = "") -> str:
        content = f"{prefix}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text_to_hash: str, prefix: str) -> Optional[str]:
        key = self._get_key(text_to_hash, prefix)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('result')
            except Exception as e:
                print(f"    ⚠ Ошибка чтения кеша: {e}")
        return None

    def set(self, text_to_hash: str, prefix: str, value: str, metadata: Optional[Dict] = None):
        key = self._get_key(text_to_hash, prefix)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            data = {
                'result': value,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"    ⚠ Ошибка записи в кеш: {e}")

    def get_full(self, key: str) -> Optional[Dict]:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"    ⚠ Ошибка чтения кеша: {e}")
        return None

    def set_full(self, key: str, data: Dict):
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"    ⚠ Ошибка записи в кеш: {e}")

def retry_on_error(max_retries: int = 3, delay: float = 10.0):
    """Декоратор для повторных попыток при ошибках API"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs): # 'self' добавлен
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    if "429" in error_str or "Resource" in error_str:
                        wait_time = delay * (2 ** attempt)
                        print(f"    ⚠ Лимит API (попытка {attempt+1}/{max_retries}). Ожидание {wait_time:.1f}с...")
                        time.sleep(wait_time)
                    elif "500" in error_str or "503" in error_str:
                        wait_time = delay * (attempt + 1)
                        print(f"    ⚠ Ошибка сервера (попытка {attempt+1}/{max_retries}). Ожидание {wait_time:.1f}с...")
                        time.sleep(wait_time)
                    else:
                        print(f"    ✗ КРИТИЧЕСКАЯ ОШИБКА API (не 429/50x): {e}")
                        self.processing_stats['errors'] += 1
                        raise
            print(f"    ✗ Все {max_retries} попытки неудачны")
            self.processing_stats['errors'] += 1
            raise last_error
        return wrapper
    return decorator

# ============================================================================
# ОСНОВНОЙ КЛАСС АНАЛИЗАТОРА
# ============================================================================

class BookCycleAnalyzer:

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """Инициализация анализатора с конфигом, логгером и кешем"""
        self.config = config or AnalyzerConfig()
        self.logger = Logger()
        self.cache = CacheManager()
        self.output_dir: Path = Path('analyses_map_reduce')
        self.genre = self.config.genre
        self.single_book_mode = self.config.single_book_mode

        # Инициализация API
        try:
            api_key = os.getenv('OR_AI_API_KEY')
            if not api_key:
                raise ValueError("Переменная окружения 'OR_AI_API_KEY' не установлена")

            self.smart_model = AiClient(self.config.smart_model)
            self.fast_model = AiClient(self.config.fast_model)

            print(f"✓ Модели инициализированы (Логи: {self.logger.session_dir}):")
            print(f"  - Pro: {self.config.smart_model}")
            print(f"  - Flash: {self.config.fast_model}")
            if self.genre:
                print(f"  - Жанр: {self.genre}")
            if self.single_book_mode:
                print(f"  - Режим: Только одна книга")


        except Exception as e:
            print(f"✗ КРИТИЧЕСКАЯ ОШИБКА инициализации: {e}")
            raise

        self.book_analyses_for_report = {}
        self.processing_stats = {
            'start_time': datetime.now(),
            'chunks_processed': 0,
            'books_processed': 0,
            'api_calls': 0,
            'cache_hits_chunk': 0,
            'cache_hits_book': 0,
            'cache_hits_cycle': 0,
            'errors': 0
        }

    # ========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================================

    def _split_text_into_chunks(self, text: str, title: str) -> List[str]:
        """Разделение текста на части с улучшенной логикой"""
        words = text.split()
        total_words = len(words)
        target = self.config.chunk_target_words
        print(f"    → Разделение '{title}' ({total_words:,} слов) на части по ~{target:,} слов...")
        if total_words <= target * 1.2:
            print("    ✓ Текст слишком мал для разделения, используется 1 часть.")
            return [text]
        chunks = []
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = []
        current_words = 0
        for para in paragraphs:
            if not para.strip(): continue
            para_words = len(para.split())
            if para_words > target * 1.5:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_words = 0
                chunks.append(para)
                continue
            if current_words + para_words > target and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_words = para_words
            else:
                current_chunk.append(para)
                current_words += para_words
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        avg_size = total_words // len(chunks) if chunks else 0
        print(f"    ✓ Создано {len(chunks)} частей (в среднем {avg_size:,} слов)")
        return chunks

    def _extract_text_between(self, text: str, start: str, end: str) -> Optional[str]:
        """Извлечение текста между тегами"""
        try:
            pattern = re.compile(f'{re.escape(start)}(.+){re.escape(end)}', re.DOTALL)
            match = pattern.search(text)
            if match:
                return match.group(1).strip()

            start_flex = start.replace('[', r'\[')
            end_flex = end.replace('[', r'\[')
            pattern_flex = re.compile(f'{start_flex}(.*?){end_flex}', re.DOTALL | re.IGNORECASE)
            match_flex = pattern_flex.search(text)
            if match_flex:
                print(f"    ! Предупреждение: Теги найдены с гибким поиском ('{start}'...'{end}')")
                return match_flex.group(1).strip()

            print(f"    ! ОШИБКА: Теги '{start}'...'{end}' не найдены в ответе.")
            return None
        except Exception as e:
            print(f"    ! ОШИБКА извлечения тегов: {e}")
            return None

    def _api_pause(self):
        """Пауза между вызовами API с индикацией"""
        delay = self.config.api_delay_seconds
        if delay < 1: return
        print(f"    → Пауза {delay:.1f}с (лимиты API)", end='', flush=True)
        for _ in range(int(delay * 10)):
            time.sleep(0.1)
            print(".", end='', flush=True)
        print(" ✓")

    # ========================================================================
    # ЭТАП 1 (MAP) - ИЗМЕНЕНИЕ V5 (ПОЛНЫЙ СБОР ДАННЫХ)
    # ========================================================================

    @retry_on_error(max_retries=3, delay=10.0)
    def _analyze_chunk(
        self,
        chunk_text: str,
        chunk_num: int,
        total_chunks: int,
        book_title: str
    ) -> str:
        """
        Анализ одной части книги - СБОР ФАКТОВ И КАЧЕСТВЕННЫХ НАБЛЮДЕНИЙ .
        БЕЗ ОЦЕНОК (1-10) И РЕЦЕНЗИЙ.
        """
        cached = self.cache.get(chunk_text, f"chunk_{book_title}_{chunk_num}")
        if cached:
            self.processing_stats['cache_hits_chunk'] += 1
            print(f"    ✓ Часть {chunk_num}/{total_chunks} взята из кеша")
            return cached

        print(f"    → Извлечение данных из части {chunk_num}/{total_chunks} ('{book_title}')... (Модель: {self.config.fast_model})")

        # --- ИЗМЕНЕНИЕ V5: Промпт переработан для сбора данных по ВСЕМ слепым зонам ---
        prompt = f"""Ты — ассистент литературного критика.

Твоя задача — извлечь из фрагмента текста (часть {chunk_num} из {total_chunks}, роман "{book_title}") два типа данных:
1.  **Факты** (события, реплики).
2.  **Качественные наблюдения** (стиль, психология, темп, темы) С ПОДТВЕРЖДАЮЩИMI ЦИТАТАМИ И СВОДКАМИ.

ЗАПРЕЩЕНО:
- Давать итоговые оценки ("хорошо", "плохо", "оценка 8/10", "клише").
- Писать рецензию или выводы по всей книге.
- Анализировать "ПОТЕНЦИАЛЬНЫЕ ПРОБЛЕМЫ".

Ты собираешь *сырой материал* для главного критика. Оберни весь результат в теги <CHUNK_SUMMARY>...</CHUNK_SUMMARY>.

---
**РАЗДЕЛ 1: ФАКТОЛОГИЧЕСКИЕ ДАННЫЕ (ЧТО ПРОИЗОШЛО)**
---

1. КЛЮЧЕВЫЕ СОБЫТИЯ ФРАГМЕНТА:
   - Перечисли по пунктам основные события, произошедшие в ЭТОМ фрагменте.

2. ДЕТАЛИ ПЕРСОНАЖЕЙ (ФАКТЫ):
   Для КАЖДОГО персонажа, УПОМЯНУТОГО в этом фрагменте:
   [Имя персонажа]:
   • Появление: (Если появился впервые, опиши внешность/статус).
   • Ключевые действия: (Что делал? **Список фактов**).
   • Высказанная мотивация: (Только если персонаж *сам* говорит о своих мотивах).

---
**РАЗДЕЛ 2: КАЧЕСТВЕННЫЕ НАБЛЮДЕНИЯ (КАК ЭТО НАПИСАНО)**
---

3. СТИЛЬ, ЯЗЫК И ДИАЛОГИ:

   А) АВТОРСКАЯ ПРОЗА (ОПИСАНИЯ):
   • **Детальный анализ стиля (3-5 предл.):** Опиши *качество* стиля в этом фрагменте. Насколько он богат? Используются ли метафоры? Есть ли проблемы (канцеляризмы, штампы)?
   • **Примеры стиля (3-5 цитаты):** Приведи 3-5 примера *описаний* или *рефлексии*, показывающих этот стиль.

   Б) ДИАЛОГИ (РЕЧЬ ПЕРСОНАЖЕЙ):
   • **Сводка диалогов (1-2 предл.):** Опиши *качество* диалогов в этом фрагменте (напр: "Естественные, живые", "Книжные, искусственные", "Все персонажи говорят одинаково", "Диалоги хорошо раскрывают характеры").
   • **Примеры диалогов (3-5 цитаты):** Приведи 3-5 *показательных* диалога (можно коротких), которые подтверждают твою сводку (удачные или неудачные).

4. АТМОСФЕРА И ДОСТОВЕРНОСТЬ:
   • **Сводка (1-2 предл.):** Какая атмосфера доминирует? (напр: "Напряженная", "Юмористическая"). Насколько достоверны детали мира?
   • **Примеры (1-2 цитаты):** Приведи деталь, работающую на атмосферу или достоверность.

5. ПСИХОЛОГИЗМ И РАЗВИТИЕ ПЕРСОНАЖЕЙ :
   * **Анализ психологизма (3-5 предл.):** Насколько глубоко передано внутреннее состояние? Это прямая рефлексия ('он подумал') или косвенная (через действия)? Убедительны ли эмоции?
   • **КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ (1-2 предл.):** Замечены ли в этом чанке *значимые внутренние изменения* или *противоречия* у главных героев? (напр: "Герой А впервые усомнился в...", "Герой Б поступил вопреки своим принципам", "Изменений нет").
   • **Примеры (1-2 цитаты):** Приведи пример передачи эмоций или *момент* внутреннего изменения.

6. ТЕМП И НАПРЯЖЕНИЕ :
   • **Сводка по темпу (1-2 предл.):** Опиши темп этого фрагмента (напр: "Очень медленный, рефлексивный", "Динамичный, экшен-сцена", "Скомканный, 'галопом по Европам'", "Темп провисает, мало событий").
   • **Управление напряжением (1-2 предл.):** Как автор работает с саспенсом в этом чанке? (напр: "Создает напряжение...", "Разряжает обстановку...", "Нет напряжения").

7. ТЕМЫ И ИДЕИ :
   • **Возникающие темы (1-2 предл.):** Какие темы или философские вопросы поднимаются *в этом фрагменте*? (напр: "Тема верности", "Вопрос цены победы", "Одиночество", "Конфликт отцов и детей", "Темы не замечены").

═══════════════════════════════════════════════════════════════
ТЕКСТ ФРАГМЕНТА:

{chunk_text}
═══════════════════════════════════════════════════════════════

Напоминаю: БЕЗ ОЦЕНОК. Только сбор материала и качественные *сводки наблюдений* по чанку. Оберни весь ответ в <CHUNK_SUMMARY>...</CHUNK_SUMMARY>.
Ответ на русском языке.
"""

        self.logger.log('prompt', prompt, {
            'book': book_title,
            'chunk': f"{chunk_num}/{total_chunks}",
            'model': self.config.fast_model,
            'type': 'chunk_extraction' # Обновлено
        })

        response = self.fast_model.generate_content(
            prompt
        )

        self.processing_stats['api_calls'] += 1

        result = self._extract_text_between(response, '<CHUNK_SUMMARY>', '</CHUNK_SUMMARY>')
        if not result:
            print(f"  ! ОШИБКА: Не удалось извлечь <CHUNK_SUMMARY> из ответа Flash. Используется сырой ответ.")
            result = response # Откат к сырому ответу в случае сбоя парсинга

        self.logger.log('response', result, {
            'book': book_title,
            'chunk': f"{chunk_num}/{total_chunks}"
        })

        self.cache.set(chunk_text, f"chunk_{book_title}_{chunk_num}", result, {
            'book': book_title,
            'chunk': chunk_num,
            'total': total_chunks
        })

        return result

    # ========================================================================
    # ЭТАП 2 (REDUCE - Книга)
    # ========================================================================

    @retry_on_error(max_retries=3, delay=15.0)
    def _synthesize_book_analysis(
        self,
        fact_summaries_text: str,
        book_title: str,
        book_number: int,
        full_book_text_for_hashing: str,
        previous_books_context: str
    ) -> Tuple[Optional[str], Optional[str]]:
        cache_key_text = f"{previous_books_context}\n{full_book_text_for_hashing}"
        cache_prefix = f"book_synthesis_{book_title}"
        cached = self.cache.get(cache_key_text, cache_prefix)

        if cached:
            self.processing_stats['cache_hits_book'] += 1
            print(f"  → Синтез '{book_title}' взят из кеша")
            response_text = cached
        else:
            print(f"  → Синтез анализа книги '{book_title}'... (Модель: {self.config.smart_model})")

            context_section = (
                f"\n{'='*60}\nКОНТЕКСТ ПРЕДЫДУЩИХ КНИГ ЦИКЛА:\n{previous_books_context}\n{'='*60}\n"
                if previous_books_context and not self.single_book_mode else
                "Это первая книга цикла (или анализ в режиме одной книги).\n"
            )

            genre_context = (
                f"\n{'='*60}\nКОНТЕКСТ ЖАНРА: {self.genre}\n{'='*60}\n"
                f"Критикуй с учетом особенностей этого жанра (например, для ЛитРПГ важна логика системы, для романтики - химия, для детектива - улики).\n"
                if self.genre else ""
            )

            prompt = f"""Ты — профессиональный литературный критик.

ЗАДАНИЕ: Написать ОБЪЕКТИВНУЮ, ПОДРОБНУЮ критическую рецензию на роман "{book_title}" (книга #{book_number}).
Ты работаешь ТОЛЬКО на основе предоставленных "СЫРЫХ ФАКТОЛОГИЧЕСКИХ ДАННЫХ И КАЧЕСТВЕННЫХ НАБЛЮДЕНИЙ", собранных ассистентами.

{context_section}
{genre_context}

{UNIVERSAL_CRITERIA_WITH_RATINGS}

{RATING_SCALE}

═══════════════════════════════════════════════════════════════
ПРИНЦИПЫ АНАЛИЗА:
1. ОБЪЕКТИВНОСТЬ: Используй "СЫРЫЕ ДАННЫЕ" (Раздел 1) для поиска логических дыр, OOC.
2. ГЛУБИНА: Используй "КАЧЕСТВЕННЫЕ НАБЛЮДЕНИЯ" (Раздел 2) для анализа стиля, психологии, арок, темпа и тем.
3. КОНКРЕТНОСТЬ: ВСЕГДА подтверждай оценки цитатами или примерами из данных.

═══════════════════════════════════════════════════════════════

ЗАДАЧА: Выполни анализ в 3 ЭТАПА (думай пошагово):

ЭТАП 1: ЧЕРНОВИК (Роль: Критик-Аналитик)
(Ты не показываешь этот текст)
Проанализируй ВСЕ "СЫРЫЕ ДАННЫЕ". Напиши черновик рецензии по 7-ми пунктам (см. ниже) и заполни Итоговую Таблицу Оценок. Будь строгим, но справедливым.
(Особое внимание удели сводкам по темпу, темам, аркам и диалогам из Раздела 2).

ЭТАП 2: САМОКРИТИКА (Роль: Критик-Оппонент)
(Ты не показываешь этот текст)
Прочти свой черновик из Этапа 1. Найди 2-3 самых СЛАБЫХ, ПРЕДВЗЯТЫХ или НЕДОСТАТОЧНО АРГУМЕНТИРОВАННЫХ вывода в своей же рецензии. Укажи на них и объясни, почему они слабые.
(Пример: "Мой вывод о том, что 'все диалоги плохие', предвзят. Данные из чанка 5 (Раздел 2) показывают хороший диалог X, который я проигнорировал. Оценку стоит скорректировать.")

ЭТАП 3: ФИНАЛЬНЫЙ ОТЧЁТ (Роль: Главный Редактор)
(Ты показываешь ТОЛЬКО этот текст)
На основе Этапа 1 и критики из Этапа 2, напиши финальную, взвешенную и объективную рецензию.
Создай ДВА блока в ОДНОМ ответе: <ANALYSIS>...</ANALYSIS> и <KEYPOINTS>...</KEYPOINTS>.

===============================================================
<ANALYSIS>
(Здесь твоя финальная подробная рецензия, написанная на основе Этапа 3)

**0. ИТОГОВАЯ ОЦЕНОЧНАЯ ТАБЛИЦА**
(ОБЯЗАТЕЛЬНО ЗАПОЛНИ эту таблицу. Комментарии должны быть краткими и ёмкими.)

| Критерий | Оценка (1–10) | Комментарий |
| :--- | :--- | :--- |
| Язык и стиль | ... | [Основывайся на сводках "Авторская проза"] |
| Композиция и структура | ... | [Основывайся на "Событиях" и "Темпе"] |
| Персонажи и развитие | ... | [Основывайся на "Психологизм и Развитие"] |
| Диалоги | ... | [Основывайся на сводках "Диалоги"] |
| Мир и сеттинг | ... | [Основывайся на "Детали мира" и "Достоверность"] |
| Темы и идеи | ... | [Основывайся на "Темы и Идеи"] |
| Оригинальность | ... | ... |
| Темп и драматургия | ... | [Основывайся на "Темп и Напряжение"] |
| Целостность тона | ... | [Основывайся на "Атмосфера"] |
| **Общее впечатление** | **...** | **...** |

---
**1. СЮЖЕТ И КОМПОЗИЦИЯ** (300-400 слов)
• Синтез сюжета (без спойлеров финала).
* Структура, темп, динамика. **Критически оцени связность и переходы темпа *между* данными из разных чанков.** Есть ли резкие скачки или провалы в повествовании, не объяснённые в тексте? Логическая цельность
• **КРИТИЧЕСКАЯ ОЦЕНКА:** Логическая цельность, наличие "роялей", провисания.

---
**2. ПЕРСОНАЖИ И ХАРАКТЕРЫ** (350-450 слов)
А) ГЛАВНЫЕ ГЕРОИ:
   • Проработка, мотивация, развитие, арки (на основе сводок "Психологизм и Развитие").
Б) ВТОРОСТЕПЕННЫЕ ПЕРСОНАЖИ:
   • Объемность vs. картонность. Антагонисты.
В) ДИАЛОГИ:
   • Естественность, индивидуальность (на основе сводок "Диалоги").
• **КРИТИЧЕСКАЯ ОЦЕНКА:** OOC моменты, неубедительные мотивации, психологическая достоверность.

---
**3. МИРОСТРОЙ И СЕТТИНГ (Worldbuilding)** (350-450 слов)
А) ИСТОРИЧЕСКИЙ/ФАНТАСТИЧЕСКИЙ ФОН:
   • Достоверность быта, культуры, политики (на основе сводок "Достоверность").
Б) УБЕДИТЕЛЬНОСТЬ МИРА:
   • Обоснованность правил мира, технологий, магии. Атмосфера.
• **КРИТИЧЕСКАЯ ОЦЕНКА:** нелогичности, нарушения правил с примерами.

---
**4. СТИЛЬ И ЯЗЫК** (300-350 слов)
• **(Важно!)** Оценка стиля, работа с языком, описания (на основе сводок "Авторская проза").
• **КРИТИЧЕСКАЯ ОЦЕНКА:** Типичные стилистические ошибки, штампы, клише.

---
**5. ТЕМЫ И ИДЕИ** (250-350 слов)
• Центральные темы, глубина их раскрытия, оригинальность (на основе сводок "Темы и Идеи").
• **КРИТИЧЕСКАЯ ОЦЕНКА:** недораскрытые темы, противоречия.

---
**6. КРИТИЧЕСКИЙ РАЗБОР НЕДОСТАТКОВ** (400-500 слов)
(КЛЮЧЕВОЙ РАЗДЕЛ. Синтезируй ВСЕ проблемы, найденные тобой при анализе "СЫРЫХ ДАННЫХ")
А) ЛОГИЧЕСКИЕ И СЮЖЕТНЫЕ: (Сюжетные дыры, "рояли"). **Обязательно найди несоответствия *между* сводками чанков (например, персонаж в Чанке 2 внезапно забывает то, что знал в Чанке 1; или мотивация, заявленная в Чанке 3, противоречит действиям в Чанке 5).**
Б) ПРОБЛЕМЫ С ПЕРСОНАЖАМИ: (OOC, мотивация, Мэри Сью, плоские арки).
В) НАРУШЕНИЯ МИРОСТРОЯ: (Несоответствия, анахронизмы, проблемы логики).
Г) МЕЖКНИЖНЫЕ НЕСООТВЕТСТВИЯ: (Используй "КОНТЕКСТ ПРЕДЫДУЩИХ КНИГ"!)

---
**7. ИТОГОВОЕ ВПЕЧАТЛЕНИЕ**
• **Обоснование:** (Краткое резюме, баланс достоинств и недостатков, основанное на таблице оценок и {RATING_SCALE}).

</ANALYSIS>
===============================================================
<KEYPOINTS>
(Здесь сжатая выжимка для анализа ЦИКЛА. Будь краток и структурирован.)

1. ПЕРСОНАЖИ (Ключевое развитие в ЭТОЙ книге, 1-2 предложения на ГГ).
2. СЮЖЕТНЫЕ ЛИНИИ (Главные арки и их разрешение/развитие в ЭТОЙ книге).
3. КЛЮЧЕВЫЕ ЭЛЕМЕНТЫ (Главные идеи, события, изменения мира в этой книге).
4. ТЕМЫ (Топ-3 темы этой книги, из сводок).
5. ГЛАВНЫЕ НЕСООТВЕТСТВИЯ И ПРОБЛЕМЫ (Топ-5 самых критичных проблем, найденных тобой в ЭТАПЕ 2).
6. СВЯЗЬ С ЦИКЛОМ (Как эта книга двигает общий сюжет? Какие линии продолжает?).
</KEYPOINTS>
===============================================================

СЫРЫЕ ФАКТОЛОГИЧЕСКИЕ ДАННЫЕ И КАЧЕСТВЕННЫЕ НАБЛЮДЕНИЯ (из всех частей):

{fact_summaries_text}

Отвечай строго по структуре, на русском языке. Выдай ТОЛЬКО ЭТАП 3 (<ANALYSIS> и <KEYPOINTS>).
"""

            self.logger.log('prompt', prompt, {
                'book': book_title,
                'book_num': book_number,
                'model': self.config.smart_model,
                'type': 'book_synthesis_prompt'
            })

            response = self.smart_model.generate_content(
                prompt
            )

            self.processing_stats['api_calls'] += 1
            response_text = response

            self.logger.log('response', response_text, {
                'book': book_title,
                'book_num': book_number
            })

            self.cache.set(cache_key_text, cache_prefix, response_text, {
                'book': book_title,
                'book_num': book_number
            })

        # Парсинг ответа
        full_analysis = self._extract_text_between(response_text, '<ANALYSIS>', '</ANALYSIS>')
        key_elements = self._extract_text_between(response_text, '<KEYPOINTS>', '</KEYPOINTS>')

        if not full_analysis or not key_elements:
            print(f"  ! ОШИБКА: Не удалось распарсить ответ 'Pro' для книги {book_number}. Ответ целиком:\n{response_text[:1000]}...")
            self.processing_stats['errors'] += 1
            return None, None

        return full_analysis, key_elements

    # ========================================================================
    # ЭТАП 3 (REDUCE - Цикл)
    # ========================================================================

    @retry_on_error(max_retries=3, delay=20.0)
    def _analyze_cycle(self, all_key_elements: List[Dict]) -> Dict:
        """
        Анализ ЦИКЛА на основе сжатых представлений (key_elements) всех книг.
        """
        print(f"\n{'='*60}\n→ Запрос на финальный анализ ЦИКЛА... (Модель: {self.config.smart_model})")

        summaries_text = "\n\n=====\n\n".join([
            f"КНИГА {i+1}: {summary['title']}\n\n{summary['key_elements']}"
            for i, summary in enumerate(all_key_elements)
        ])

        cache_key = "cycle_analysis"
        cached_data = self.cache.get(summaries_text, cache_key)

        if cached_data:
            self.processing_stats['cache_hits_cycle'] += 1
            print("  ✓ Анализ ЦИКЛА взят из кеша")
            return json.loads(cached_data)

        genre_context = (
            f"\n{'='*60}\nКОНТЕКСТ ЖАНРА: {self.genre}\n{'='*60}\n"
            f"Оценивай цикл с учетом этого жанра.\n"
            if self.genre else ""
        )

        prompt = f"""Ты — главный литературный критик. Проведи АНАЛИЗ ЦИКЛА романов на основе ключевых элементов (выжимок) каждой книги.

{genre_context}

{UNIVERSAL_CRITERIA_WITH_RATINGS}

{RATING_SCALE}

ИСХОДНЫЕ ДАННЫЕ (Выжимки по книгам):
{summaries_text}

═══════════════════════════════════════════════════════════════

СТРУКТУРА АНАЛИЗА ЦИКЛА:

**0. ИТОГОВАЯ ОЦЕНОЧНАЯ ТАБЛИЦА (ПО ВСЕМУ ЦИКЛУ)**
(ОБЯЗАТЕЛЬНО ЗАПОЛНИ эту таблицу для ЦИКЛА В ЦЕЛОМ. Оценивай цикл как единое произведение.)

| Критерий | Оценка (1–10) | Комментарий |
| :--- | :--- | :--- |
| Язык и стиль | ... | [Оцени рост или деградацию стиля] |
| Композиция и структура | ... | [Оцени связность всего цикла] |
| Персонажи и развитие | ... | [Оцени качество арок персонажей в цикле] |
| Диалоги | ... | ... |
| Мир и сеттинг | ... | [Оцени итоговую глубину мира] |
| Темы и идеи | ... | [Оцени развитие тем через весь цикл] |
| Оригинальность | ... | ... |
| Темп и драматургия | ... | [Оцени общий темп цикла] |
| Целостность тона | ... | ... |
| **Общее впечатление** | **...** | **[Итоговый комментарий по циклу]** |

---
**1. ЦЕЛОСТНОСТЬ ЦИКЛА И КОНСИСТЕНТНОСТЬ** (300-400 слов)
   - Единство замысла, связность книг.
   - **(Важно!)** Обязательно используй данные из разделов 'ГЛАВНЫЕ НЕСООТВЕТСТВИЯ' в выжимках, чтобы оценить общую консистентность цикла. Насколько автор был внимателен к деталям?

**2. РАЗВИТИЕ ПЕРСОНАЖЕЙ В ЦИКЛЕ** (300-400 слов)
   - Эволюция главных героев через все книги. Качество их арок (от и до).
   - Развитие второстепенных персонажей.

**3. СКВОЗНЫЕ СЮЖЕТНЫЕ ЛИНИИ И РАЗВИТИЕ ИДЕЙ** (300-400 слов)
   - Главные арки цикла и качество их разрешения.
   - **(Важно!)** Как **эволюционировали** ключевые идеи и темы: от простого к сложному, логичность развития. (Используй данные 'Темы' из выжимок).

**4. ТЕМАТИЧЕСКОЕ ЕДИНСТВО** (250-300 слов)
   - Развитие и углубление тем через цикл.

**5. ДИНАМИКА КАЧЕСТВА И РАБОТА НАД ОШИБКАМИ** (250-350 слов)
   - **(Важно!)** Как **менялось качество** от книги к книге? Улучшался ли язык?
   - **(Важно!)** Учитывал ли автор ошибки, найденные в предыдущих книгах (судя по заметкам о несоответствиях)? "Скатился" ли цикл или "вырос"?

**6. ОБЩАЯ ОЦЕНКА ЦИКЛА** (150-200 слов)
   - Итоговое письменное резюме (обоснование для 'Общего впечатления' в таблице), рекомендации.

Проведи профессиональный, но доступный анализ на русском языке.
"""

        self.logger.log('prompt', prompt, {
            'model': self.config.smart_model,
            'type': 'cycle_analysis_prompt'
        })

        response = self.smart_model.generate_content(
            prompt
        )

        self.processing_stats['api_calls'] += 1
        cycle_analysis_text = response

        self.logger.log('response', cycle_analysis_text, {
            'type': 'cycle_analysis'
        })
        print("  ✓ Анализ цикла получен.")

        analysis_data = {
            'cycle_analysis': cycle_analysis_text,
            'timestamp': datetime.now().isoformat(),
            'books_count': len(all_key_elements)
        }

        self.cache.set(summaries_text, cache_key, json.dumps(analysis_data, ensure_ascii=False), {
            'books': len(all_key_elements)
        })

        return analysis_data

    # ========================================================================
    # ГЛАВНЫЙ ПРОЦЕСС
    # ========================================================================

    def process_book_cycle(self, book_files: List[str], output_dir: str = 'analyses_map_reduce'):
        """
        Полный процесс анализа цикла: (Map -> Reduce) -> Cycle
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Начинаем анализ {len(book_files)} книг...")
        print(f"Результаты будут в: {self.output_dir.absolute()}")
        print(f"Логи сессии: {self.logger.session_dir.absolute()}")

        all_key_elements_for_cycle = []
        cumulative_inter_book_context = "" # Контекст *между книгами*

        for i, book_file in enumerate(tqdm(book_files, desc="Анализ цикла (по книгам)"), 1):
            print(f"\n{'='*80}")
            print(f"ОБРАБОТКА КНИГИ {i}/{len(book_files)}: {book_file}")
            print(f"{'='*80}")

            try:
                with open(book_file, 'r', encoding='utf-8') as f:
                    book_text = f.read()
                if len(book_text.split()) < 100:
                    print(f"  ! Файл '{book_file}' слишком мал, пропускаем.")
                    continue
            except FileNotFoundError:
                print(f"  ! ОШИБКА: Файл не найден: {book_file}. Пропускаем.")
                self.processing_stats['errors'] += 1
                continue
            except Exception as e:
                print(f"  ! ОШИБКА чтения файла {book_file}: {e}. Пропускаем.")
                self.processing_stats['errors'] += 1
                continue

            book_title = book_text.split('\n')[0].strip() or f"Книга {i}"

            # --- ЭТАП 1 (MAP): Анализ частей книги ---
            chunks = self._split_text_into_chunks(book_text, book_title)
            book_fact_summaries = []
            stats_lock = threading.Lock()
            max_workers = min(len(chunks), 5)

            def process_chunk(j, chunk, total_chunks, book_title):
                try:
                    chunk_summary = self._analyze_chunk(chunk, j, total_chunks, book_title)
                    with stats_lock:
                        self.processing_stats['chunks_processed'] += 1
                    if chunk_summary:
                        return (j, f"--- ДАННЫЕ ИЗ ЧАСТИ {j}/{total_chunks} ---\n{chunk_summary}")
                except Exception as e:
                    print(f"  ! Ошибка в чанке {j}: {e}")
                    with stats_lock:
                        self.processing_stats['errors'] += 1
                return (j, None)

            results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_chunk, j, chunk, len(chunks), book_title): j
                           for j, chunk in enumerate(chunks, 1)}
                with tqdm(total=len(chunks), desc=f"  Извлечение данных '{book_title}' (по частям)") as pbar:
                    for future in as_completed(futures):
                        try:
                            j, summary = future.result()
                            if summary:
                                results[j] = summary
                        except Exception as e:
                            print(f"  ! Ошибка потока: {e}")
                        pbar.update(1)

            book_fact_summaries = [results[j] for j in sorted(results.keys()) if results[j]]

            full_fact_summaries_text = "\n\n".join(book_fact_summaries)
            if not full_fact_summaries_text:
                print(f"  ! КРИТИЧЕСКАЯ ОШИБКА: Не удалось извлечь данные для книги {book_title}. Книга пропущена.")
                continue

            print(f"  ✓ Все {len(chunks)} частей книги '{book_title}' обработаны.")

            # --- ЭТАП 2 (REDUCE): Синтез анализа книги ---
            try:
                full_analysis, key_elements = self._synthesize_book_analysis(
                    full_fact_summaries_text,
                    book_title,
                    i,
                    book_text,
                    cumulative_inter_book_context
                )
            except Exception as e:
                print(f"  ! КРИТИЧЕСКАЯ ОШИБКА синтеза книги {book_title}: {e}")
                self.processing_stats['errors'] += 1
                full_analysis, key_elements = None, None

            if full_analysis and key_elements:
                print(f"  ✓ Синтезирован полный анализ для '{book_title}'.")
                self.processing_stats['books_processed'] += 1

                analysis_data = {
                    'book_number': i,
                    'title': book_title,
                    'analysis': full_analysis,
                    'key_elements_for_cycle': key_elements,
                    'timestamp': datetime.now().isoformat()
                }
                output_file = self.output_dir / f"book_{i:02d}_{book_title.replace(' ', '_')[:30]}_analysis.json"
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis_data, f, ensure_ascii=False, indent=4)
                    print(f"  ✓ Полный анализ сохранён: {output_file}")
                except Exception as e:
                    print(f"  ! Ошибка сохранения JSON: {e}")

                self.book_analyses_for_report[i] = analysis_data

                all_key_elements_for_cycle.append({
                    'title': book_title,
                    'key_elements': key_elements
                })

                cumulative_inter_book_context += f"--- Книга {i}: {book_title} (Ключевые элементы) ---\n{key_elements}\n\n"
            else:
                print(f"  ! КРИТИЧЕСКАЯ ОШИБКА: Не удалось синтезировать анализ для {book_title}.")

            if i < len(book_files):
                self._api_pause()

            if self.single_book_mode:
                print(f"\n✓ РЕЖИМ ОДНОЙ КНИГИ: Анализ завершен после первой книги '{book_title}'.")
                break

        cycle_analysis_data = None

        if not self.single_book_mode and all_key_elements_for_cycle:
            self._api_pause()
            try:
                cycle_analysis_data = self._analyze_cycle(all_key_elements_for_cycle)
                if cycle_analysis_data:
                    output_file = self.output_dir / "ZZZ_CYCLE_ANALYSIS.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(cycle_analysis_data, f, ensure_ascii=False, indent=4)
                    print(f"✓ Анализ цикла сохранён: {output_file}")
                else:
                    raise ValueError("Анализ цикла вернул None")
            except Exception as e:
                print(f"  ! КРИТИЧЕСКАЯ ОШИБКА анализа цикла: {e}")
                self.processing_stats['errors'] += 1
        elif self.single_book_mode:
            print("✓ РЕЖИМ ОДНОЙ КНИГИ: Пропуск финального анализа цикла.")
        else:
            print("! Не найдено ни одной выжимки (key_elements), анализ цикла невозможен.")

        self.create_final_report(cycle_analysis_data)
        self.print_stats()

    # ========================================================================
    # ФИНАЛЬНАЯ СБОРКА ОТЧЕТА И СТАТИСТИКА
    # ========================================================================

    def create_final_report(self, cycle_data: Dict = None):
        """Создание финального отчёта в markdown"""
        print(f"\n→ Создание финального отчета 'FULL_REPORT.md'...")
        report_lines = ["# ЛИТЕРАТУРНЫЙ АНАЛИЗ ЦИКЛА РОМАНОВ\n\n"]
        report_lines.append(f"*Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        report_lines.append(f"*Pro-модель: {self.config.smart_model}*\n")
        report_lines.append(f"*Flash-модель: {self.config.fast_model}*\n")
        if self.genre:
             report_lines.append(f"*Жанр: {self.genre}*\n")
        report_lines.append("\n---\n\n")

        if cycle_data and 'cycle_analysis' in cycle_data:
            report_lines.append("## АНАЛИЗ ЦИКЛА В ЦЕЛОМ\n\n")
            report_lines.append(cycle_data['cycle_analysis'])
            report_lines.append("\n\n---\n\n")
        elif self.single_book_mode:
             report_lines.append("## АНАЛИЗ ЦИКЛА НЕ ВЫПОЛНЯЛСЯ (Режим одной книги)\n\n")
             report_lines.append("\n\n---\n\n")
        else:
            report_lines.append("## АНАЛИЗ ЦИКЛА НЕ УДАЛСЯ ИЛИ НЕ БЫЛ ЗАПУЩЕН\n\n")
            report_lines.append("\n\n---\n\n")

        report_lines.append("# АНАЛИЗ ОТДЕЛЬНЫХ КНИГ ЦИКЛА\n\n")
        if not self.book_analyses_for_report:
            report_lines.append("*Не удалось проанализировать ни одной книги.*\n")

        for i in sorted(self.book_analyses_for_report.keys()):
            analysis = self.book_analyses_for_report[i]
            report_lines.append(f"## КНИГА {i}: {analysis['title']}\n\n")
            report_lines.append(analysis['analysis'])
            report_lines.append("\n\n---\n\n")

        report_file = self.output_dir / "FULL_REPORT.md"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.writelines(report_lines)
            print(f"✓ Финальный отчёт успешно создан: {report_file}")
        except Exception as e:
            print(f"  ! ОШИБКА сохранения финального отчета: {e}")

    def print_stats(self):
        """Вывод итоговой статистики"""
        end_time = datetime.now()
        duration = end_time - self.processing_stats['start_time']
        print(f"\n{'='*80}")
        print("АНАЛИЗ ЗАВЕРШЁН!")
        print(f"Время выполнения: {str(duration).split('.')[0]}")
        print(f"  Книг обработано: {self.processing_stats['books_processed']}")
        print(f"  Частей обработано: {self.processing_stats['chunks_processed']}")
        print(f"  Вызовов API: {self.processing_stats['api_calls']}")
        print("  Попаданий в кеш:")
        print(f"    - Части (Flash): {self.processing_stats['cache_hits_chunk']}")
        print(f"    - Книги (Pro): {self.processing_stats['cache_hits_book']}")
        if not self.single_book_mode:
            print(f"    - Цикл (Pro): {self.processing_stats['cache_hits_cycle']}")
        print(f"  Ошибок: {self.processing_stats['errors']}")
        print(f"Результаты сохранены в: {self.output_dir.absolute()}")
        print(f"Логи сессии: {self.logger.session_dir.absolute()}")
        print(f"{'='*80}")

if __name__ == "__main__":
    import argparse

    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(
        description='Анализатор книжного цикла с использованием AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s book-1.txt book-2.txt book-3.txt
  %(prog)s *.txt --genre "Фантастика" --output reports
  %(prog)s book.txt --single-book --chunk-words 5000
        """
    )

    # Позиционные аргументы
    parser.add_argument(
        'files',
        nargs='*',
        help='Пути к файлам с текстами романов (.txt или .md)'
    )

    # Опциональные аргументы
    parser.add_argument(
        '--output', '-o',
        default='critical_analysis_report',
        help='Директория для сохранения результатов (по умолчанию: critical_analysis_report)'
    )

    parser.add_argument(
        '--genre', '-g',
        default='Киберпанк, технотриллер, кибернуар',
        help='Жанр произведения (по умолчанию: "Киберпанк, технотриллер, кибернуар")'
    )

    parser.add_argument(
        '--single-book',
        action='store_true',
        help='Режим анализа одной книги'
    )

    parser.add_argument(
        '--chunk-words',
        type=int,
        default=10000,
        help='Целевой размер чанка в словах (по умолчанию: 10000)'
    )

    parser.add_argument(
        '--api-delay',
        type=float,
        default=2.0,
        help='Задержка между API-запросами в секундах (по умолчанию: 2.0)'
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Максимальное количество повторных попыток при ошибках (по умолчанию: 3)'
    )

    parser.add_argument(
        '--smart-model',
        default=DEFAULT_SMART_MODEL,
        help=f'Модель для сложных задач (по умолчанию: {DEFAULT_SMART_MODEL})'
    )

    parser.add_argument(
        '--fast-model',
        default=DEFAULT_FAST_FACTS_MODEL,
        help=f'Модель для быстрых задач (по умолчанию: {DEFAULT_FAST_FACTS_MODEL})'
    )

    # Парсим аргументы
    args = parser.parse_args()

    # 1. Проверяем API-ключ
    if not os.getenv('OR_AI_API_KEY'):
        print("=" * 80)
        print("!!! ВНИМАНИЕ: API-ключ 'OR_AI_API_KEY' не найден в переменных окружения.")
        print("Скрипт не сможет работать. Установите ключ и перезапустите.")
        print("=" * 80)
        parser.exit(1)

    # 2. Определяем список файлов
    if args.files:
        book_files = [name for name in args.files if '.txt' in name or '.md' in name]
    else:
        print("Предупреждение: Файлы не переданы. Используются 'book-1.txt', 'book-2.txt'...")
        book_files = [
            'book-1.txt',
            'book-2.txt',
            'book-3.txt',
        ]

    # 3. Проверяем существование файлов
    existing_books = []
    for f in book_files:
        if Path(f).exists():
            existing_books.append(f)
        else:
            print(f"Предупреждение: Файл '{f}' не найден и будет пропущен.")

    if not existing_books:
        print("Ошибка: Не найдено ни одного существующего .txt/.md файла для анализа.")
        parser.exit(1)

    try:
        # 4. Инициализация анализатора с параметрами из argparse
        config = AnalyzerConfig(
            smart_model=args.smart_model,
            fast_model=args.fast_model,
            api_delay_seconds=args.api_delay,
            chunk_target_words=args.chunk_words,
            max_retries=args.max_retries,
            genre=args.genre,
            single_book_mode=args.single_book
        )
        analyzer = BookCycleAnalyzer(config)

        # 5. Запуск анализа
        analyzer.process_book_cycle(
            existing_books,
            output_dir=args.output
        )
    except Exception as e:
        print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА В ОСНОВНОМ ПРОЦЕССЕ: {e}")
        import traceback
        traceback.print_exc()
        parser.exit(1)
