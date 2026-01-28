from openai import OpenAI
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import json
import os
import sys
import re
import logging


DEFAULT_PRO_MODEL = 'anthropic/claude-sonnet-4.5'

COUNTERS = {'prompt': 0, 'response': 0, 'completion': 0}
BACK_SCHEMA = {
  "name": "backward_analysis_v1",
  "schema": {
      "type": "object",
      "required": [
        "chapter_number",
        "resolved_notes",
        "new_notes"
      ],
      "properties": {
        "chapter_number": {
          "type": "integer",
          "description": "the number of the chapter."
        },
        "resolved_notes": {
          "type": "array",
          "description": "list of notes that have been addressed in this chapter.",
          "items": {
            "type": "object",
            "required": [
              "note_id",
              "how_resolved"
            ],
            "properties": {
              "note_id": {
                "type": "string",
                "description": "unique identifier of the previously created note. example: ch21_n01"
              },
              "how_resolved": {
                "type": "string",
                "description": "explanation of how this note was addressed in the current text."
              }
            }
          }
        },
        "new_notes": {
          "type": "array",
          "description": "list of new notes identified in this chapter for future reference.",
          "items": {
            "type": "object",
            "required": [
              "id",
              "type",
              "description",
              "specific_element",
              "urgency"
            ],
            "properties": {
              "id": {
                "type": "string",
                "description": "unique identifier for the new note."
              },
              "type": {
                "type": "string",
                "enum": [
                  "character",
                  "event",
                  "object",
                  "skill",
                  "relationship"
                ],
                "description": "category of the note."
              },
              "description": {
                "type": "string",
                "description": "what needs to be mentioned or addressed earlier."
              },
              "specific_element": {
                "type": "string",
                "description": "the specific text element triggering this note."
              },
              "urgency": {
                "type": "string",
                "enum": [
                  "critical",
                  "important",
                  "minor"
                ],
                "description": "the urgency level of addressing this note."
              }
            }
          }
        }
      }
  }
}
FORWARD_SCHEMA = {
  "name": "forward_analysis_v1",
  "schema": {
      "type": "object",
      "required": [
        "chapter_number",
        "revised_content",
        "changes_made"
      ],
      "properties": {
        "chapter_number": {
          "type": "integer",
          "description": "the number of the chapter."
        },
        "revised_content": {
          "type": "string",
          "description": "the complete, revised text of the chapter."
        },
        "changes_made": {
          "type": "array",
          "description": "list of specific changes made to the text.",
          "items": {
            "type": "object",
            "required": [
              "addresses_note_id",
              "description",
              "location"
            ],
            "properties": {
              "addresses_note_id": {
                "type": "string",
                "description": "the id of the note that prompted this change."
              },
              "description": {
                "type": "string",
                "description": "description of what was added or modified."
              },
              "location": {
                "type": "string",
                "description": "where in the text the change occurred (e.g., paragraph number or a short quote)."
              }
            }
          }
        }
    }
  }
}


def log_me(kind, text):
    Path('request_logs').mkdir(exist_ok=True)
    COUNTERS[kind] += 1
    number = COUNTERS[kind]
    f = open(f'request_logs/{kind}_{number:02}.md', 'a')
    f.write(text)
    f.close()

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


class AiClient:
    def __init__(self, model):
        self.model = model
        self.client = client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OR_AI_API_KEY')
        )

    def generate_content(self, prompt, *args, **kwargs):
        log_me('prompt', prompt)
        schema = kwargs.get('schema', {})
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": schema,
            },
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


class BookConsistencyChecker:
    def __init__(self, api_key: str):
        """Инициализация с API ключом Gemini"""
        self.model = AiClient(DEFAULT_PRO_MODEL)

    def read_book(self, file_path: str) -> List[Dict[str, str]]:
        """Читает книгу и разбивает на главы"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Простое разбиение по главам (можно адаптировать под формат)
        chapters = []
        lines = content.split('\n')
        current_chapter = {'title': '', 'content': ''}
        chapter_num = 0

        for line in lines:
            # Определяем начало главы (можно настроить паттерн)
            if line.strip().lower().startswith(('глава', 'chapter', 'часть')):
                if current_chapter['content']:
                    chapters.append(current_chapter)
                chapter_num += 1
                current_chapter = {
                    'number': chapter_num,
                    'title': line.strip(),
                    'content': ''
                }
            else:
                current_chapter['content'] += line + '\n'

        if current_chapter['content']:
            chapters.append(current_chapter)

        return chapters

    def analyze_backwards(self, chapters: List[Dict[str, str]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Этап 1: Анализ от последней главы к первой
        Собираем требования, которые должны быть упомянуты ранее
        """
        print("\n=== ЭТАП 1: Анализ от конца к началу ===\n")

        accumulated_notes = []
        analysis_results = []

        # Идём от последней главы к первой
        for i in range(len(chapters) - 1, -1, -1):
            chapter = chapters[i]
            print(f"Анализирую главу {chapter['number']}: {chapter['title']}")

            prompt = f"""
Ты анализируешь книгу методом обратного чтения для проверки согласованности.

ТЕКУЩАЯ ГЛАВА #{chapter['number']}: {chapter['title']}

СОДЕРЖАНИЕ ГЛАВЫ:
{chapter['content']}

НАКОПЛЕННЫЕ ЗАМЕЧАНИЯ из последующих глав:
{json.dumps(accumulated_notes, ensure_ascii=False, indent=2)}

ЗАДАЧИ:
1. Найди элементы в этой главе, которые требуют предварительной подготовки/упоминания в предыдущих главах:
   - Персонажи, появляющиеся без введения
   - События, требующие предыстории
   - Предметы/локации без описания
   - Навыки/знания без объяснения их получения
   - Отношения между персонажами без развития

2. Проверь НАКОПЛЕННЫЕ ЗАМЕЧАНИЯ: какие из них УДОВЛЕТВОРЕНЫ в этой главе?
   - Если замечание учтено - укажи как "resolved"
   - Если замечание из накопленных дублируется в новых замечаниях - укажи как "resolved" и "how_resolved": "duplicates"
   - Если не учтено - оставь как "active"

3. Добавь НОВЫЕ ЗАМЕЧАНИЯ для предыдущих глав

Ответь в формате JSON:
{{
  "chapter_number": {chapter['number']},
  "resolved_notes": [
    {{"note_id": "id_замечания", "how_resolved": "как учтено в этой главе"}}
  ],
  "new_notes": [
    {{
      "id": "ch21_n_01",
      "type": "character/event/object/skill/relationship",
      "description": "что должно быть упомянуто ранее",
      "specific_element": "конкретный элемент из текста",
      "urgency": "critical/important/minor"
    }}
  ]
}}
"""

            try:
                response = self.model.generate_content(prompt, schema=BACK_SCHEMA)
                result = robust_json_parser(response)

                # Обновляем список замечаний
                # Удаляем решённые
                resolved_ids = [n['note_id'] for n in result.get('resolved_notes', [])]
                accumulated_notes = [n for n in accumulated_notes if n['id'] not in resolved_ids]

                # Добавляем новые
                for note in result.get('new_notes', []):
                    note['found_in_chapter'] = chapter['number']
                    accumulated_notes.append(note)

                analysis_results.append({
                    'chapter': chapter['number'],
                    'result': result,
                    'active_notes': len(accumulated_notes)
                })

                print(f"  ✓ Найдено новых замечаний: {len(result.get('new_notes', []))}")
                print(f"  ✓ Решено замечаний: {len(result.get('resolved_notes', []))}")
                print(f"  ✓ Активных замечаний: {len(accumulated_notes)}\n")

            except Exception as e:
                print(f"  ✗ Ошибка: {e}\n")
                raise e

        return accumulated_notes, analysis_results

    def fix_forward(self, chapters: List[Dict[str, str]], notes: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Этап 2: Правка от первой главы к последней
        Применяем найденные замечания для улучшения согласованности
        Возвращает исправленные главы
        """
        print("\n=== ЭТАП 2: Правка от начала к концу ===\n")

        fixed_chapters = []
        fixes_log = []
        addressed_notes = set()

        for chapter in chapters:
            # Находим замечания, относящиеся к этой или предыдущим главам
            relevant_notes = [n for n in notes if n['found_in_chapter'] > chapter['number'] and n['id'] not in addressed_notes]

            if not relevant_notes:
                print(f"Глава {chapter['number']}: нет замечаний для правки")
                fixed_chapters.append(chapter)
                continue

            print(f"Исправляю главу {chapter['number']}: {chapter['title']}")

            prompt = f"""
Ты редактор, улучшающий согласованность книги.

ТЕКУЩАЯ ГЛАВА #{chapter['number']}: {chapter['title']}

ИСХОДНОЕ СОДЕРЖАНИЕ:
{chapter['content']}

ЗАМЕЧАНИЯ, которые нужно учесть (из последующих глав):
{json.dumps(relevant_notes, ensure_ascii=False, indent=2)}

ЗАДАЧА:
Перепиши главу, органично добавив упоминания/подготовку для элементов из замечаний:
1. Сохрани весь оригинальный текст и сюжет
2. Добавь необходимые упоминания/описания естественным образом
3. Не меняй стиль автора
4. Подготовь почву для событий из последующих глав

Ответь в формате JSON:
{{
  "chapter_number": {chapter['number']},
  "revised_content": "ПОЛНЫЙ исправленный текст главы",
  "changes_made": [
    {{
      "addresses_note_id": "id_замечания",
      "description": "что было добавлено/изменено",
      "location": "где в тексте"
    }}
  ]
}}
"""

            try:
                response = self.model.generate_content(prompt, schema=FORWARD_SCHEMA)
                result = robust_json_parser(response)

                fixed_chapter = chapter.copy()
                fixed_chapter['content'] = result['revised_content']
                fixed_chapter['original_content'] = chapter['content']
                fixed_chapters.append(fixed_chapter)

                fixes_log.append({
                    'chapter': chapter['number'],
                    'changes': result.get('changes_made', [])
                })
                for edit in result.get('changes_made',[]):
                    addressed_notes.add(edit.get('addresses_note_id'))

                print(f"  ✓ Внесено изменений: {len(result.get('changes_made', []))}\n")

            except Exception as e:
                print(f"  ✗ Ошибка: {e}")
                print(f"  Сохраняю главу без изменений\n")
                fixed_chapters.append(chapter)

        return fixed_chapters, fixes_log

    def generate_report(self, analysis_results: List[Dict], fixes_log: List[Dict], output_path: str):
        """Генерирует итоговый отчёт"""
        report = {
            'summary': {
                'total_chapters_analyzed': len(analysis_results),
                'total_notes_found': sum(r['active_notes'] for r in analysis_results),
                'total_chapters_fixed': len(fixes_log),
                'total_changes_made': sum(len(f.get('changes', [])) for f in fixes_log)
            },
            'backward_analysis': analysis_results,
            'forward_fixes': fixes_log
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n=== ОТЧЁТ СОХРАНЁН: {output_path} ===")
        print(f"Проанализировано глав: {report['summary']['total_chapters_analyzed']}")
        print(f"Найдено несоответствий: {report['summary']['total_notes_found']}")
        print(f"Исправлено глав: {report['summary']['total_chapters_fixed']}")
        print(f"Всего изменений: {report['summary']['total_changes_made']}")

    def save_fixed_book(self, chapters: List[Dict[str, str]], output_path: str):
        """Сохраняет исправленную книгу в один файл"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for chapter in chapters:
                f.write(f"{chapter['title']}\n\n")
                f.write(chapter['content'])
                f.write("\n\n" + "="*60 + "\n\n")

        print(f"\n=== ИСПРАВЛЕННАЯ КНИГА СОХРАНЕНА: {output_path} ===")

    def process_book(self, book_path: str, output_dir: str = 'output'):
        """Полный процесс обработки книги"""
        Path(output_dir).mkdir(exist_ok=True)

        print("Чтение книги...")
        chapters = self.read_book(book_path)
        print(f"Найдено глав: {len(chapters)}\n")

        # Этап 1: Обратный анализ
        notes, analysis_results = self.analyze_backwards(chapters)

        # Сохраняем найденные замечания
        with open(f'{output_dir}/notes.json', 'w', encoding='utf-8') as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)

        # Этап 2: Прямая правка
        fixed_chapters, fixes_log = self.fix_forward(chapters, notes)

        # Сохраняем исправленную книгу
        self.save_fixed_book(fixed_chapters, f'{output_dir}/book_fixed.txt')

        # Генерируем отчёт
        self.generate_report(analysis_results, fixes_log, f'{output_dir}/report.json')

        return notes, fixed_chapters


# Пример использования
if __name__ == "__main__":
    # Получаем API ключ
    API_KEY = os.getenv('OR_AI_API_KEY') or input("Введите Gemini API ключ: ")

    checker = BookConsistencyChecker(API_KEY)

    book_file = sys.argv[1]

    notes, fixed_chapters = checker.process_book(book_file)

    print("\n" + "="*60)
    print("ОБРАБОТКА ЗАВЕРШЕНА!")
    print("="*60)
    print("\nФайлы созданы в папке 'output/':")
    print("  - book_fixed.txt - исправленная книга")
    print("  - notes.json - все найденные замечания")
    print("  - report.json - полный отчёт с изменениями")
