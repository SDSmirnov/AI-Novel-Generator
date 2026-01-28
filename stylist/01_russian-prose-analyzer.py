#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import google.generativeai as genai
import json
import logging
import os
import re
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from google.api_core import retry
from google.genai.errors import APIError
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from typing import Any, Dict, Optional, Tuple, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prose_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    status: str  # 'OK', 'SLIGHT_REPETITION', 'STRONG_CLICHE'
    motive: str
    diagnosis: str
    recommendation: str
    confidence: float

MAX_WORKERS = 16

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

SYSTEM_PROMPT = """
### Use Praise Calibration and Avoidance of Sycophancy (Anti-Pleasing Protocol):
* Do not use phrases that exaggerate the user's merits ("brilliant," "titanic," "amazing"). Instead, use effort validation techniques: "I can see how hard you are trying...".
* Do not express admiration for observations that are obvious to the user. Acknowledge their importance while maintaining a calm and neutral tone.
* Limit direct praise to once per session, and only if the user clearly needs support and a self-esteem boost.

РОЛЬ: Вы — опытный литературный редактор, специализирующийся на русской художественной прозе. Ваша задача — выявлять **стилистические самоповторы, которые ослабляют эмоциональную динамику**.

Жанр романа - современная проза, эротический триллер, dark romance.

Проанализируйте группу семантически похожих фрагментов текста и определите, является ли повтор **необходимым лейтмотивом** (OK) или **стилистической инерцией** (SLIGHT_REPETITION/STRONG_CLICHE).

Особое внимание уделите:
- **Кинетике:** Повторяемость поз (например, "откинулся"), жестов (например, "усмешка губами") и движений (например, "медленно повернул голову").
- **Сенсорике:** Чрезмерное использование одного и того же сенсорного маркера (например, "холодный узел", "писк дросселей") для описания напряжения.
- **Нарративной функции:** Определите, служит ли повтор развитию характера или же просто является "заглушкой" для обозначения паузы/ожидания/перехода.
- **Физиология:** Повторяемость физиологических ощущений
- **Психологизм:** Называние эмоций вместо показа

Учитывайте:
- Расстояние между фрагментами (в разных главах повторы более допустимы)
- Художественную ценность повтора
- Частотность использования приёма

Ответьте **строго** в формате JSON. Ваши рекомендации должны быть максимально конкретными, предлагая стилистически более разнообразные альтернативы.
ПРАВИЛА ДЛЯ ОТВЕТА:
1. Используйте ТОЛЬКО двойные кавычки (") для ключей и значений JSON.
2. Не используйте одинарные кавычки (') внутри значений. Вместо них используйте русские кавычки-елочки («»).
3. Не включайте никакой текст или пояснения вне блока JSON
Формат JSON:
{
    "status": "OK/SLIGHT_REPETITION/STRONG_CLICHE",
    "motive": "Краткое описание повторяющегося мотива (напр., 'Фиксация напряжения через холодный взгляд Ушакова').",
    "diagnosis": "Детальный анализ причины повтора. Укажите, в каких главах наблюдается дублирование. Оцените, не является ли это шаблонным способом показать эмоцию (например, 'хриплый голос' для стресса).",
    "recommendation": "Конкретные, стилистически разнообразные предложения по замене или варьированию. Например: 'Заменить 'усмешку' на 'напряжение челюсти', 'позу кресла' на 'медленное растирание висков'.'",
    "confidence": 0.0-1.0
}
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": [
                "OK",
                "SLIGHT_REPETITION",
                "STRONG_CLICHE"
            ],
            "description": "The overall evaluation status."
        },
        "motive": {
            "type": "string",
            "description": "A brief description of the repeating motive (e.g., 'Fixation of tension through a cold stare')."
        },
        "diagnosis": {
            "type": "string",
            "description": "A detailed analysis of the reason for repetition, including chapter locations and an assessment of cliches."
        },
        "recommendation": {
            "type": "string",
            "description": "Specific, stylistically diverse suggestions for replacement or variation."
        },
        "confidence": {
            "type": "number",
            #"minimum": 0.0,
            #"maximum": 1.0,
            "description": "The confidence score of the analysis, from 0.0 to 1.0."
        }
    },
    "required": [
        "status",
        "motive",
        "diagnosis",
        "recommendation",
        "confidence"
    ]
}

@retry.Retry(predicate=retry.if_transient_error, deadline=60)
def generate_embedding(text: str):
    """Генерация эмбеддинга с ретраями"""
    result = genai.embed_content(
        model='models/text-embedding-004',
        content=text,
        task_type="retrieval_document",
        title="Text chunk"
    )
    return result['embedding']

def embed_chunk(chunk: TextChunk, embeddings: Dict[int, List[float]], len_chunks: int):
    """Целевая функция для многопоточного запуска генерации эмбеддинга"""
    try:
        embedding = generate_embedding(chunk.text)
        embeddings[chunk.chunk_id] = embedding
        if chunk.chunk_id % 50 == 0:
            logger.info(f"Обработано {chunk.chunk_id + 1}/{len_chunks} чанков")

    except Exception as e:
        logger.error(f"Ошибка при генерации эмбеддинга для чанка {chunk.chunk_id}: {e}")
        embeddings[chunk.chunk_id] = [0.0] * 768

def analyze_one_cluster(x_self, cluster, len_clusters, results):
    try:
        # Формируем запрос для модели
        fragments_text = f"ОРИГИНАЛЬНЫЙ ФРАГМЕНТ (Глава {cluster.original.chapter}):\n"
        fragments_text += f'"{cluster.original.text}"\n\n'
        fragments_text += "ПОХОЖИЕ ФРАГМЕНТЫ:\n"

        for similar_chunk, score in cluster.similar_chunks[:4]:
            fragments_text += f"- Глава {similar_chunk.chapter} (схожесть: {score:.2%}):\n"
            fragments_text += f'  "{similar_chunk.text}"\n\n'

        prompt = f"Проанализируйте следующие фрагменты:\n\n{fragments_text}"

        response_text = x_self._call_llm_analysis(prompt)
        analysis_data = json.loads(response_text)
        result = AnalysisResult(
            cluster=cluster,
            status=analysis_data.get('status', 'OK'),
            motive=analysis_data.get('motive', ''),
            diagnosis=analysis_data.get('diagnosis', ''),
            recommendation=analysis_data.get('recommendation', ''),
            confidence=float(analysis_data.get('confidence', 0.5))
        )

        results.append(result)
        if len(results) % 5 == 0:
            logger.info(f"Проанализировано {len(results)} кластеров")
        return result

    except Exception as e:
        logger.error(f"Критическая ошибка при LLM анализе кластера {cluster.cluster_id}: {e}")
        if isinstance(e, APIError):
            logger.error("API ошибка. Прерывание анализа LLM.")

class RussianProseAnalyzer:
    def __init__(self, gemini_api_key: str, qdrant_host: str = "localhost", qdrant_port: int = 6333):

        self.gemini_api_key = gemini_api_key
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

        genai.configure(api_key=gemini_api_key)
        self.embedding_model = "models/text-embedding-004"
        self.analysis_model = genai.GenerativeModel(
            model_name="gemini-flash-lite-latest",
            system_instruction=SYSTEM_PROMPT,
            safety_settings=SAFETY_SETTINGS)

        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = f"prose_chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.vector_size = 768

        self.similarity_threshold = 0.9
        self.max_similar_chunks = 5
        self._chapters: List[str] = []
        logger.info("Анализатор инициализирован успешно")

    def load_text(self, file_path: str) -> str:
        """Загрузка текста из файла"""
        logger.info(f"Загружаем текст из {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Загружено {len(content)} символов")
            return content
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {e}")
            raise

    def create_chunks(self, text: str) -> List[TextChunk]:
        chunks = []
        chunk_id = 0

        # Паттерны для поиска структурных элементов
        chapter_pattern = r'(?:## Глава|ГЛАВА|Chapter|CHAPTER)\s+(?:\d+|[IVXLCDM]+)'
        paragraph_pattern = r'\n'

        chapters = re.split(f'({chapter_pattern})', text)
        current_chapter = 0
        self._chapters = []

        for i in range(len(chapters)):
            if re.match(chapter_pattern, chapters[i]):
                current_chapter += 1
                logger.info(f"Обрабатываем главу {current_chapter}")
                continue

            chapter_text = chapters[i]
            if not chapter_text.strip():
                continue

            paragraphs = re.split(paragraph_pattern, chapter_text)
            self._chapters.append(paragraphs)

            joiner = ""
            for para_idx, paragraph in enumerate(paragraphs):
                paragraph = joiner + "\n" + paragraph if joiner else paragraph
                joiner = ""

                if not paragraph.strip():
                    continue

                if len(paragraph.strip()) < 20 or len(paragraph.strip().split(' ')) < 3:
                    joiner = paragraph.strip()
                    continue

                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=paragraph,
                    chapter=current_chapter,
                    paragraph_id=para_idx,
                )
                chunks.append(chunk)
                chunk_id += 1
            logger.info(f"Создано {len(paragraphs)} чанков")

        logger.info(f"Создано {len(chunks)} чанков всего")
        return chunks

    def generate_embeddings(self, chunks: List[TextChunk]) -> Dict[int, List[float]]:
        """Генерация эмбеддингов для чанков с помощью Gemini (многопоточно)"""
        logger.info("Начинаем генерацию эмбеддингов...")
        embeddings = {}

        batch_size = 5
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {
                executor.submit(embed_chunk, chunk, embeddings, len(chunks)): chunk
                for chunk in chunks
            }
            for future in as_completed(future_to_chunk):
                try:
                    future.result()
                except Exception as e:
                    chunk = future_to_chunk[future]
                    logger.error(f"Критическая ошибка вне потока при обработке чанка {chunk.chunk_id}: {e}")
                    pass

        logger.info(f"Сгенерировано {len(embeddings)} эмбеддингов")
        return embeddings

    def create_qdrant_collection(self):
        """Создание коллекции в Qdrant"""
        logger.info(f"Создаём коллекцию {self.collection_name} в Qdrant")

        try:
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info("Коллекция создана успешно")
        except Exception as e:
            logger.error(f"Ошибка при создании коллекции: {e}")
            raise

    def store_embeddings(self, chunks: List[TextChunk], embeddings: Dict[int, List[float]]):
        """Сохранение эмбеддингов в Qdrant"""
        logger.info("Сохраняем эмбеддинги в Qdrant...")

        points = []
        for chunk in chunks:
            if chunk.chunk_id not in embeddings or not embeddings[chunk.chunk_id]:
                continue

            point = PointStruct(
                id=chunk.chunk_id,
                vector=embeddings[chunk.chunk_id],
                payload={
                    "text": chunk.text,
                    "chapter": chunk.chapter,
                    "paragraph_id": chunk.paragraph_id,
                }
            )
            points.append(point)

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            except Exception as e:
                logger.error(f"Ошибка при загрузке батча {i//batch_size}: {e}")

        logger.info(f"Загружено {len(points)} точек в Qdrant")


    def find_similar_chunks(self, chunks: List[TextChunk], embeddings: Dict[int, List[float]]) -> List[SimilarityCluster]:
        """
        Поиск семантически похожих чанков с исключением дублирующих пар и сильно пересекающихся чанков.
        """
        logger.info("Начинаем поиск похожих фрагментов...")
        clusters = []
        # Множество для хранения уникальных пар (min_id, max_id) для дедупликации
        chunks.sort(key=lambda c: -len(c.text))

        chunk_dict = {chunk.chunk_id: chunk for chunk in chunks}
        already_clustered = set()

        for chunk in chunks:
            if chunk.chunk_id not in embeddings or not embeddings[chunk.chunk_id]:
                continue

            try:
                already_clustered.add(chunk.chunk_id)
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=embeddings[chunk.chunk_id],
                    limit=(self.max_similar_chunks * 2) + 1,
                    score_threshold=self.similarity_threshold
                )

                similar_chunks = []
                for hit in search_results:
                    if str(hit.id) == str(chunk.chunk_id):
                        continue

                    similar_chunk = chunk_dict.get(hit.id)
                    if not similar_chunk:
                        continue

                    if hit.id in already_clustered:
                        continue

                    already_clustered.add(hit.id)

                    # Если проверки пройдены:
                    similar_chunks.append((similar_chunk, hit.score))
                    if len(similar_chunks) == self.max_similar_chunks:
                        break

                if similar_chunks:
                    cluster = SimilarityCluster(
                        cluster_id=len(clusters),
                        original=chunk,
                        similar_chunks=similar_chunks
                    )
                    clusters.append(cluster)

            except Exception as e:
                logger.error(f"Ошибка при поиске для чанка {chunk.chunk_id}: {e}")
                continue

        logger.info(f"Найдено {len(clusters)} дедуплицированных кластеров похожих фрагментов")
        return clusters

    @retry.Retry(predicate=retry.if_transient_error, deadline=120)
    def _call_llm_analysis(self, prompt: str) -> str:
        response = self.analysis_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                max_output_tokens=32000,
                response_mime_type='application/json',
                response_schema=RESPONSE_SCHEMA,
            )
        )
        if not response.candidates or not response.candidates[0].content.parts:
            finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
            raise APIError(f"LLM returned empty response. FinishReason: {finish_reason}")

        return response.text

    def analyze_clusters(self, clusters: List[SimilarityCluster]) -> List[AnalysisResult]:
        """Анализ кластеров с помощью Gemini для определения типа повтора"""
        logger.info("Начинаем анализ кластеров с помощью LLM...")
        results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_cluster = {
                executor.submit(analyze_one_cluster, self, cluster, len(clusters), results): cluster
                for cluster in clusters if len(cluster.similar_chunks) > 2
            }
            logger.info(f"Кластеры > 2 : {len(future_to_cluster.values())} из {len(clusters)}")
            for future in as_completed(future_to_cluster):
                cluster = future_to_cluster[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Ошибка при обработке кластера {cluster.cluster_id}: {e}")

        logger.info(f"Проанализировано {len(results)} кластеров")
        return results

    def generate_report(self, results: List[AnalysisResult], output_file: str = "analysis_report.md"):
        """Генерация отчёта по результатам анализа"""
        logger.info(f"Генерируем отчёт в {output_file}")

        results_sorted = sorted(results, key=lambda x: (
            (100 if x.status == 'STRONG_CLICHE' else 0) + (10 if x.status == 'SLIGHT_REPETITION' else 0) + x.confidence
        ), reverse=True)

        reporting_data = {
            'chapters': self._chapters,
            'analysis': [],
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Отчёт по анализу стилистических повторов и клише\n\n")
            f.write(f"**Дата анализа:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Всего проанализировано кластеров:** {len(results)}\n\n")

            stats = {
                'STRONG_CLICHE': sum(1 for r in results if r.status == 'STRONG_CLICHE'),
                'SLIGHT_REPETITION': sum(1 for r in results if r.status == 'SLIGHT_REPETITION'),
                'OK': sum(1 for r in results if r.status == 'OK')
            }

            f.write("## Статистика\n\n")
            f.write(f"- **Сильные клише:** {stats['STRONG_CLICHE']}\n")
            f.write(f"- **Незначительные повторы:** {stats['SLIGHT_REPETITION']}\n")
            f.write(f"- **Допустимые повторы:** {stats['OK']}\n\n")

            f.write("## Детальный анализ\n\n")

            for idx, result in enumerate(results_sorted):
                if result.status == 'OK' and idx > 20:
                    continue

                status_emoji = {
                    'STRONG_CLICHE': '🔴',
                    'SLIGHT_REPETITION': '🟡',
                    'OK': '🟢'
                }.get(result.status, '⚪')

                f.write(f"### {status_emoji} Кластер #{result.cluster.cluster_id}\n\n")
                f.write(f"**Статус:** {result.status}\n")
                f.write(f"**Уверенность:** {result.confidence:.0%}\n")
                f.write(f"**Мотив:** {result.motive}\n\n")

                f.write("**Оригинальный фрагмент** (Глава {}):\n".format(result.cluster.original.chapter))
                f.write(f"> {result.cluster.original.text}\n\n")

                f.write("**Похожие фрагменты:**\n")
                similar_part = []
                report_row = {
                    'cluster': result.cluster.cluster_id,
                    'status': result.status,
                    'confidence': result.confidence,
                    'motive': result.motive,
                    'diagnosis': result.diagnosis,
                    'recommendation': result.recommendation,
                    'original': {
                        'chapter': result.cluster.original.chapter,
                        'text': result.cluster.original.text,
                        'paragraph': result.cluster.original.paragraph_id,
                    },
                    'similar': similar_part,
                }
                reporting_data['analysis'].append(report_row)

                for chunk, score in result.cluster.similar_chunks[:5]:
                    similar_part.append({
                        'score': score,
                        'chapter': chunk.chapter,
                        'paragraph': chunk.paragraph_id,
                        'text': chunk.text,
                    })
                    f.write(f"- Глава {chunk.chapter} (схожесть {score:.0%}):\n")
                    f.write(f"  > {chunk.text[:600]}{'...' if len(chunk.text) > 600 else ''}\n\n")

                if result.diagnosis:
                    f.write(f"**Диагноз:** {result.diagnosis}\n\n")

                if result.recommendation and result.status != 'OK':
                    f.write(f"**Рекомендация:** {result.recommendation}\n\n")

                f.write("---\n\n")

        fout = open(f"{output_file}.json", "w")
        fout.write(json.dumps(reporting_data, indent=2, ensure_ascii=False))
        fout.close()
        logger.info(f"Отчёт успешно сохранён в {output_file}")

    def run_full_analysis(self, input_file: str, output_file: str = "analysis_report.md"):
        """Запуск полного цикла анализа"""
        try:
            text = self.load_text(input_file)
            chunks = self.create_chunks(text)

            if not chunks:
                logger.error("Не удалось создать чанки из текста")
                return

            embeddings = self.generate_embeddings(chunks)
            self.create_qdrant_collection()
            self.store_embeddings(chunks, embeddings)
            clusters = self.find_similar_chunks(chunks, embeddings)

            if not clusters:
                logger.info("Похожие фрагменты не найдены")
                return

            results = self.analyze_clusters(clusters)
            self.generate_report(results, output_file)
            logger.info("Анализ завершён успешно!")

        except Exception as e:
            logger.error(f"Критическая ошибка при анализе: {e}")
            raise
        finally:
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                logger.info("Временная коллекция удалена")
            except:
                pass


def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(
        description="Анализ русской художественной прозы на предмет стилистических повторов и клише"
    )
    parser.add_argument(
        "input_file",
        help="Путь к файлу с текстом для анализа"
    )
    parser.add_argument(
        "--output",
        default="analysis_report.md",
        help="Путь к файлу отчёта (по умолчанию: analysis_report.md)"
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API ключ (или установите переменную AI_API_KEY)"
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Хост Qdrant (по умолчанию: localhost)"
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Порт Qdrant (по умолчанию: 6333)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Порог схожести для поиска (по умолчанию: 0.85)"
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("AI_API_KEY")
    if not api_key:
        logger.error("Необходимо указать Gemini API ключ через --api-key или переменную окружения AI_API_KEY")
        return 1

    analyzer = RussianProseAnalyzer(
        gemini_api_key=api_key,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port
    )

    analyzer.similarity_threshold = args.threshold

    try:
        analyzer.run_full_analysis(args.input_file, args.output)
        return 0
    except Exception as e:
        logger.error(f"Ошибка при выполнении анализа: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
