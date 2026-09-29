"""Búsqueda de incidencias: palabras, significado y fusión de ambos rankings."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


DEMO_PATH = Path(__file__).parent / "data" / "incidencias_demo.json"
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
STOPWORDS = {
    "a", "al", "con", "de", "del", "debido", "el", "en", "la", "las",
    "los", "por", "que", "se", "un", "una", "y",
}


@dataclass(frozen=True)
class Incident:
    id: str
    texto: str
    tipo: str
    zona: str
    causa: str
    referencia: str | None

    @classmethod
    def from_record(cls, record: dict) -> Incident:
        return cls(**{field: record[field] for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SearchHit:
    incident: Incident
    score: float


def load_incidents(path: Path = DEMO_PATH) -> list[Incident]:
    records = json.loads(path.read_text(encoding="utf-8"))
    incidents = [Incident.from_record(record) for record in records]
    if len({incident.id for incident in incidents}) != len(incidents):
        raise ValueError("Cada incidencia debe tener un ID único.")
    return incidents


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return [word for word in re.findall(r"\w+", without_accents) if word not in STOPWORDS]


def matches(incident: Incident, filters: dict[str, str]) -> bool:
    return all(getattr(incident, field) == value for field, value in filters.items())


class BM25Retriever:
    """Puntúa coincidencias de palabras sobre el subconjunto filtrado."""

    def search(self, incidents: list[Incident], query: str, limit: int = 3) -> list[SearchHit]:
        if not incidents or limit < 1:
            return []
        terms = [Counter(tokenize(incident.texto)) for incident in incidents]
        lengths = [sum(document.values()) for document in terms]
        average_length = sum(lengths) / len(lengths)
        document_frequency = Counter(word for document in terms for word in document)
        scores = []
        for incident, document, length in zip(incidents, terms, lengths):
            score = 0.0
            for word in set(tokenize(query)):
                frequency = document[word]
                if not frequency:
                    continue
                found_in = document_frequency[word]
                idf = math.log(1 + (len(incidents) - found_in + 0.5) / (found_in + 0.5))
                denominator = frequency + 1.5 * (0.25 + 0.75 * length / average_length)
                score += idf * frequency * 2.5 / denominator
            if score > 0:
                scores.append(SearchHit(incident, score))
        return sorted(scores, key=lambda hit: (-hit.score, hit.incident.id))[:limit]


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEncoder:
    """Carga el modelo local solo cuando se pide búsqueda semántica."""

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Para búsqueda semántica instala: pip install -r requirements.txt"
            ) from exc
        self.model = SentenceTransformer(MODEL_ID)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Los vectores deben tener la misma dimensión.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class SemanticRetriever:
    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def search(self, incidents: list[Incident], query: str, limit: int = 3) -> list[SearchHit]:
        if not incidents or limit < 1:
            return []
        vectors = self.encoder.encode([query] + [incident.texto for incident in incidents])
        query_vector = vectors[0]
        hits = [
            SearchHit(incident, cosine(query_vector, vector))
            for incident, vector in zip(incidents, vectors[1:])
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.incident.id))[:limit]


def reciprocal_rank_fusion(*rankings: list[SearchHit], limit: int = 3) -> list[SearchHit]:
    """Combina posiciones, ya que BM25 y coseno tienen escalas distintas."""
    incidents: dict[str, Incident] = {}
    scores: Counter[str] = Counter()
    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            incidents[hit.incident.id] = hit.incident
            scores[hit.incident.id] += 1 / (60 + position)
    return [
        SearchHit(incidents[id], score)
        for id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def search(
    incidents: list[Incident], query: str, mode: str = "bm25",
    filters: dict[str, str] | None = None, limit: int = 3,
    encoder: Encoder | None = None,
) -> list[SearchHit]:
    candidates = [incident for incident in incidents if matches(incident, filters or {})]
    if not candidates or limit < 1:
        return []
    if mode == "bm25":
        return BM25Retriever().search(candidates, query, limit)
    if mode not in {"semantic", "hybrid"}:
        raise ValueError(f"Modo desconocido: {mode}")
    semantic = SemanticRetriever(encoder or SentenceTransformerEncoder())
    if mode == "semantic":
        return semantic.search(candidates, query, limit)
    bm25_hits = BM25Retriever().search(candidates, query, len(candidates))
    semantic_hits = semantic.search(candidates, query, len(candidates))
    return reciprocal_rank_fusion(bm25_hits, semantic_hits, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca incidencias logísticas históricas.")
    parser.add_argument("query", help="Pregunta o descripción del problema")
    parser.add_argument("--mode", choices=("bm25", "semantic", "hybrid"), default="bm25")
    parser.add_argument("--data", type=Path, default=DEMO_PATH)
    parser.add_argument("--limit", type=int, default=3)
    for field in ("tipo", "zona", "causa", "referencia"):
        parser.add_argument(f"--{field}", help=f"Filtra por {field} exacto")
    args = parser.parse_args()
    filters = {
        field: value for field in ("tipo", "zona", "causa", "referencia")
        if (value := getattr(args, field)) is not None
    }
    hits = search(load_incidents(args.data), args.query, args.mode, filters, args.limit)
    if not hits:
        print("No hay coincidencias.")
    for hit in hits:
        incident = hit.incident
        print(f"{incident.id}  score={hit.score:.4f}  tipo={incident.tipo}  zona={incident.zona}")
        print(f"   causa={incident.causa}  referencia={incident.referencia or '-'}")
        print(f"   {incident.texto}\n")


if __name__ == "__main__":
    main()
