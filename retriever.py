"""Primer paso del proyecto: buscar invitados antes de añadir un agente."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DATASET_ID = "agents-course/unit3-invitees"
DEMO_PATH = Path(__file__).parent / "data" / "invitados_demo.json"


@dataclass(frozen=True)
class Guest:
    name: str
    relation: str
    description: str
    email: str

    @classmethod
    def from_record(cls, record: dict[str, str]) -> Guest:
        return cls(**{field: str(record[field]) for field in cls.__dataclass_fields__})

    def as_text(self) -> str:
        return (
            f"Nombre: {self.name}\n"
            f"Relación: {self.relation}\n"
            f"Descripción: {self.description}\n"
            f"Email: {self.email}"
        )


def load_demo_guests(path: Path = DEMO_PATH) -> list[Guest]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [Guest.from_record(record) for record in records]


def load_huggingface_guests() -> list[Guest]:
    """Descarga el conjunto del curso; requiere `pip install -r requirements.txt`."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Instala las dependencias: pip install -r requirements.txt") from exc
    return [Guest.from_record(record) for record in load_dataset(DATASET_ID, split="train")]


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"\w+", without_accents)


class GuestRetriever:
    """Índice BM25 pequeño, sin servicios externos ni claves de API."""

    def __init__(self, guests: list[Guest]):
        self.guests = guests
        self.terms = [Counter(tokenize(guest.as_text())) for guest in guests]
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.average_length = sum(self.lengths) / len(guests) if guests else 0
        self.document_frequency = Counter(
            term for terms in self.terms for term in terms
        )

    def search(self, query: str, limit: int = 3) -> list[tuple[Guest, float]]:
        if limit < 1 or not self.guests:
            return []

        query_terms = set(tokenize(query))
        scores: list[tuple[int, float]] = []
        for index, (guest, terms, length) in enumerate(
            zip(self.guests, self.terms, self.lengths)
        ):
            score = 0.0
            for term in query_terms:
                frequency = terms[term]
                if not frequency:
                    continue
                found_in = self.document_frequency[term]
                idf = math.log(1 + (len(self.guests) - found_in + 0.5) / (found_in + 0.5))
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * length / self.average_length
                )
                score += idf * frequency * 2.5 / denominator
            if score > 0:
                scores.append((index, score))

        scores.sort(key=lambda item: (-item[1], item[0]))
        return [(self.guests[index], score) for index, score in scores[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca invitados por nombre, relación o descripción.")
    parser.add_argument("query", help="Texto que quieres buscar")
    parser.add_argument("--source", choices=("demo", "huggingface"), default="demo")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    guests = load_demo_guests() if args.source == "demo" else load_huggingface_guests()
    results = GuestRetriever(guests).search(args.query, args.limit)
    if not results:
        print("No hay coincidencias.")
        return
    for guest, score in results:
        print(f"[{score:.3f}] {guest.as_text()}\n")


if __name__ == "__main__":
    main()
