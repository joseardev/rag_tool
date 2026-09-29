import unittest

from retriever import load_incidents, search


class FakeEncoder:
    """Vectores controlados para verificar la lógica sin descargar un modelo."""

    def encode(self, texts):
        vectors = [[1.0, 0.0]]
        for text in texts[1:]:
            if "18 minutos" in text:
                vectors.append([0.99, 0.01])  # D: parecido aunque no comparta palabras
            elif "contenedor KLT" in text:
                vectors.append([0.9, 0.1])
            elif "milk-run" in text:
                vectors.append([0.8, 0.2])
            else:
                vectors.append([0.1, 0.9])
        return vectors


class IncidentSearchTests(unittest.TestCase):
    def setUp(self):
        self.incidents = load_incidents()

    def test_bm25_finds_line_supply(self):
        ids = [hit.incident.id for hit in search(self.incidents, "material línea", limit=5)]
        self.assertIn("B", ids[:2])
        self.assertNotIn("D", ids)

    def test_metadata_filters_before_ranking(self):
        ids = [
            hit.incident.id for hit in search(
                self.incidents, "retraso entrega", filters={"tipo": "transporte"}
            )
        ]
        self.assertEqual(set(ids), {"A", "D"})

    def test_reference_filter_is_exact(self):
        ids = [
            hit.incident.id for hit in search(
                self.incidents, "línea", filters={"referencia": "8V0123456"}
            )
        ]
        self.assertEqual(set(ids), {"A", "C"})

    def test_semantic_can_find_d_without_shared_terms(self):
        hits = search(self.incidents, "material llega tarde", "semantic", encoder=FakeEncoder())
        self.assertEqual(hits[0].incident.id, "D")

    def test_hybrid_and_filter(self):
        hits = search(
            self.incidents, "retraso entrega", "hybrid",
            filters={"tipo": "transporte"}, encoder=FakeEncoder()
        )
        self.assertEqual({hit.incident.id for hit in hits}, {"A", "D"})


if __name__ == "__main__":
    unittest.main()
