import unittest

from retriever import GuestRetriever, load_demo_guests


class GuestRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.retriever = GuestRetriever(load_demo_guests())

    def test_finds_name_without_requiring_accent(self):
        self.assertEqual(self.retriever.search("Marta Rios")[0][0].name, "Marta Ríos")

    def test_finds_description(self):
        self.assertEqual(self.retriever.search("platos vegetarianos")[0][0].name, "Luis Moreno")

    def test_unknown_term_has_no_result(self):
        self.assertEqual(self.retriever.search("ornitorrinco"), [])


if __name__ == "__main__":
    unittest.main()
