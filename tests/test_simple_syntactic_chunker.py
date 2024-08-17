import unittest
from chunkifyr import SimpleSyntacticChunker
from chunkifyr.base import Chunk

class TestSpacySyntacticChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = SimpleSyntacticChunker(chunk_size=2048)

    def test_init(self):
        self.assertEqual(self.chunker._chunk_size, 2048)
        self.assertEqual(self.chunker._chunk_overlap, 102.4)
        self.assertEqual(self.chunker._sep, "\n\n")

    def test_join_docs(self):
        docs = ["This is a sentence.", "This is another sentence."]
        result = self.chunker._join_docs(docs, " ")
        expected = "This is a sentence. This is another sentence."
        self.assertEqual(result, expected)

    def test_merge_splits(self):
        splits = ["This is a long sentence that needs", "to be split and merged correctly."]
        result = self.chunker._merge_splits(splits, " ")
        expected = [Chunk(text="This is a long sentence that needs to be split and merged correctly.")]
        self.assertEqual([chunk.text for chunk in result], [chunk.text for chunk in expected])

    def test_chunk(self):
        path = "data/test.txt"
        result = self.chunker.from_file(path)
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0], Chunk)

if __name__ == '__main__':
    unittest.main()