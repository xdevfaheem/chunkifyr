import unittest
from chunkifyr import SyntacticChunker
from chunkifyr.base import Chunk

class TestSyntacticChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = SyntacticChunker(tokenizer_repo="bert-base-uncased")

    def test_init(self):
        self.assertIsNotNone(self.chunker.tokenizer)
        self.assertGreater(self.chunker.chunk_size, 0)

    def test_split_text(self):
        text = "This is a sentence. And this is another sentence."
        splitter, is_whitespace, splits = self.chunker._split_text(text)
        self.assertEqual(splitter, " ")
        self.assertTrue(is_whitespace)
        self.assertEqual(splits, ["This", "is", "a", "sentence.", "And", "this", "is", "another", "sentence."])

    def test_merge_splits(self):
        splits = ["This is a sentence", " And this is another sentence"]
        result_idx, result_chunk = self.chunker.merge_splits(splits, self.chunker.chunk_size, ".", self.chunker.token_counter)
        self.assertEqual(result_idx, 2)
        self.assertEqual(result_chunk, "This is a sentence. And this is another sentence")

    def test_chunk(self):
        text = "This is a long text that needs to be chunked properly. It contains multiple sentences and sections."
        chunks = self.chunker._chunk(text, self.chunker.chunk_size, self.chunker.token_counter)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], Chunk)

    def test_chunk_file(self):
        path = "data/test.txt"
        result = self.chunker.from_file(path)
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0], Chunk)

if __name__ == '__main__':
    unittest.main()
