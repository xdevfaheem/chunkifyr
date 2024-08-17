import unittest
from chunkifyr import SpacySemanticChunker, Chunk
import numpy as np
from langchain_core.embeddings.fake import FakeEmbeddings

class TestSpacySemanticChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = SpacySemanticChunker(embedder=FakeEmbeddings(size=100))

    def test_chunking(self):
        file_path = "data/test.txt"
        print()
        chunks = self.chunker.from_file(file_path)
        
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], Chunk)

if __name__ == "__main__":
    unittest.main()
