import unittest
from chunkifyr import SemanticChunker
from chunkifyr.base import Chunk
from langchain_core.embeddings.fake import FakeEmbeddings

class TestSemanticChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = SemanticChunker(embedder=FakeEmbeddings(size=100))

    def test_chunking(self):
        file_path = "data/test.txt"
        chunks = self.chunker.from_file(file_path)
        
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], Chunk)

if __name__ == "__main__":
    unittest.main()
