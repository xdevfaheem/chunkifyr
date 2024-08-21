import unittest
import numpy as np
from chunkifyr.base import Chunk
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from chunkifyr import ClusterSemanticChunker


class TestClusterSemanticChunker(unittest.TestCase):
    def setUp(self):
        """Set up the necessary components for testing."""
        self.embedder = FakeEmbeddings()
        self.chunker = ClusterSemanticChunker(self.embedder)

    def test_chunking(self):
        chunks = self.chunker.from_files("data/test.txt")
        
        # Assert that we get at least one chunk
        self.assertGreaterEqual(len(chunks), 2)
    
    def test_empty_text(self):
        """Test behavior with empty text."""
        text = ""
        chunks = self.chunker.chunk(text)
        
        # Ensure no chunks are returned
        self.assertEqual(len(chunks), 0)

if __name__ == '__main__':
    unittest.main()
