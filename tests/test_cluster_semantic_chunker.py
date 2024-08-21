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

    def test_small_text_chunking(self):
        """Test chunking on small text."""
        text = "This is a short sentence. Here is another one."
        chunks = self.chunker.chunk(text)
        
        # Assert that we get at least one chunk
        self.assertGreaterEqual(len(chunks), 1)
        
        # Check the content and metadata of the chunk
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].meta["start"], 0)
        self.assertEqual(chunks[0].meta["end"], 1)

    def test_large_text_chunking(self):
        """Test chunking on large text."""
        chunks = self.chunker.from_files("data/test.txt")
        
        # Assert that we get multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that each chunk respects the max chunk size
        for chunk in chunks:
            chunk_size = len(chunk.text.split())
            self.assertLessEqual(chunk_size, self.chunker._chunk_size)
    
    def test_empty_text(self):
        """Test behavior with empty text."""
        text = ""
        chunks = self.chunker.chunk(text)
        
        # Ensure no chunks are returned
        self.assertEqual(len(chunks), 0)


if __name__ == '__main__':
    unittest.main()
