import unittest
from chunkifyr import LMChunker, Chunk

class TestLMChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = LMChunker(model="anything", openai_api="nothing", openai_base_url="http://localhost:1234/v1") # from local openai server from llamacpp

    def test_chunking(self):
        file_path = "data/test.txt"
        chunks = self.chunker.from_file(file_path)
        
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], Chunk)
        self.assertIsInstance(chunks[0].meta["description"], str)

if __name__ == "__main__":
    unittest.main()
