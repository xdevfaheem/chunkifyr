import unittest
from chunkifyr import LMChunker
from chunkifyr.base import Chunk
from openai import OpenAI

class TestLMChunker(unittest.TestCase):

    def setUp(self):
        client = OpenAI(api_key="empty", base_url="http://localhost:1234/v1") # from local openai server from llamacpp
        self.chunker = LMChunker(model="anything", openai_client=client)

    def test_chunking(self):
        file_path = "data/test.txt"
        chunks = self.chunker.from_file(file_path)
        
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], Chunk)
        self.assertIsInstance(chunks[0].meta["description"], str)

if __name__ == "__main__":
    unittest.main()
