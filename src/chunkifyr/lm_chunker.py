from chunkifyr.base import Chunker, Chunk
import instructor
from pydantic import BaseModel
from openai import OpenAI as OpenAIClient
from instructor import Mode
from pydantic import BaseModel, Field
from typing import List

class Chunk(BaseModel):
    start: int = Field(..., description="The starting artifact index of the chunk")
    end: int = Field(..., description="The ending artifact index of the chunk")
    context: str = Field(..., description="The context or topic of this chunk. Make this as thorough as possible, including information from the rest of the text so that the chunk makes good sense.")

class TextChunks(BaseModel):
    chunks: List[Chunk] = Field(..., description="List of chunks in the text")

class LMChunker(Chunker):

    def __init__(self, model: str, openai_client: OpenAIClient): #model: str, openai_api: str, openai_base_url: str):
        super().__init__()
        """
        model (str): name of the OAI model, or anything if your using local OAI server (llama_cpp, llamafile, ollama)
        openai_client (str): OpenAI class object, can be from proprietary OAI api key & base_url or OS OpenAI class created from local OAI server creds.
        """

        self.model = model
        self.instructor_client = instructor.client.from_openai(openai_client, mode=Mode.JSON_SCHEMA)

    def chunk(self, text):

        sentences = self.split_sentences(text)

        # Insert artifacts after each sentence
        text_with_artifacts = ""
        for i, sentence in enumerate(sentences):
            text_with_artifacts += f"{sentence} [{i}]\n"

        # Determine chunk boundaries with constrained response format
        chunks: TextChunks = self.instructor_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI assistant tasked with chunking a text into cohesive sections. Your goal is to create chunks that maintain topic coherence and context."},
                {"role": "user", "content": f"Here's a text with numbered artifacts. Determine the best chunks by specifying start and end artifact numbers. Make the chunks as large as possible while maintaining coherence. Provide a thorough context for each chunk, including information from the rest of the text to ensure the chunk makes good sense. Ensure no overlap between chunks, and also not gaps. the end of one chunk should be the start of the next. Here's the text:\n\n{text_with_artifacts}"}
            ],
            response_model=TextChunks,
            max_retries=4
        )

        chunks = []
        for chunk in chunks.chunks:
            chunk_text = " ".join(sentences[chunk.start:chunk.end+1])
            chunk = Chunk(
                text=chunk_text, # add the chunk ctx as meta desc
                meta={"description": chunk.context}
            )
            chunks.append(chunk)
        return chunks
