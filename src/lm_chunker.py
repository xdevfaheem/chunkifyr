from intellique.chunkers.src.chunkifyr.base import Chunker, Chunk
from intellique.lm_engine.base import LM
import spacy
import instructor
from instructor import Mode
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from typing import List
import time

class Chunk(BaseModel):
    start: int = Field(..., description="The starting artifact index of the chunk")
    end: int = Field(..., description="The ending artifact index of the chunk")
    context: str = Field(..., description="The context or topic of this chunk. Make this as thorough as possible, including information from the rest of the text so that the chunk makes good sense.")

class TextChunks(BaseModel):
    chunks: List[Chunk] = Field(..., description="List of chunks in the text")

class LMChunker(Chunker):

    def __init__(self, lm: LM):
        super().__init__()
        self.model = lm

    def chunk(self, text):

        sentences = self.split_sentences(text)

        # Insert artifacts after each sentence
        text_with_artifacts = ""
        for i, sentence in enumerate(sentences):
            text_with_artifacts += f"{sentence} [{i}]\n"

        # Determine chunk boundaries with constrained response format
        chunks: TextChunks = self.model.chat(
            messages=[
                {"role": "system", "content": "You are an AI assistant tasked with chunking a text into cohesive sections. Your goal is to create chunks that maintain topic coherence and context."},
                {"role": "user", "content": f"Here's a text with numbered artifacts. Determine the best chunks by specifying start and end artifact numbers. Make the chunks as large as possible while maintaining coherence. Provide a thorough context for each chunk, including information from the rest of the text to ensure the chunk makes good sense. Ensure no overlap between chunks, and also not gaps. the end of one chunk should be the start of the next. Here's the text:\n\n{text_with_artifacts}"}
            ],
            response_model=TextChunks,
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