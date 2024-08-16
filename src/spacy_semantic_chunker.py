import torch
import spacy
import numpy as np
from sentence_transformers import SimilarityFunction
from chunkifyr import Chunker, Chunk

class SpacySemanticChunker(Chunker):

    def __init__(self, embedder, similarity_threshold=0.80, group_max_sentences=5):
        super().__init__()

        self.embedder = embedder
        self.similarity_threshold = float(similarity_threshold)
        self.similarity_fn = embedder.get_similarity
        self.group_max_sentences = group_max_sentences

    def _calculate_cosine_distances(self, embeddings):
        # Calculate the cosine distance (1 - cosine similarity) between consecutive embeddings.
        distances = []
        for i in range(1, len(embeddings)):
            similarity = self.similarity_fn([embeddings[i]], [embeddings[i - 1]])
            # distance = 1 - similarity
            distances.append(similarity)
        return distances

    def chunk(self, text):

        sentences = self.split_sentences(text)
        if len(sentences) == 0:
            return []
        # generate embeddings and calculate cosine distances
        embeddings = self.embedder(sentences, batch_size=8, progress_bar=True, return_type="list")
        distances = self._calculate_cosine_distances(embeddings)

        # The first sentence is always in the first group.
        groups = [[sentences[0]]]
        for i in range(1, len(sentences)):

            if len(groups[-1]) >= self.group_max_sentences:
                groups.append([sentences[i]])
            elif distances[i - 1] >= self.similarity_threshold:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        return [Chunk(text=" ".join(g)) for g in groups]
