# cluster_semantic_chunker proposed on chroma's technical blog on evaluation strategy.
# A must read, if you want to learn more about various chunking algos and it's efectiveness: https://research.trychroma.com/evaluating-chunking
# this code is adapted from their codebase. https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/cluster_semantic_chunker.py

from typing import List
import numpy as np
from chunkifyr.base import Chunker, Chunk
from langchain_core.embeddings import Embeddings
from sentence_transformers import SimilarityFunction


class ClusterSemanticChunker(Chunker):
    def __init__(self, embedder: Embeddings, max_cluster=8):
        super().__init__()
        
        self.max_cluster = max_cluster
        self.embedder = embedder

        # cosine similarity function to be used to genearete similiarity matrix that captures the semantic similarity between all pairs of sentences in the text.
        self.similarity_fn = SimilarityFunction.to_similarity_fn("cosine")

    def _calculate_reward(self, matrix, start, end):
        """Computes a reward score for a potential chunk"""
        sub_matrix = matrix[start:end+1, start:end+1]
        return np.sum(sub_matrix)
    
    def segment_sentences(self, matrix, max_cluster_size, window_size=3):
        """Divides the text into optimal chunks based on the similarity matrix and reward scores."""

        n = matrix.shape[0]

        # Normalize the matrix by subtracting the mean of the upper triangle
        mean_value = np.mean(matrix[np.triu_indices(n, k=1)])
        matrix -= mean_value
        np.fill_diagonal(matrix, 0)

        # DP array to store maximum rewards
        dp = np.full(n, float('-inf'))
        # Segmentation points to reconstruct the solution
        segmentation = np.zeros(n, dtype=int)

        # Initialize the first element
        dp[0] = 0
        segmentation[0] = 0

        for i in range(1, n):
            for j in range(max(0, i - max_cluster_size), i):
                reward = self._calculate_reward(matrix, j, i)
                if j > 0:
                    reward += dp[j-1]  # Add the best reward up to the previous segment

                if reward > dp[i]:
                    dp[i] = reward
                    segmentation[i] = j

        # Reconstruct the clusters
        clusters = []
        i = n - 1
        while i > 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        # Add the first cluster if it starts from 0
        if i == 0:
            clusters.append((0, clusters[-1][0] - 1))

        clusters.reverse()
        return clusters

    def chunk(self, text: str) -> List[Chunk]:

        # split
        sentences = self.split_sentences(text)
        # embed
        embeddings = np.array(self.embedder.embed_documents(sentences))
        # calculate similarities
        similarity_matrix = np.dot(embeddings, embeddings.T)
        # determine optimal clusters and join
        clusters = self.segment_sentences(similarity_matrix, max_cluster_size=self.max_cluster)
        chunks = [Chunk(text=' '.join(sentences[start:end+1]), meta={"start": start, "end": end}) for start, end in clusters]
        return chunks
