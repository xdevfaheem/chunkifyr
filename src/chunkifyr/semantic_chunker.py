from chunkifyr.base import Chunker, Chunk
from langchain_core.embeddings import Embeddings
from sentence_transformers import SimilarityFunction
from tqdm.auto import tqdm
import numpy as np

class SemanticChunker(Chunker): # can be called as Semantic Chunker

    def __init__(self, embedder: Embeddings, similarity_threshold: float = 0.75):
        super().__init__()
        
        self.embedder = embedder
        self.similarity_fn = SimilarityFunction.to_similarity_fn("cosine")
        self.similarity_threshold = similarity_threshold*100

    def get_embeddings(self, texts):

        # generate embeddings for a list of texts using a pre-trained model and handle any exceptions.
        embeddings = self.embedder.embed_query(texts)
        return embeddings

    def _combine_sentences(self, sentences):
        # Create a buffer by combining each sentence with its previous and next sentence to provide a wider context.
        combined_sentences = []
        for i in range(len(sentences)):
            combined_sentence = sentences[i]
            if i > 0:
                combined_sentence = sentences[i-1] + ' ' + combined_sentence
            if i < len(sentences) - 1:
                combined_sentence += ' ' + sentences[i+1]
            combined_sentences.append(combined_sentence)
        return combined_sentences

    def _calculate_cosine_distances(self, embeddings):
        # Calculate the cosine distance (1 - cosine similarity) between consecutive embeddings.
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = self.similarity_fn([embeddings[i]], [embeddings[i + 1]])
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def _pipeline(self, doc_text):

        # split into sentences
        single_sentences_list = self.split_sentences(doc_text)
        self.sents = single_sentences_list

        # combine the current sentence with previuos and next sentences
        combined_sentences = self._combine_sentences(single_sentences_list)

        # calculate embeddings and cosine distances
        embeddings = self.get_embeddings(combined_sentences)
        distances = self._calculate_cosine_distances(embeddings)

        return distances

    def chunk(self, text):

        distances = self._pipeline(text)
        breakpoint_percentile_threshold = self.similarity_threshold
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        indices_above_thresh = [i for i, distance in enumerate(distances) if distance > breakpoint_distance_threshold]
        chunks = []
        start_index = 0
        for index in tqdm(indices_above_thresh):
            chunk = ' '.join(self.sents[start_index:index+1])

            # if chunk length is higher than 3k than chunk it down, chunk it don further
            if len(chunk) > 3000:
                print("!!!")
                sub_distances = self._pipeline(chunk)
                sub_ts = self.similarity_threshold
                breakingpoint_ts = np.percentile(sub_distances, sub_ts)
                ts_indices = [i for i, sub_distance in enumerate(sub_distances) if sub_distance > breakingpoint_ts]

                si = 0
                for i in ts_indices:
                    sub_chunk = ' '.join(self.sents[si:i+1])
                    chunks.append(Chunk(text=sub_chunk))
                    si = i+1

                if si < len(self.sents):
                    sub_chunk = ' '.join(self.sents[si:])
                    chunks.append(Chunk(text=sub_chunk))

            else:
                chunks.append(Chunk(text=chunk))
                start_index = index + 1

        # If there are any sentences left after the last breakpoint, add them as the final chunk.
        if start_index < len(self.sents):
            chunk = ' '.join(self.sents[start_index:])
            chunks.append(Chunk(text=chunk))

        # Return the list of text chunks.
        return [chunk for chunk in chunks if chunk.text.strip()] # filter out empty whitespace para