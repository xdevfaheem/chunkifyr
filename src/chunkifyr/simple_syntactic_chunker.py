from chunkifyr.base import Chunker, Chunk
from typing import List

class SimpleSyntacticChunker(Chunker):

    def __init__(self, sep="\n\n", max_len=1_000_000, chunk_size=2000, chunk_overlap_percentage=0.050):
        super().__init__()

        self.sentencizer.max_length = int(max_len)
        self._sep = sep
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_size * chunk_overlap_percentage

    def _join_docs(self, docs, separator):
        return separator.join(docs).strip()

    def _merge_splits(self, splits: str, separator: str):
        # adapted from: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/base.py#L107
        separator_len = len(separator)

        docs = []
        current_doc = []
        total = 0
        for d in splits:
            _len = len(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    print.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(Chunk(text=doc))
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= len(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            doc = Chunk(text=doc)
            docs.append(doc)
        return docs

    def chunk(self, text) -> List[Chunk]:
        splits = self.split_sentences(text)
        return self._merge_splits(splits, self._sep)