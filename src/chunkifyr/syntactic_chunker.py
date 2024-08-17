import re
from bisect import bisect_left
from typing import Callable
from functools import cache
from itertools import accumulate
from contextlib import suppress
import transformers
from chunkifyr.base import Chunker, Chunk

# full credits goes to:https://github.com/umarbutler/semchunk/tree/main

_NON_WHITESPACE_SEMANTIC_SPLITTERS = (
    '.', '?', '!', '*', # Sentence terminators.
    ';', ',', '(', ')', '[', ']', "“", "”", '‘', '’', "'", '"', '`', # Clause separators.
    ':', '—', '…', # Sentence interrupters.
    '/', '\\', '–', '&', '-', # Word joiners.
)
"""A tuple of semantically meaningful non-whitespace splitters that may be used to chunk texts, ordered from most desirable to least desirable."""

_memoized_token_counters = {}
"""A map of token counters to their memoized versions."""

class SyntacticChunker(Chunker):

    def __init__(self, tokenizer_repo: str, max_token_chars=None, chunk_size = None) -> None:
        super().__init__()
        """
        Args:
            tokenizer_repo (str): HF tokenizer repo id of the tokenizer to be used.
            max_token_chars: Maximum token length
            chunk_size: Desired size of the chunks
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_repo)

        # set max_token_char
        if max_token_chars is None:
            if hasattr(self.tokenizer, "get_vocab") and callable(getattr(self.tokenizer, "get_vocab")):
                vocab = getattr(self.tokenizer, "get_vocab")()

                if hasattr(vocab, '__iter__') and vocab and all(hasattr(token, '__len__') for token in vocab):
                    max_token_chars = max(len(token) for token in vocab)

        # set chunk size
        if chunk_size is None:
            if hasattr(self.tokenizer, 'model_max_length') and isinstance(self.tokenizer.model_max_length, int):
                chunk_size = self.tokenizer.model_max_length

                # Attempt to reduce the chunk size by the number of special characters typically added by the tokenizer.
                with suppress(Exception):
                    chunk_size -= len(self.tokenizer.encode(''))

        if max_token_chars is not None:
            max_token_chars = max_token_chars - 1

            def token_counter(text: str) -> int:
                return len(self.tokenizer.encode(text, add_special_tokens = False))

            original_token_counter = token_counter

            def faster_token_counter(text: str) -> int:
                heuristic = chunk_size * 6

                if len(text) > heuristic and original_token_counter(text[:heuristic + max_token_chars]) > chunk_size:
                    return chunk_size + 1

                return original_token_counter(text)

            token_counter = faster_token_counter

        self.chunk_size = chunk_size
        self.token_counter = token_counter

    def _split_text(self, text: str) -> tuple[str, bool, list[str]]:
        """Split text using the most semantically meaningful splitter possible."""

        splitter_is_whitespace = True

        # Try splitting at, in order of most desirable to least desirable:
        # - The largest sequence of newlines and/or carriage returns;
        # - The largest sequence of tabs;
        # - The largest sequence of whitespace characters; and
        # - A semantically meaningful non-whitespace splitter.
        if '\n' in text or '\r' in text:
            splitter = max(re.findall(r'[\r\n]+', text))

        elif '\t' in text:
            splitter = max(re.findall(r'\t+', text))

        elif re.search(r'\s', text):
            splitter = max(re.findall(r'\s+', text))

        else:
            # Identify the most desirable semantically meaningful non-whitespace splitter present in the text.
            for splitter in _NON_WHITESPACE_SEMANTIC_SPLITTERS:
                if splitter in text:
                    splitter_is_whitespace = False
                    break

            # If no semantically meaningful splitter is present in the text, return an empty string as the splitter and the text as a list of characters.
            else: # NOTE This code block will only be executed if the for loop completes without breaking.
                return '', splitter_is_whitespace, list(text)

        # Return the splitter and the split text.
        return splitter, splitter_is_whitespace, text.split(splitter)

    def merge_splits(self, splits: list[str], chunk_size: int, splitter: str, token_counter: Callable) -> tuple[int, str]:
        """Merge splits until a chunk size is reached, returning the index of the last split included in the merged chunk along with the merged chunk itself."""

        average = 0.2
        low = 0
        high = len(splits) + 1
        cumulative_lengths = list(accumulate([len(split) for split in splits], initial=0))
        cumulative_lengths.append(cumulative_lengths[-1])

        while low < high:
            i = bisect_left(cumulative_lengths[low : high + 1], chunk_size * average)
            midpoint = min(i + low, high - 1)

            tokens = token_counter(splitter.join(splits[:midpoint]))

            average = cumulative_lengths[midpoint] / tokens if cumulative_lengths[midpoint] and tokens > 0 else average

            if tokens > chunk_size:
                high = midpoint
            else:
                low = midpoint + 1

        return low - 1, splitter.join(splits[:low - 1])

    def _chunk(
            self,
            text: str,
            chunk_size: int,
            token_counter: Callable[[str], int],
            memoize: bool = True,
            _recursion_depth: int = 0,
            _reattach_whitespace_splitters: bool = False,
        ) -> list[str]:
        """Split a text into semantically meaningful chunks of a specified size as determined by the provided token counter.

        Args:
            text (str): The text to be chunked.
            chunk_size (int): The maximum number of tokens a chunk may contain.
            token_counter (Callable[[str], int]): A callable that takes a string and returns the number of tokens in it.
            memoize (bool, optional): Whether to memoize the token counter. Defaults to `True`.

        Returns:
            list[str]: A list of chunks up to `chunk_size`-tokens-long, with any whitespace used to split the text removed."""

        # If this is not a recursive call and memoization is enabled, overwrite the `token_counter` with a memoized version of itself.
        if not _recursion_depth and memoize:
            token_counter = _memoized_token_counters.setdefault(token_counter, cache(token_counter))

        # Split the text using the most semantically meaningful splitter possible.
        splitter, splitter_is_whitespace, splits = self._split_text(text)
        if _reattach_whitespace_splitters: splitter_is_whitespace = False

        chunks = []
        skips = set()
        """A list of indices of splits to skip because they have already been added to a chunk."""

        # Iterate through the splits.
        for i, split in enumerate(splits):
            # Skip the split if it has already been added to a chunk.
            if i in skips:
                continue

            # If the split is over the chunk size, recursively chunk it.
            if token_counter(split) > chunk_size:
                chunks.extend(self._chunk(split, chunk_size, token_counter = token_counter, memoize = memoize, _recursion_depth = _recursion_depth + 1, _reattach_whitespace_splitters = _reattach_whitespace_splitters))

            # If the split is equal to or under the chunk size, add it and any subsequent splits to a new chunk until the chunk size is reached.
            else:
                # Merge the split with subsequent splits until the chunk size is reached.
                final_split_in_chunk_i, new_chunk = self.merge_splits(splits[i:], chunk_size, splitter, token_counter)

                # Mark any splits included in the new chunk for exclusion from future chunks.
                skips.update(range(i + 1, i + final_split_in_chunk_i))

                # Add the chunk.
                chunks.append(Chunk(text=new_chunk))

            # If the splitter is not whitespace and the split is not the last split, add the splitter to the end of the last chunk if doing so would not cause it to exceed the chunk size otherwise add the splitter as a new chunk.
            if not splitter_is_whitespace and not (i == len(splits) - 1 or all(j in skips for j in range(i + 1, len(splits)))):
                if token_counter(last_chunk_with_splitter := chunks[-1] + splitter) <= chunk_size:
                    chunks[-1] = last_chunk_with_splitter
                else:
                    chunks.append(Chunk(text=splitter))

        # If this is not a recursive call, remove any empty chunks.
        if not _recursion_depth:
            chunks = [chunk for chunk in chunks if chunk.text.strip()]
        return chunks

    def chunk(self, texts) -> list[Chunk]:

        if isinstance(texts, str):
            return self._chunk(texts, self.chunk_size, self.token_counter, memoize = True)
        else:
            raise ValueError("Expected `str` but got input of type {}".format(type(texts)))