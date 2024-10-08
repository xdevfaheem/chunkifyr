from abc import ABC, abstractmethod
import pymupdf
import docx
import requests
from bs4 import BeautifulSoup
from typing import List, Union
from pydantic import BaseModel, Field
from chunkifyr.util import install_package

class Chunk(BaseModel):
    text: str
    meta: dict = Field(default_factory=dict)

class Chunker(ABC):

    def __init__(self):
        
        try:
            import spacy
        except ImportError:
            print("spacy library is not installed. installing it now")
            install_package("spacy")
        
        model_name = "en_core_web_sm"
        if model_name not in spacy.util.get_installed_models():
            spacy.cli.download(model_name)
        self.sentencizer = spacy.load(model_name, exclude=["ner", "tagger"])

    def _extract_text(self, file_paths: Union[List[str], str]):
        """
        Extract text from given file filepath

        Args:
            file_paths: single or list of path to the file .

        Returns:
            text (str): extracted text from the files
        """
        text = ""
        
        file_paths = [file_paths] if type(file_paths)==str else file_paths
        for file_path in file_paths:
            # Handle PDF file
            if file_path.endswith(".pdf"):
                pdf = pymupdf.open(file_path)
                for page in pdf:
                    text += page.get_text()

            # Handle word file
            elif file_path.endswith(".docx"):
                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    text += para.text # + "\n"

            # Handle plain text files
            elif file_path.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text += file.read()

            # Handle webpage
            elif file_path.startswith("http://") or file_path.startswith("https://"):
                response = requests.get(file_path)
                soup = BeautifulSoup(response.content, 'html.parser')
                text += soup.get_text(separator='\n')

        return text.strip()
    
    def split_sentences(self, text):

        # can also use regular expressions to split the text into sentences based on punctuation followed by whitespace.
        nlp = self.sentencizer(text)
        sentences = [sent for sent in nlp.sents]
        return [s.text.strip() for s in sentences]
    
    def from_files(self, file_path):
        """
        Chunk the texts from single or multiple file

        Args:
            file_path: single or list of path to the file .

        Returns:
            List[Chunk]: A list of Chunk objects representing the chunks of text.
        """
        return self.chunk(self._extract_text(file_path))

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk the input text.

        Args:
            text (str): The input text to be chunked.

        Returns:
            List[Chunk]: A list of Chunk objects representing the chunks of text.
        """
        raise NotImplementedError