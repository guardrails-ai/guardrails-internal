from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Document:
    """Document holds text and metadata of a document.

    Examples of documents are PDFs, Word documents, etc. A collection of
    related text in an NLP application can be thought of a document as
    well.
    """

    id: str
    pages: Dict[int, str]
    metadata: Dict[Any, Any] = None


# PageCoordinates is a datastructure that points to the location
# of a page in a document.
PageCoordinates = namedtuple("PageCoordinates", ["doc_id", "page_num"])


@dataclass
class Page:
    """Page holds text and metadata of a page in a document.

    It also containts the coordinates of the page in the document.
    """

    cordinates: PageCoordinates
    text: str
    metadata: Dict[Any, Any]


class DocumentStoreBase(ABC):
    """Abstract class for a store that can store text, and metadata from
    documents.

    The store can be queried by text for similar documents.
    """

    @abstractmethod
    def add_document(self, document: Document):
        """Adds a document to the store.

        Args:
            document: Document object to be added

        Returns:
            None if the document was added successfully
        """
        ...

    @abstractmethod
    def search(self, query: str, k: int = 4) -> List[Page]:
        """Searches for pages which contain the text similar to the query.

        Args:
            query: Text to search for.
            k: Number of similar pages to return.

        Returns:
            List[Pages] List of pages which contains similar texts
        """
        ...

    @abstractmethod
    def add_text(self, text: str, meta: Dict[Any, Any]) -> str:
        """Adds a text to the store.
        Args:
            text: Text to add.
            meta: Metadata to associate with the text.

        Returns:
            The id of the text.
        """
        ...

    @abstractmethod
    def add_texts(self, texts: Dict[str, Dict[Any, Any]]) -> List[str]:
        """Adds a list of texts to the store.
        Args:
            texts: List of texts to add, and their associalted metadata.
            example: [{"I am feeling good", {"sentiment": "postive"}}]

        Returns:
            List of ids of the texts."""
        ...

    @abstractmethod
    def flush():
        """Flushes the store to disk."""
        ...
