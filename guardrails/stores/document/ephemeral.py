import hashlib
from typing import Any, Dict, List, Optional

from guardrails.embedding import EmbeddingBase
from guardrails.stores.document.base import Document, DocumentStoreBase, Page
from guardrails.stores.document.sql import SQLMetadataStore
from guardrails.vectordb import VectorDBBase

try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None
    orm = None


class EphemeralDocumentStore(DocumentStoreBase):
    """EphemeralDocumentStore is a document store that stores the documents on
    local disk and use a ephemeral vector store like Faiss."""

    def __init__(
        self,
        vector_db: Optional["VectorDBBase"] = None,
        path: Optional[str] = None,
        embedding_model: Optional["EmbeddingBase"] = None,
    ):
        """Creates a new EphemeralDocumentStore.

        Args:
            vector_db: VectorDBBase instance to use for storing the vectors.
            path: Path to the database file store metadata.
        """
        if sqlalchemy is None:
            raise ImportError(
                "SQLAlchemy is required for EphemeralDocumentStore"
                "Please install it using `pip install SqlAlchemy`"
            )
        if vector_db is None:
            from guardrails.vectordb import Faiss

            if embedding_model is None:
                from guardrails.embedding import OpenAIEmbedding

                embedding_model = OpenAIEmbedding()

            vector_db = Faiss.new_flat_ip_index(
                embedding_model.output_dim, embedder=embedding_model
            )
        self._vector_db = vector_db
        self._storage = SQLMetadataStore(path=path)

    def add_document(self, document: Document):
        # Add the document, in case the document is already there it
        # would raise an exception and we assume the document and
        # vectors are present.
        try:
            self._storage.add_docs(
                [document], vdb_last_index=self._vector_db.last_index()
            )
        except sqlalchemy.exc.IntegrityError:
            return
        self._vector_db.add_texts(list(document.pages.values()))

    def add_text(self, text: str, meta: Dict[Any, Any]) -> str:
        hash = hashlib.md5()
        hash.update(text.encode("utf-8"))
        hash.update(str(meta).encode("utf-8"))
        id = hash.hexdigest()

        doc = Document(id, {0: text}, meta)
        self.add_document(doc)
        return doc.id

    def add_texts(self, texts: Dict[str, Dict[Any, Any]]) -> List[str]:
        doc_ids = []
        for text, meta in texts.items():
            doc_id = self.add_text(text, meta)
            doc_ids.append(doc_id)
        return doc_ids

    def search(self, query: str, k: int = 4) -> List[Page]:
        vector_db_indexes = self._vector_db.similarity_search(query, k)
        filtered_ids = filter(lambda x: x != -1, vector_db_indexes)
        return self._storage.get_pages_for_for_indexes(filtered_ids)

    def search_with_threshold(
        self, query: str, threshold: float, k: int = 4
    ) -> List[Page]:
        vector_db_indexes = self._vector_db.similarity_search_with_threshold(
            query, k, threshold
        )
        filtered_ids = filter(lambda x: x != -1, vector_db_indexes)
        return self._storage.get_pages_for_for_indexes(filtered_ids)

    def flush(self, path: Optional[str] = None):
        self._vector_db.save(path)
