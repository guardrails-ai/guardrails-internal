from typing import Any, Dict, List

from guardrails.document_store import Document, DocumentStoreBase, Page


class MockDocumentStore(DocumentStoreBase):
    def add_document(self, document: Document):
        pass

    def search(self, query: str, k: int = 4) -> List[Page]:
        return []

    def add_text(self, text: str, meta: Dict[Any, Any]) -> str:
        return text

    def add_texts(self, texts: Dict[str, Dict[Any, Any]]) -> List[str]:
        return texts

    def flush():
        pass
