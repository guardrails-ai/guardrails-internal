from typing import List, Optional

from guardrails.stores.document.base import Document, Page, PageCoordinates

try:
    import sqlalchemy
    import sqlalchemy.engine as Engine
    import sqlalchemy.orm as orm
except ImportError:
    sqlalchemy = None
    orm = None


if orm is not None:
    Base = orm.declarative_base()

    class SqlDocument(Base):
        __tablename__ = "documents"

        id: orm.Mapped[int] = orm.mapped_column(primary_key=True)
        page_num: orm.Mapped[int] = orm.mapped_column(
            sqlalchemy.Integer, primary_key=True
        )
        text: orm.Mapped[str] = orm.mapped_column(sqlalchemy.String)
        meta: orm.Mapped[dict] = orm.mapped_column(sqlalchemy.PickleType)
        vector_index: orm.Mapped[int] = orm.mapped_column(sqlalchemy.Integer)

else:

    class SqlDocument:
        pass


class SQLMetadataStore:
    _engine: Engine

    def __init__(self, path: Optional[str] = None):
        self._path = path
        self._conn = f"sqlite:///{path}" if path is not None else "sqlite://"
        self._engine = None

    def init_engine(self):
        if not self._engine:
            self._engine = sqlalchemy.create_engine(self._conn)
            SqlDocument.metadata.create_all(self._engine, checkfirst=True)

    def add_docs(self, docs: List[Document], vdb_last_index: int):
        self.init_engine()
        vector_id = vdb_last_index
        with orm.Session(self._engine) as session:
            for doc in docs:
                for page_num, text in doc.pages.items():
                    session.add(
                        SqlDocument(
                            id=doc.id,
                            page_num=page_num,
                            text=text,
                            meta=doc.metadata,
                            vector_index=vector_id,
                        )
                    )
                    vector_id += 1

            session.commit()

    def get_pages_for_for_indexes(self, indexes: List[int]) -> List[Page]:
        self.init_engine()
        pages: List[Page] = []
        with orm.Session(self._engine) as session:
            for index in indexes:
                query = sqlalchemy.select(SqlDocument).where(
                    SqlDocument.vector_index == index
                )
                sql_docs = session.execute(query)
                sql_doc = sql_docs.first()[0]
                pages.append(
                    Page(
                        PageCoordinates(sql_doc.id, sql_doc.page_num),
                        sql_doc.text,
                        sql_doc.meta,
                    )
                )

        return pages
