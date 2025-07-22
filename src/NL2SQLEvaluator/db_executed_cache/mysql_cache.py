import hashlib
import json
import logging
import pickle
from typing import Iterable, Optional

from sqlalchemy import Column, String, func, insert, select, create_engine
from sqlalchemy.dialects.mysql import LONGTEXT, LONGBLOB
from sqlalchemy.orm import DeclarativeBase, Session

from sqlalchemy.types import TypeDecorator


from NL2SQLEvaluator.logger import get_logger


def hash_db_id_sql(uri, query) -> str:
    value = f"{uri}|{query}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CompressedJSON(TypeDecorator):
    impl = LONGBLOB
    cache_ok = True

    def bind_expression(self, bindvalue):
        """Wraps INSERT/UPDATE values"""
        # Convert Python object → JSON bytes → COMPRESS() when inserting/updating
        return func.compress(func.cast(bindvalue, LONGBLOB))

    def column_expression(self, col):
        """Wraps SELECT columns"""
        # Automatically UNCOMPRESS in SELECTs
        return func.uncompress(col)


class Base(DeclarativeBase):
    pass


class CachedData(Base):
    __tablename__ = "cache_data"

    hash_key = Column(
        String(64), primary_key=True, index=True
    )  # 64 hex chars from SHA‑256

    uri = Column(String(1000), nullable=False)
    query = Column(LONGTEXT, nullable=False)
    result = Column(CompressedJSON, nullable=True)

    def __repr__(self):
        return f"CachedData(hash_key={self.hash_key!r}, uri={self.uri!r}, query={self.query!r}, result={self.result!r})"


class MySQLCache:
    def __init__(
        self,
        engine,
        logger: Optional[logging.Logger] = None,
        timeout: Optional[int | float] = 400,
        *args,
        **kwargs,
    ):
        self.timeout = timeout
        self.engine = engine
        Base.metadata.create_all(engine)
        self.logger = logger or get_logger(name=__name__, level="INFO")
        self.engine_url = str(engine.url)

    @classmethod
    def from_uri(cls, *args, **kwargs) -> "MySQLCache":
        # https://docs.sqlalchemy.org/en/20/core/engines.html
        # Remove Decimal and NewDecimal types from pymysql converters
        # this is necessry for compression otherwise it is
        # conv = converters.conversions.copy()  # start from the defaults
        # conv[FIELD_TYPE.DECIMAL] = float  # DECIMAL
        # conv[FIELD_TYPE.NEWDECIMAL] = float  # DECIMAL in newer MySQLs
        user = kwargs.get("user", "root")
        password = kwargs.get("password", "")
        port = kwargs.get("port", 3306)
        db_name = kwargs.get("db_name", "cache")
        host = kwargs.get("host", "localhost")

        return cls(
            create_engine(
                f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}",
                connect_args={
                    # "conv": conv,
                    "connect_timeout": 120,
                    "read_timeout": 120,
                    "write_timeout": 120,
                    "charset": "utf8mb4",
                    "autocommit": True,
                },
                echo=True,
            )
        )

    def get_from_cache(self, uri, query: str) -> list[list] | None:
        """Retrieve the result of a query from the cache."""
        hash_id = hash_db_id_sql(uri, query)
        self.logger.debug(f"Fetching from cache with hash_id: {hash_id}")
        with Session(self.engine) as session:
            stmt = select(CachedData.result).where(CachedData.hash_key == hash_id)
            result = session.execute(stmt).scalars().first()
        return pickle.loads(result) if result else None

    def insert_in_cache(self, uri, query: str, result: list) -> None:
        """Insert the result of a query into the cache."""
        hash_id = hash_db_id_sql(uri, query)
        self.logger.debug(
            f"Inserting (or Ignore if duplicate) in cache with hash_id: {hash_id}"
        )

        pickled_result = pickle.dumps(result)
        hash_id = hash_db_id_sql(uri, query)
        stmt = (
            insert(CachedData)
            .prefix_with("IGNORE")
            .values(hash_key=hash_id, uri=uri, query=query, result=pickled_result)
        )
        with Session(self.engine) as session:
            session.execute(stmt)
            session.commit()


if __name__ == "__main__":
    # Example usage
    cache = MySQLCache.from_uri(port=3307)
    uri = "test_db_uri"
    query = "SELECT * FROM yolewjwde"
    # Insert data into cache
    cache.insert_in_cache(
        uri, query, [("row1_col1", "row1_col2"), ("row2_col1", "row2_col2")]
    )

    # Retrieve data from cache
    cached_result = cache.get_from_cache(uri, query)
    print(cached_result)
    assert cached_result == [("row1_col1", "row1_col2"), ("row2_col1", "row2_col2")]
