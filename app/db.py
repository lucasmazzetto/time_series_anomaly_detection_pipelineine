from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.utils.env import get_database_url

DATABASE_URL = get_database_url()

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_session():
    """@brief Yield a database session and ensure it is closed.

    @description Creates a SQLAlchemy session for the request lifecycle and
    guarantees cleanup after use.

    @return Generator that yields an active SQLAlchemy session.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
