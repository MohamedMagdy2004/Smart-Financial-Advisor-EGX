from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker , declarative_base

SQLALCHEMY_DATABASE_URL = "postgresql://root:root@localhost:5432/EGX_database"

engine = create_engine(SQLALCHEMY_DATABASE_URL)


sessionlocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = sessionlocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)
