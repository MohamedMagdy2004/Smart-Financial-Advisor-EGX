from db import Base
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.sql import func
import uuid
from sqlalchemy.dialects.postgresql import UUID

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)

    full_name = Column(String, nullable=True)

    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

