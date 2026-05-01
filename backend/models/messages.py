from db import Base
from sqlalchemy import Column, String, ForeignKey, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    role = Column(String, nullable=False)  
    # "user" or "assistant"

    content = Column(String, nullable=False)

    llm_output = Column(JSON, nullable=True)  
    # your Qwen structured JSON

    created_at = Column(DateTime(timezone=True), server_default=func.now())