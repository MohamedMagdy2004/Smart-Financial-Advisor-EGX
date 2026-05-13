from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List
from uuid import UUID

# ==================== User Schemas ====================

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None

class User(UserBase):
    id: UUID
    is_active: bool
    created_at: datetime = datetime.now()

    class Config:
        from_attributes = True

class UserList(BaseModel):
    count: int
    users: List[User]

# ==================== Message Schemas ====================

class MessageBase(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class MessageCreate(MessageBase):
    user_id: UUID
    llm_output: Optional[dict] = None

class Message(MessageBase):
    id: UUID
    user_id: UUID
    llm_output: Optional[dict] = None
    created_at: datetime = datetime.now()

    class Config:
        from_attributes = True

class MessageList(BaseModel):
    count: int
    messages: List[Message]

# ==================== Portfolio & Watchlist Schemas ====================

class WatchlistBase(BaseModel):
    ticker: str

class WatchlistCreate(WatchlistBase):
    user_id: UUID

class WatchlistResponse(WatchlistBase):
    id: UUID
    user_id: UUID
    last_analysis: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PortfolioHoldingBase(BaseModel):
    ticker: str
    quantity: float
    avg_buy_price: float

class PortfolioHoldingCreate(PortfolioHoldingBase):
    user_id: UUID

class PortfolioHoldingUpdate(BaseModel):
    quantity: Optional[float] = None
    avg_buy_price: Optional[float] = None

class PortfolioHoldingResponse(PortfolioHoldingBase):
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PortfolioSummary(BaseModel):
    holdings: List[PortfolioHoldingResponse]
    total_invested: float

# ==================== Auth Schemas ====================

# class LoginRequest(BaseModel):
#     email: EmailStr
#     password: str

# class LoginResponse(BaseModel):
#     access_token: str
#     token_type: str
#     user: User