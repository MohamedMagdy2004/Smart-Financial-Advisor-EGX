from models import User, Message, Watchlist, PortfolioHolding
from sqlalchemy.orm import Session
from schemas import UserCreate, MessageCreate, WatchlistCreate, PortfolioHoldingCreate, PortfolioHoldingUpdate, UserUpdate
import bcrypt
import uuid

# دالة مساعدة لضمان تحويل النص القادم من الواجهة إلى UUID نقي لقاعدة البيانات
def _ensure_uuid(uid) -> uuid.UUID:
    if isinstance(uid, str):
        return uuid.UUID(uid)
    return uid

# ==================== Watchlist Services ====================

def get_watchlist(db: Session, user_id):
    clean_uid = _ensure_uuid(user_id)
    return db.query(Watchlist).filter(Watchlist.user_id == clean_uid).all()

def add_watchlist_item(db: Session, data: WatchlistCreate):
    clean_uid = _ensure_uuid(data.user_id)
    
    # Check if already exists to prevent duplicates
    existing = db.query(Watchlist).filter(
        Watchlist.user_id == clean_uid, 
        Watchlist.ticker == data.ticker
    ).first()
    if existing:
        return existing
    
    # تحويل البيانات واستبدال المعرف بالنقي
    dumped_data = data.model_dump()
    dumped_data['user_id'] = clean_uid
    
    item = Watchlist(**dumped_data)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

def remove_watchlist_item(db: Session, item_id):
    clean_item_id = _ensure_uuid(item_id)
    item = db.query(Watchlist).filter(Watchlist.id == clean_item_id).first()
    if item:
        db.delete(item)
        db.commit()
    return item

# ==================== Portfolio Services ====================

def get_portfolio(db: Session, user_id):
    clean_uid = _ensure_uuid(user_id)
    return db.query(PortfolioHolding).filter(PortfolioHolding.user_id == clean_uid).all()

def add_portfolio_holding(db: Session, data: PortfolioHoldingCreate):
    clean_uid = _ensure_uuid(data.user_id)
    
    dumped_data = data.model_dump()
    dumped_data['user_id'] = clean_uid
    
    holding = PortfolioHolding(**dumped_data)
    db.add(holding)
    db.commit()
    db.refresh(holding)
    return holding

def update_portfolio_holding(db: Session, holding_id, data: PortfolioHoldingUpdate):
    clean_holding_id = _ensure_uuid(holding_id)
    holding = db.query(PortfolioHolding).filter(PortfolioHolding.id == clean_holding_id).first()
    if not holding:
        return None
    
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(holding, key, value)
    
    db.commit()
    db.refresh(holding)
    return holding

def delete_portfolio_holding(db: Session, holding_id):
    clean_holding_id = _ensure_uuid(holding_id)
    holding = db.query(PortfolioHolding).filter(PortfolioHolding.id == clean_holding_id).first()
    if holding:
        db.delete(holding)
        db.commit()
    return holding

# ==================== User Services ====================

def create_user(db: Session, data: UserCreate):
    # تم إصلاح تداخل الدالة هنا وإعادتها لمكانها الطبيعي
    hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt())
    
    user_data = data.model_dump()
    user_data['password'] = hashed_password.decode('utf-8')

    user_instance = User(**user_data)
    db.add(user_instance)
    db.commit()
    db.refresh(user_instance)
    return user_instance

def get_users(db: Session):
    return db.query(User).all()

def get_user(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_user_by_full_name(db: Session, full_name: str):
    return db.query(User).filter(User.full_name == full_name).first()

def authenticate_user(db: Session, email: str, password: str):
    user = get_user(db, email)
    if not user:
        return None
    if not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return None
    return user

def authenticate_user_by_full_name(db: Session, full_name: str, password: str):
    user = get_user_by_full_name(db, full_name)
    if not user:
        return None
    if not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return None
    return user

def update_user(db: Session, user_id, data: UserUpdate):
    clean_uid = _ensure_uuid(user_id)
    user_instance = db.query(User).filter(User.id == clean_uid).first()
    if not user_instance:
        return None
    
    dumped_data = data.model_dump(exclude_unset=True)
    if 'password' in dumped_data and dumped_data['password']:
        hashed_password = bcrypt.hashpw(dumped_data['password'].encode('utf-8'), bcrypt.gensalt())
        dumped_data['password'] = hashed_password.decode('utf-8')
    
    for key, value in dumped_data.items():
        setattr(user_instance, key, value)
    
    db.commit()
    db.refresh(user_instance)
    return user_instance

def delete_user(db: Session, user_id):
    clean_uid = _ensure_uuid(user_id)
    user_instance = db.query(User).filter(User.id == clean_uid).first()
    if not user_instance:
        return None
    db.delete(user_instance)
    db.commit()
    return user_instance

# ==================== Message Services ====================

def create_message(db: Session, data: MessageCreate):
    clean_uid = _ensure_uuid(data.user_id)
    dumped_data = data.model_dump()
    dumped_data['user_id'] = clean_uid
    
    message_instance = Message(**dumped_data)
    db.add(message_instance)
    db.commit()
    db.refresh(message_instance)
    return message_instance

def get_messages(db: Session):
    return db.query(Message).all()

def get_message(db: Session, user_id):
    clean_uid = _ensure_uuid(user_id)
    return db.query(Message).filter(Message.user_id == clean_uid).all()

def delete_message(db: Session, message_id):
    clean_msg_id = _ensure_uuid(message_id)
    message_instance = db.query(Message).filter(Message.id == clean_msg_id).first()
    if not message_instance:
        return None
    db.delete(message_instance)
    db.commit()
    return message_instance

def delete_messages_by_user(db: Session, user_id):
    clean_uid = _ensure_uuid(user_id)
    messages = db.query(Message).filter(Message.user_id == clean_uid).all()
    for message in messages:
        db.delete(message)
    db.commit()
    return messages
