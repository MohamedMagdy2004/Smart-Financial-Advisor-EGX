from models import User, Message
from sqlalchemy.orm import Session
from schemas import UserCreate, MessageCreate
import bcrypt
import uuid

def create_user(db: Session, data: UserCreate):
    # Hash the password
    hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt())
    
    # Prepare user data with hashed password
    user_data = data.model_dump()
    user_data['password'] = hashed_password.decode('utf-8')  # Store hash in 'password' field

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

    # Compare hashed password
    if not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return None

    return user

def authenticate_user_by_full_name(db: Session, full_name: str, password: str):
    user = get_user_by_full_name(db, full_name)
    
    if not user:
        return None

    # Compare hashed password
    if not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return None

    return user

def update_user(db: Session, user_id: uuid.UUID, data: UserCreate):
    user_instance = db.query(User).filter(User.id == user_id).first()
    if not user_instance:
        return None
    
    # Hash the new password if it's being updated
    if data.password:
        hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt())
        data.password = hashed_password.decode('utf-8')
    
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(user_instance, key, value)
    
    db.commit()
    db.refresh(user_instance)
    return user_instance

def delete_user(db: Session, user_id: uuid.UUID):
    user_instance = db.query(User).filter(User.id == user_id).first()
    if not user_instance:
        return None
    db.delete(user_instance)
    db.commit()
    return user_instance

def create_message(db: Session, data: MessageCreate):
    message_instance = Message(**data.model_dump())
    db.add(message_instance)
    db.commit()
    db.refresh(message_instance)
    return message_instance

def get_messages(db: Session):
    return db.query(Message).all()

def get_message(db: Session, user_id: uuid.UUID):
    return db.query(Message).filter(Message.user_id == user_id).all()

def delete_message(db: Session, message_id: uuid.UUID):
    message_instance = db.query(Message).filter(Message.id == message_id).first()
    if not message_instance:
        return None
    db.delete(message_instance)
    db.commit()
    return message_instance

def delete_messages_by_user(db: Session, user_id: uuid.UUID):
    messages = db.query(Message).filter(Message.user_id == user_id).all()
    for message in messages:
        db.delete(message)
    db.commit()
    return messages