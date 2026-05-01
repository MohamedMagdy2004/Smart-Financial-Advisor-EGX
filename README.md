# EGX Trading System

مشروع تحليل أسهم EGX مع:

- Backend بـ FastAPI
- Frontend بسيط Chatbot
- Modal لتحليل الأخبار
- Groq للقرار النهائي

## الملفات المهمة

- [backend/app.py](backend/app.py)
- [frontend/index.html](frontend/index.html)
- [modal_functions/ai_model.py](modal_functions/ai_model.py)
- [backend/.env.example](backend/.env.example)

## قبل التشغيل

1. انسخ ملف الإعدادات:

   - من [backend/.env.example](backend/.env.example) إلى [backend/.env](backend/.env)

2. ضع القيم التالية داخل [backend/.env](backend/.env):

   - `GROQ_API_KEY`
   - `MODAL_API_TOKEN` إذا كنت ستستخدم Modal

3. تأكد أن `USE_MODAL=True` إذا كان الـ Analyzer على Modal.

## التشغيل السريع

### 1) شغّل الـ Backend

من داخل مجلد المشروع:

```bash
cd backend
docker compose up 
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2) شغّل الـ Frontend

في Terminal جديد:

```bash
cd frontend
python -m http.server 3000
```

### 3) افتح الواجهة

- http://localhost:3000

## ملاحظات مهمة

- لا ترفع ملفات `.env`
- لا ترفع مجلدات `.venv` أو `__pycache__`
- لا ترفع ملفات النتائج داخل [backend/output](backend/output)
- لو ظهر خطأ في Groq، تأكد أن المفتاح موجود في `backend/.env`

## اختبار سريع

```bash
curl http://localhost:8000/health
```
