# دليل تشغيل المشروع على نفس الجهاز (واضح وبالخطوات)

هذا هو **الملف الوحيد** الذي تحتاجه لتشغيل المشروع مرة أخرى بدون لخبطة.

---

## شكل التشغيل الصحيح

> **للتشغيل بدون Frontend (مطلوب حاليًا):** يكفي Terminal واحد للـ Backend + أوامر API/CLI.

ستستخدم 2 ترمينال فقط:

- **Terminal 1 (Backend):** لتشغيل API على المنفذ 8000
- **Terminal 2 (Frontend):** لتشغيل الواجهة على المنفذ 3000

> لا تكتب أوامر الاختبار داخل ترمينال السيرفر.

### تشغيل بدون واجهة (Backend Only)

- **Terminal 1 (Backend):** تشغيل API على `8000`
- ثم نفّذ طلبات `curl` أو شغّل سكربت `backend/run_part3.py`

---

## أول مرة فقط (تجهيز البيئة)

### 1) الدخول إلى المشروع وتفعيل البيئة الافتراضية

```bash
cd "/media/mo/D/pforpen/EGX extarctor news/project"
source ../.venv/bin/activate
```

### 2) تثبيت المكتبات

```bash
pip install -r backend/requirements.txt
pip install -r modal_functions/requirements.txt
python -m playwright install chromium
```

### 3) تسجيل دخول Modal (مرة واحدة)

```bash
modal token new
```

### 4) نشر موديل التحليل على Modal

```bash
cd modal_functions
modal deploy ai_model.py
cd ..
```

لو ظهر نجاح deployment، إذًا Modal جاهز.

---

## التشغيل اليومي (كل مرة تريد تشغيل المشروع)

## Terminal 1 — تشغيل Backend

```bash
cd "/media/mo/D/pforpen/EGX extarctor news/project"
source ../.venv/bin/activate
kill -9 $(lsof -i :8000 -t) 2>/dev/null
export USE_MODAL=True
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

يجب أن ترى:
- `Uvicorn running on http://0.0.0.0:8000`
- `Application startup complete`

## Terminal 2 — تشغيل Frontend

```bash
cd "/media/mo/D/pforpen/EGX extarctor news/project/frontend"
kill -9 $(lsof -i :3000 -t) 2>/dev/null
python -m http.server 3000
```

ثم افتح المتصفح على:

`http://localhost:3000`

> يمكنك تجاهل هذا الجزء بالكامل إذا كنت تعمل بدون واجهة.

> مهم: لا تستخدم `file://...` ولا `http://0.0.0.0:3000`.

---

## تجربة المشروع كمستخدم

من الواجهة:

1. اختر شركة (مثال: COMI)
2. اضغط **سحب الأخبار**
3. ثم اضغط **تحليل الأخبار**

عند النجاح، سيتم حفظ النتائج في:

- `project/backend/output/`

---

## واجهة الشات الجديدة (End-to-End)

الواجهة الآن تعمل كـ Chatbot وتنفذ تلقائيًا:

1) استنتاج رمز السهم من رسالة المستخدم (LLM)
2) تشغيل الجزء الأول (سحب + تحليل الأخبار)
3) تشغيل الجزء الثاني (توليد التحليل المالي)
4) تشغيل الجزء الثالث (القرار النهائي)

لتشغيلها:

1. شغل الـ Backend على `http://localhost:8000`
2. افتح الواجهة من:

`http://localhost:3000`

3. اكتب رسالة مثل:

`حلل سهم التجاري الدولي وهل مناسب للشراء الآن؟`

4. عدّل أسئلة المخاطر من اللوحة اليسار ثم أرسل.

الواجهة تستدعي endpoint جديد:

- `POST /chat/message`

والناتج يعرض:

- القرار المختصر داخل الشات
- مسارات الملفات الناتجة من الأجزاء الثلاثة

---

## اختبار سريع (اختياري)

في أي ترمينال جديد:

### صحة السيرفر

```bash
curl http://localhost:8000/health
```

### اختبار السحب فقط

```bash
curl "http://localhost:8000/pipeline/scrape?ticker=COMI&max_news=3"
```

### اختبار كامل (سحب + تحليل)

```bash
curl -X POST "http://localhost:8000/pipeline/full?ticker=COMI&max_news=3"

### اختبار الجزء الثالث (Decision Engine)

> يتطلب ملفين JSON:
> 1) ملف الأخبار من الجزء الأول (مثل: `backend/output/COMI_*.json`)
> 2) ملف الجزء الثاني من النوتبوك (Part2 JSON)

#### عبر API

```bash
curl -X POST "http://localhost:8000/pipeline/decision" \
	-H "Content-Type: application/json" \
	-d '{
		"ticker": "COMI",
		"news_json_path": "COMI_20260325_082333.json",
		"financial_json_path": "/absolute/path/to/part2_financial.json",
		"user_risk_profile": "moderate"
	}'
```

#### عبر CLI (بدون API Request)

```bash
cd "/media/mo/D/pforpen/project/backend"
python run_part3.py \
	--ticker COMI \
	--news output/COMI_20260325_082333.json \
	--financial /absolute/path/to/part2_financial.json \
	--ask-risk-questions
```

الناتج النهائي يُحفظ تلقائيًا داخل:

- `backend/output/*_final_decision_*.json`

وسيتم أيضًا إنشاء نسخة محدثة من ملف الجزء الثاني بعد إضافة دعم/مقاومة من Mubasher:

- `backend/output/*_financial_enriched_*.json`
```

---

## إيقاف كل الخدمات مرة واحدة

```bash
kill -9 $(lsof -t -i:8000 -i:3000 -i:5173) 2>/dev/null
```

وتأكد أنها توقفت:

```bash
lsof -i :8000 -i :3000 -i :5173
```

لو لا يوجد مخرجات، فكل شيء توقف بنجاح.

---

## مشاكل شائعة وحلها السريع

### 1) `Address already in use`

المنفذ مستخدم من عملية قديمة.

الحل:

```bash
kill -9 $(lsof -i :8000 -t) 2>/dev/null
kill -9 $(lsof -i :3000 -t) 2>/dev/null
```

### 2) `NetworkError when attempting to fetch resource`

الواجهة ليست على `localhost:3000` أو Backend غير شغال.

الحل:
- افتح الواجهة على `http://localhost:3000`
- تأكد أن `curl http://localhost:8000/health` يرجع `healthy`

### 3) فشل `modal deploy`

الحل:

```bash
source ../.venv/bin/activate
pip install modal torch transformers accelerate json-repair
cd modal_functions
modal deploy ai_model.py
```

---

## ملخص سريع جدًا (نسخ ولصق)

### Backend

```bash
cd "/media/mo/D/pforpen/EGX extarctor news/project" && source ../.venv/bin/activate && kill -9 $(lsof -i :8000 -t) 2>/dev/null && export USE_MODAL=True && cd backend && python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd "/media/mo/D/pforpen/EGX extarctor news/project/frontend" && kill -9 $(lsof -i :3000 -t) 2>/dev/null && python -m http.server 3000
```

ثم افتح:

`http://localhost:3000`
