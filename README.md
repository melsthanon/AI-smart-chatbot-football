# AI-Football
AI Football Chatbot Workflow using FAISS

โหลดและเตรียมโมเดล

ใช้โมเดล sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ที่รองรับหลายภาษา รวมถึงภาษาไทย

แปลงคำถามเป็นเวกเตอร์ (Embedding) เพื่อให้สามารถค้นหาได้รวดเร็วและแม่นยำ

ประมวลผลข้อมูลจาก CSV

โหลดข้อมูลจาก football_QA_thai.csv โดยใช้ pandas

ตรวจสอบโครงสร้างข้อมูลให้มีคอลัมน์ Question และ Answer

สร้างเวกเตอร์ของคำถามและเก็บไว้ใน FAISS สำหรับการค้นหาคำตอบ

การค้นหาข้อมูลโดยใช้ FAISS

เมื่อผู้ใช้ถามคำถาม ระบบจะสร้างเวกเตอร์จากคำถามนั้น

ใช้ FAISS ค้นหาคำถามที่ใกล้เคียง (Top 3)

หาก FAISS ไม่พบคำตอบที่มีความใกล้เคียงเพียงพอ ระบบจะเรียกใช้ GPT-4 เพื่อสร้างคำตอบเพิ่มเติม

โครงสร้าง Flask Web Application

Endpoint /: แสดงหน้าเว็บหลัก (chat.html)

API /ask:

รับคำถามจากผู้ใช้ผ่าน JSON

ค้นหาคำตอบโดยใช้ FAISS และ GPT-4

ส่งคำตอบกลับในรูปแบบ JSON

การ Deploy ระบบ

รัน Flask App บน 0.0.0.0:5000 พร้อม debug=True ในโหมดพัฒนา

สามารถปรับปรุงการใช้งานโดยใช้ Gunicorn หรือ Docker สำหรับ Production Deployment
