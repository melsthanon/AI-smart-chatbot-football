from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import openai
import logging
import os


# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ตั้งค่า OpenAI API Key (ยังคงใช้แบบ hardcoded)
openai.api_key = "API keys on readme"

# โหลดโมเดลฝังข้อความ
try:
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    logger.info("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    logger.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
    exit("Model loading failed")

# โหลดข้อมูลจาก CSV
csv_file = "football_QA_thai.csv"

try:
    df = pd.read_csv(csv_file, encoding="utf-8")
    logger.info(f"✅ โหลดข้อมูลสำเร็จ! พบ {len(df)} แถว")
except Exception as e:
    logger.error(f"❌ เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
    exit("Failed to load data")

# ตรวจสอบว่าข้อมูลมีคอลัมน์ที่จำเป็นหรือไม่
if not {"Question", "Answer"}.issubset(df.columns):
    raise ValueError("ไฟล์ CSV ต้องมีคอลัมน์ 'Question' และ 'Answer'")

# ฝังข้อความของคำถามใน CSV เป็นเวกเตอร์และสร้าง FAISS index
try:
    embeddings = model.encode(df["Question"].tolist(), convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    logger.info("✅ สร้าง FAISS index สำเร็จ")
except Exception as e:
    logger.error(f"❌ เกิดข้อผิดพลาดในการสร้าง FAISS index: {e}")
    exit("FAISS index creation failed")

def generate_answer(query, top_k=3):
    """
    ดึงข้อมูลที่เกี่ยวข้องจาก CSV ด้วย FAISS แล้วนำมารวมกับคำถามเพื่อสร้าง prompt ส่งให้ ChatGPT
    """
    try:
        # ฝังข้อความคำถามของผู้ใช้
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการค้นหาใน FAISS index: {e}")
        return f"❌ เกิดข้อผิดพลาดในการค้นหา: {e}"

    # ตรวจสอบว่า FAISS ค้นหาผลลัพธ์ที่เกี่ยวข้องหรือไม่
    if distances[0][0] > 1.0:  # หากระยะห่าง (distance) สูงเกินไป หมายความว่าไม่พบคำตอบที่มีความใกล้เคียง
        logger.info("❌ FAISS ไม่พบคำตอบที่ดีพอ จะใช้คำตอบจาก ChatGPT แทน")
        # เรียก ChatGPT ถ้า FAISS ไม่พบคำตอบที่ดีพอ
        return get_chatgpt_answer(query)
    
    # รวมข้อมูลที่ดึงได้เป็น context (โดยดึงคู่คำถาม-คำตอบ)
    retrieved_context = ""
    for idx in indices[0]:
        row = df.iloc[idx]
        retrieved_context += f"คำถาม: {row['Question']}\nคำตอบ: {row['Answer']}\n\n"
    
    # สร้าง prompt สำหรับ ChatGPT โดยรวม context ที่ดึงมา
    prompt = f"""ข้อมูลที่เกี่ยวข้อง:
{retrieved_context}
คำถาม: {query}
คำตอบ: คุณเป็นผู้เชี่ยวชาญด้านฟุตบอล โปรดให้คำตอบที่ถูกต้องและละเอียดตามข้อมูลที่มี"""
    
    logger.info("Prompt สำหรับ ChatGPT:\n" + prompt)
    
    try:
        # ใช้ OpenAI API เวอร์ชันใหม่กับ GPT-3.5 หรือ GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",  # ใช้ gpt-3.5 หรือ gpt-4 หากเข้าถึงได้
            messages=[
                {"role": "user", "content": prompt}  # ส่งข้อความ prompt ไปเป็น content
            ],
            temperature=0.7,
            max_tokens=150  # คุณสามารถตั้งค่า max_tokens ตามต้องการ
        )
        answer = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการเรียก ChatGPT: {e}")
        answer = f"❌ เกิดข้อผิดพลาด: {e}"
    
    return answer

def get_chatgpt_answer(query):
    """
    ฟังก์ชันสำหรับเรียก ChatGPT โดยตรงเมื่อ FAISS ไม่สามารถตอบได้
    """
    prompt = f"""คำถาม: {query}
    คำตอบ: คุณเป็นผู้เชี่ยวชาญด้านฟุตบอล โปรดให้คำตอบที่ถูกต้องและละเอียดตามข้อมูลที่มี"""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # ใช้ gpt-3.5 หรือ gpt-4 หากเข้าถึงได้
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        answer = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการเรียก ChatGPT: {e}")
        answer = f"❌ เกิดข้อผิดพลาด: {e}"
    
    return answer

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "กรุณาระบุคำถาม"}), 400
    answer = generate_answer(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
