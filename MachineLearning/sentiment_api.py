from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
import csv
import io
import os
from typing import List, Dict, Any

# Çoklu model yükle
MODELS = {
    "savasy": {
        "name": "savasy/bert-base-turkish-sentiment-cased",
        "description": "Türkçe için özel eğitilmiş sentiment modeli"
    },
    "dbmdz": {
        "name": "dbmdz/bert-base-turkish-cased", 
        "description": "Genel Türkçe BERT modeli"
    }
}

# Model pipeline'ları
pipelines = {}
for model_id, model_info in MODELS.items():
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
        model = AutoModelForSequenceClassification.from_pretrained(model_info["name"])
        pipelines[model_id] = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        print(f"✅ {model_id} modeli yüklendi: {model_info['name']}")
    except Exception as e:
        print(f"❌ {model_id} modeli yüklenemedi: {e}")

# API başlat
app = FastAPI(title="Türkçe Duygu Analizi API - Çoklu Model", version="2.0.0")

# Static dosyaları serve et
app.mount("/static", StaticFiles(directory="static"), name="static")

class Comment(BaseModel):
    text: str

class Comments(BaseModel):
    texts: List[str]

# Etiket eşleme - farklı modeller için
MAPPINGS = {
    "savasy": {"positive": "Olumlu", "negative": "Olumsuz", "neutral": "Nötr"},
    "dbmdz": {"LABEL_0": "Olumsuz", "LABEL_1": "Olumlu", "LABEL_2": "Nötr"}  # Genel BERT için
}

def is_neutral_comment(text: str) -> bool:
    """Nötr yorumları tespit et - özellikle iş yeri talepleri ve önerileri için"""
    text_lower = text.lower()
    
    # Nötr göstergeler (talep, öneri, rica) - daha geniş liste
    neutral_indicators = [
        'talep ediyoruz', 'istiyoruz', 'rica ediyoruz', 'açılmasını istiyoruz',
        'bulunmak istiyoruz', 'teşekkür ederiz', 'hayırlı akşamlar', 'hayırlı günler',
        'personel', 'lojman', 'mescit', 'kahve makinesi', 'dinlenme alanı',
        'yönetim kurulu', 'sizden ricamız', 'açılmasını talep ediyoruz',
        'eklenmesini istiyoruz', 'kurulmasını istiyoruz', 'yapılmasını istiyoruz',
        'düzenlenmesini istiyoruz', 'iyileştirilmesini istiyoruz',
        'olmasını istiyoruz', 'olmasını talep ediyoruz', 'olmasını rica ediyoruz',
        'artırılmasını istiyoruz', 'azaltılmasını istiyoruz', 'değiştirilmesini istiyoruz',
        'yemekhane', 'internet', 'çalışma saati', 'çalışma ortamı', 'sosyal alan',
        'spor salonu', 'otopark', 'ulaşım', 'servis', 'yemek', 'çay', 'kahve',
        'temizlik', 'güvenlik', 'bakım', 'onarım', 'yenileme', 'modernizasyon'
    ]
    
    # Şikayet göstergeleri (gerçekten olumsuz olanlar)
    complaint_indicators = [
        'kötü', 'berbat', 'rezalet', 'çok kötü', 'hiç beğenmedim', 'beğenmedim',
        'şikayet', 'memnun değilim', 'kızgınım', 'sinirliyim', 'üzgünüm',
        'yetersiz', 'kötü kalite', 'düşük kalite', 'sorunlu', 'problemli',
        'çalışmıyor', 'bozuk', 'arızalı', 'hatalı', 'yanlış', 'kırık',
        'eski', 'kirli', 'pis', 'kötü kokuyor', 'gürültülü', 'sıcak', 'soğuk'
    ]
    
    # Pozitif göstergeler
    positive_indicators = [
        'çok iyi', 'harika', 'mükemmel', 'süper', 'güzel', 'beğendim',
        'memnun', 'teşekkür', 'başarılı', 'kaliteli', 'profesyonel',
        'sorunsuz', 'tam istediğimiz gibi', 'çok güzel', 'çok başarılı'
    ]
    
    neutral_count = sum(1 for indicator in neutral_indicators if indicator in text_lower)
    complaint_count = sum(1 for indicator in complaint_indicators if indicator in text_lower)
    positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
    
    # Nötr yorum kriterleri - daha esnek:
    # 1. Nötr göstergeler yeterli (2+)
    # 2. Şikayet göstergeleri az (2'den az)
    # 3. Pozitif göstergeler varsa da nötr olabilir (talep + pozitif = nötr)
    if neutral_count >= 2 and complaint_count <= 1:
        return True
    
    # Özel durum: "herşey çok iyi sorunsuz fakat" gibi yapılar
    if 'fakat' in text_lower or 'ama' in text_lower or 'ancak' in text_lower:
        if neutral_count >= 1:
            return True
    
    # Özel durum: "istiyoruz", "talep ediyoruz" gibi direkt talepler
    if any(word in text_lower for word in ['istiyoruz', 'talep ediyoruz', 'rica ediyoruz']):
        if complaint_count == 0:
            return True
    
    # Özel durum: Personel, lojman, yemekhane gibi iş yeri konuları
    workplace_topics = ['personel', 'lojman', 'yemekhane', 'mescit', 'dinlenme', 'çalışma']
    if any(topic in text_lower for topic in workplace_topics):
        if neutral_count >= 1 and complaint_count <= 1:
            return True
    
    return False

def clean_text(text: str) -> str:
    """Metni normalize et ve temizle"""
    text = text.lower()
    # Türkçe karakterleri koru, gereksiz işaretleri temizle
    text = re.sub(r'[^\w\sçğıöşü]', '', text)
    # Tekrar eden harfleri azalt (ör: aaa -> aa)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_with_model(text: str, model_id: str) -> Dict[str, Any]:
    """Belirli bir model ile analiz"""
    try:
        pipeline = pipelines.get(model_id)
        if not pipeline:
            return {"error": f"Model {model_id} bulunamadı"}
        
        result = pipeline(text)[0]
        label = str(result['label'])
        confidence = float(result['score'])
        
        # Model'e göre etiket eşleme
        mapping = MAPPINGS.get(model_id, {})
        sentiment = mapping.get(label, label)
        
        return {
            "model_id": model_id,
            "sentiment": sentiment,
            "confidence": confidence,
            "raw_label": label
        }
    except Exception as e:
        return {"error": str(e)}

def combine_model_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Çoklu model sonuçlarını birleştir"""
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        return {"error": "Hiçbir model çalışmadı"}
    
    if len(valid_results) == 1:
        return valid_results[0]
    
    # Çoklu model sonuçlarını analiz et
    sentiments = [r["sentiment"] for r in valid_results]
    confidences = [r["confidence"] for r in valid_results]
    
    # En yüksek güven skoruna sahip sonucu al
    best_result = max(valid_results, key=lambda x: x["confidence"])
    
    # Sonuçlar tutarlı mı kontrol et
    unique_sentiments = set(sentiments)
    is_consistent = len(unique_sentiments) == 1
    
    return {
        "final_sentiment": best_result["sentiment"],
        "final_confidence": best_result["confidence"],
        "model_used": best_result["model_id"],
        "consistency": is_consistent,
        "all_results": valid_results,
        "method": "multi_model"
    }

def analyze_comment(comment: str) -> Dict[str, Any]:
    """Tek yorum analizi - gelişmiş hibrit yaklaşım"""
    if not comment or len(comment.strip()) < 2:
        raise HTTPException(status_code=400, detail="Yorum çok kısa veya boş")
    
    # 1. Önce kural tabanlı kontrol
    if is_neutral_comment(comment):
        return {
            "yorum": comment,
            "analiz": "Nötr",
            "güven": 0.95,
            "yöntem": "kural_tabanlı",
            "açıklama": "Talep, öneri veya rica içeren yorum",
            "model_sonuçları": None
        }
    
    cleaned = clean_text(comment)
    if len(cleaned.split()) < 2:
        return {
            "yorum": comment,
            "analiz": "Geçersiz / Yetersiz Yorum",
            "güven": 0.0,
            "yöntem": "kural_tabanlı",
            "model_sonuçları": None
        }
    
    try:
        # 2. Çoklu model analizi
        model_results = []
        for model_id in pipelines.keys():
            result = analyze_with_model(cleaned, model_id)
            if "error" not in result:
                model_results.append(result)
        
        # 3. Model sonuçlarını birleştir
        combined_result = combine_model_results(model_results)
        
        if "error" in combined_result:
            raise Exception(combined_result["error"])
        
        final_sentiment = combined_result["final_sentiment"]
        final_confidence = combined_result["final_confidence"]
        
        # 4. Model sonucunu kontrol et - eğer çok yüksek güvenle yanlış sınıflandırıyorsa
        if final_confidence > 0.9 and final_sentiment == "Olumsuz":
            if is_neutral_comment(comment):
                return {
                    "yorum": comment,
                    "analiz": "Nötr",
                    "güven": 0.90,
                    "yöntem": "hibrit_düzeltme",
                    "açıklama": "Model yanlış sınıflandırdı, kural tabanlı düzeltme uygulandı",
                    "model_sonuçları": combined_result
                }
        
        return {
            "yorum": comment,
            "analiz": final_sentiment,
            "güven": round(final_confidence, 3),
            "yöntem": combined_result["method"],
            "model_sonuçları": combined_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")

def analyze_comments(comments: List[str]) -> List[Dict[str, Any]]:
    """Toplu yorum analizi"""
    if not comments:
        return []
    
    # Boş yorumları filtrele
    valid_comments = [c.strip() for c in comments if c.strip()]
    if not valid_comments:
        return []
    
    try:
        outputs = []
        for comment in valid_comments:
            outputs.append(analyze_comment(comment))
        return outputs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Toplu analiz hatası: {str(e)}")

def parse_csv_file(file_content: bytes) -> List[str]:
    """CSV dosyasından yorumları oku"""
    try:
        content = file_content.decode('utf-8')
        csv_reader = csv.reader(io.StringIO(content))
        comments = []
        for row in csv_reader:
            if row and row[0].strip():
                comments.append(row[0].strip())
        return comments
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV okuma hatası: {str(e)}")

def parse_txt_file(file_content: bytes) -> List[str]:
    """TXT dosyasından yorumları oku"""
    try:
        content = file_content.decode('utf-8')
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        return lines
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"TXT okuma hatası: {str(e)}")

# Veritabanı için basit JSON dosya sistemi
import json
import os
from datetime import datetime

# Veritabanı dosyası
DB_FILE = "sentiment_database.json"

def load_database():
    """Veritabanını yükle"""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {
        "comments": [],
        "statistics": {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "invalid": 0
        },
        "last_updated": datetime.now().isoformat()
    }

def save_database(db):
    """Veritabanını kaydet"""
    db["last_updated"] = datetime.now().isoformat()
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def add_comment_to_database(comment_data):
    """Yorumu veritabanına ekle"""
    db = load_database()
    
    # Yorum ID'si oluştur
    comment_id = len(db["comments"]) + 1
    
    # Yorum verisi
    comment_entry = {
        "id": comment_id,
        "text": comment_data["yorum"],
        "sentiment": comment_data["analiz"],
        "confidence": comment_data["güven"],
        "method": comment_data.get("yöntem", "bilinmiyor"),
        "timestamp": datetime.now().isoformat(),
        "model_results": comment_data.get("model_sonuçları", None)
    }
    
    db["comments"].append(comment_entry)
    
    # İstatistikleri güncelle
    db["statistics"]["total"] += 1
    sentiment = comment_data["analiz"]
    if sentiment == "Olumlu":
        db["statistics"]["positive"] += 1
    elif sentiment == "Olumsuz":
        db["statistics"]["negative"] += 1
    elif sentiment == "Nötr":
        db["statistics"]["neutral"] += 1
    else:
        db["statistics"]["invalid"] += 1
    
    save_database(db)
    return comment_id

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Ana sayfa - HTML arayüzünü serve et"""
    try:
        html_path = os.path.join("static", "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            # HTML dosyası bulunamadıysa basit bir sayfa döndür
            return HTMLResponse(content="""
            <!DOCTYPE html>
            <html lang="tr">
            <head>
                <meta charset="UTF-8">
                <title>Türkçe Duygu Analizi - Çoklu Model</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .method { font-weight: bold; color: #007bff; }
                    .model-info { background: #e8f4fd; padding: 15px; margin: 10px 0; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎭 Türkçe Duygu Analizi API - Çoklu Model</h1>
                    <p>API çalışıyor! Kullanılabilir endpoint'ler:</p>
                    
                    <div class="model-info">
                        <h3>🤖 Yüklü Modeller</h3>
                        <ul>
                            <li><strong>savasy</strong>: Türkçe sentiment için özel eğitilmiş</li>
                            <li><strong>dbmdz</strong>: Genel Türkçe BERT modeli</li>
                        </ul>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> <code>/</code> - Bu sayfa
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">POST</span> <code>/analyze</code> - Tek yorum analizi
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">POST</span> <code>/analyze-batch</code> - Toplu yorum analizi (JSON)
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">POST</span> <code>/upload</code> - Dosya yükleme ve analiz
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> <code>/health</code> - Sağlık kontrolü
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> <code>/docs</code> - Swagger dokümantasyonu
                    </div>
                    
                    <h3>📝 Örnek Kullanım</h3>
                    <p><strong>Tek yorum analizi:</strong></p>
                    <pre>curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Bu ürün harika!"}'</pre>
                    
                    <p><strong>Dosya yükleme:</strong></p>
                    <pre>curl -X POST "http://localhost:8000/upload" \
     -F "file=@ornek_yorumlar.txt"</pre>
                </div>
            </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(content=f"""
        <html>
            <head><title>Hata</title></head>
            <body><h1>Hata oluştu: {str(e)}</h1></body>
        </html>
        """)

@app.post("/analyze")
async def analyze_single(comment: Comment):
    """Tek yorum analizi - gelişmiş hibrit yaklaşım"""
    if not comment or len(comment.strip()) < 2:
        raise HTTPException(status_code=400, detail="Yorum çok kısa veya boş")
    
    # 1. Önce kural tabanlı kontrol
    if is_neutral_comment(comment.text):
        result = {
            "yorum": comment.text,
            "analiz": "Nötr",
            "güven": 0.95,
            "yöntem": "kural_tabanlı",
            "açıklama": "Talep, öneri veya rica içeren yorum",
            "model_sonuçları": None
        }
    else:
        cleaned = clean_text(comment.text)
        if len(cleaned.split()) < 2:
            result = {
                "yorum": comment.text,
                "analiz": "Geçersiz / Yetersiz Yorum",
                "güven": 0.0,
                "yöntem": "kural_tabanlı",
                "model_sonuçları": None
            }
        else:
            try:
                # 2. Çoklu model analizi
                model_results = []
                for model_id in pipelines.keys():
                    result_model = analyze_with_model(cleaned, model_id)
                    if "error" not in result_model:
                        model_results.append(result_model)
                
                # 3. Model sonuçlarını birleştir
                combined_result = combine_model_results(model_results)
                
                if "error" in combined_result:
                    raise Exception(combined_result["error"])
                
                final_sentiment = combined_result["final_sentiment"]
                final_confidence = combined_result["final_confidence"]
                
                # 4. Model sonucunu kontrol et - eğer çok yüksek güvenle yanlış sınıflandırıyorsa
                if final_confidence > 0.9 and final_sentiment == "Olumsuz":
                    if is_neutral_comment(comment.text):
                        result = {
                            "yorum": comment.text,
                            "analiz": "Nötr",
                            "güven": 0.90,
                            "yöntem": "hibrit_düzeltme",
                            "açıklama": "Model yanlış sınıflandırdı, kural tabanlı düzeltme uygulandı",
                            "model_sonuçları": combined_result
                        }
                    else:
                        result = {
                            "yorum": comment.text,
                            "analiz": final_sentiment,
                            "güven": round(final_confidence, 3),
                            "yöntem": combined_result["method"],
                            "model_sonuçları": combined_result
                        }
                else:
                    result = {
                        "yorum": comment.text,
                        "analiz": final_sentiment,
                        "güven": round(final_confidence, 3),
                        "yöntem": combined_result["method"],
                        "model_sonuçları": combined_result
                    }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")
    
    # Yorumu veritabanına ekle
    try:
        comment_id = add_comment_to_database(result)
        result["comment_id"] = comment_id
    except Exception as e:
        print(f"Veritabanı hatası: {e}")
        result["comment_id"] = None
    
    return result

@app.post("/analyze-batch")
async def analyze_batch(payload: Comments):
    """JSON ile toplu yorum analizi"""
    return analyze_comments(payload.texts)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Dosya yükleme ve analiz"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dosya adı bulunamadı")
    
    # Dosya boyutu kontrolü (5MB)
    if file.size and file.size > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Dosya çok büyük (max 5MB)")
    
    try:
        content = await file.read()
        
        if file.filename.lower().endswith('.csv'):
            comments = parse_csv_file(content)
        elif file.filename.lower().endswith('.txt'):
            comments = parse_txt_file(content)
        else:
            raise HTTPException(status_code=400, detail="Sadece .csv ve .txt dosyaları desteklenir")
        
        if not comments:
            raise HTTPException(status_code=400, detail="Dosyada geçerli yorum bulunamadı")
        
        results = analyze_comments(comments)
        return {
            "dosya_adi": file.filename,
            "yorum_sayisi": len(comments),
            "sonuclar": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dosya işleme hatası: {str(e)}")

@app.get("/health")
async def health_check():
    """Sağlık kontrolü"""
    return {
        "status": "healthy", 
        "models": {model_id: {"name": info["name"], "status": "loaded"} for model_id, info in MODELS.items()},
        "pipelines": list(pipelines.keys())
    }

@app.get("/statistics")
async def get_statistics():
    """Güncel istatistikleri getir"""
    try:
        db = load_database()
        return {
            "status": "success",
            "data": db["statistics"],
            "last_updated": db["last_updated"],
            "total_comments": len(db["comments"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İstatistik hatası: {str(e)}")

@app.get("/comments")
async def get_comments(limit: int = 50, offset: int = 0):
    """Yorumları getir"""
    try:
        db = load_database()
        comments = db["comments"][offset:offset+limit]
        return {
            "status": "success",
            "data": comments,
            "total": len(db["comments"]),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yorum getirme hatası: {str(e)}")

@app.get("/comments/{comment_id}")
async def get_comment(comment_id: int):
    """Belirli bir yorumu getir"""
    try:
        db = load_database()
        if comment_id <= 0 or comment_id > len(db["comments"]):
            raise HTTPException(status_code=404, detail="Yorum bulunamadı")
        
        comment = db["comments"][comment_id - 1]
        return {
            "status": "success",
            "data": comment
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yorum getirme hatası: {str(e)}")

@app.delete("/comments/{comment_id}")
async def delete_comment(comment_id: int):
    """Yorumu sil"""
    try:
        db = load_database()
        if comment_id <= 0 or comment_id > len(db["comments"]):
            raise HTTPException(status_code=404, detail="Yorum bulunamadı")
        
        # Yorumu sil
        deleted_comment = db["comments"].pop(comment_id - 1)
        
        # İstatistikleri güncelle
        sentiment = deleted_comment["sentiment"]
        db["statistics"]["total"] -= 1
        if sentiment == "Olumlu":
            db["statistics"]["positive"] -= 1
        elif sentiment == "Olumsuz":
            db["statistics"]["negative"] -= 1
        elif sentiment == "Nötr":
            db["statistics"]["neutral"] -= 1
        else:
            db["statistics"]["invalid"] -= 1
        
        # ID'leri yeniden düzenle
        for i, comment in enumerate(db["comments"]):
            comment["id"] = i + 1
        
        save_database(db)
        
        return {
            "status": "success",
            "message": "Yorum başarıyla silindi",
            "deleted_comment": deleted_comment
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yorum silme hatası: {str(e)}")

@app.post("/analyze-bulk")
async def analyze_bulk_comments(comments: List[str]):
    """Toplu yorum analizi"""
    if not comments or len(comments) == 0:
        raise HTTPException(status_code=400, detail="Yorum listesi boş")
    
    if len(comments) > 100:
        raise HTTPException(status_code=400, detail="Maksimum 100 yorum analiz edilebilir")
    
    results = []
    start_time = time.time()
    
    for comment_text in comments:
        try:
            # Her yorumu analiz et
            comment_obj = Comment(text=comment_text)
            result = await analyze_single(comment_obj)
            results.append(result)
        except Exception as e:
            results.append({
                "yorum": comment_text,
                "analiz": "Hata",
                "güven": 0.0,
                "yöntem": "hata",
                "açıklama": str(e),
                "comment_id": None
            })
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return {
        "status": "success",
        "message": f"{len(comments)} yorum analiz edildi",
        "results": results,
        "processing_time": round(processing_time, 2),
        "average_time_per_comment": round(processing_time / len(comments), 3)
    }

@app.get("/export")
async def export_data(format: str = "json"):
    """Veriyi dışa aktar"""
    try:
        db = load_database()
        
        if format.lower() == "csv":
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # CSV başlıkları
            writer.writerow(["ID", "Yorum", "Sentiment", "Güven", "Yöntem", "Tarih"])
            
            # Verileri yaz
            for comment in db["comments"]:
                writer.writerow([
                    comment["id"],
                    comment["text"],
                    comment["sentiment"],
                    comment["confidence"],
                    comment["method"],
                    comment["timestamp"]
                ])
            
            output.seek(0)
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=sentiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
            )
        
        else:  # JSON format
            return {
                "status": "success",
                "data": db,
                "export_time": datetime.now().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dışa aktarma hatası: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
