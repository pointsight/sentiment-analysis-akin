# 🎭 Türkçe Duygu Analizi Web Uygulaması

Bu proje, Türkçe yorumları analiz eden ve duygusal tonlarını sınıflandıran bir web uygulamasıdır. FastAPI backend'i ve modern HTML/JavaScript frontend'i ile geliştirilmiştir.

## ✨ Özellikler

- **Tek Yorum Analizi**: Tek bir yorumu anında analiz edin
- **Toplu Analiz**: CSV veya TXT dosyalarından yorumları toplu olarak analiz edin
- **Türkçe Desteği**: Özel olarak Türkçe için eğitilmiş BERT modeli
- **Çoklu Model Desteği**: İki farklı model ile hibrit analiz
- **Kural Tabanlı Düzeltme**: İş yeri talepleri için özel nötr tespit
- **Modern Arayüz**: Responsive ve kullanıcı dostu web arayüzü
- **Gerçek Zamanlı Sonuçlar**: Anında analiz sonuçları ve güven skorları

## 🚀 Kurulum

### 1. Gereksinimler
- Python 3.8+
- pip (Python paket yöneticisi)

### 2. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatın
```bash
python -m uvicorn sentiment_api:app --host 0.0.0.0 --port 8000 --reload
```

## 🌐 Kullanım

### Web Arayüzü
Tarayıcınızda `http://localhost:8000` adresini açın.

### API Endpoint'leri

#### 1. Tek Yorum Analizi
```bash
POST /analyze
Content-Type: application/json

{
  "text": "Bu ürün gerçekten harika! Çok memnun kaldım."
}
```

**Yanıt:**
```json
{
  "yorum": "Bu ürün gerçekten harika! Çok memnun kaldım.",
  "analiz": "Olumlu",
  "güven": 0.963,
  "yöntem": "multi_model",
  "model_sonuçları": {
    "final_sentiment": "Olumlu",
    "final_confidence": 0.963,
    "model_used": "savasy",
    "consistency": true,
    "all_results": [...]
  }
}
```

#### 2. Toplu Yorum Analizi (JSON)
```bash
POST /analyze-batch
Content-Type: application/json

{
  "texts": [
    "Ürün harika!",
    "Hiç memnun kalmadım",
    "Personel yemekhanesinde daha fazla çeşit yemek olmasını istiyoruz"
  ]
}
```

#### 3. Dosya Yükleme ve Analiz
```bash
POST /upload
Content-Type: multipart/form-data

file: [CSV veya TXT dosyası]
```

**Desteklenen Formatlar:**
- **CSV**: Her satırda bir yorum (opsiyonel başlık satırı)
- **TXT**: Her satırda bir yorum
- **Maksimum boyut**: 5MB

**Yanıt:**
```json
{
  "dosya_adi": "yorumlar.txt",
  "yorum_sayisi": 20,
  "sonuclar": [
    {
      "yorum": "Ürün harika!",
      "analiz": "Olumlu",
      "güven": 0.95,
      "yöntem": "multi_model"
    }
  ]
}
```

#### 4. Sağlık Kontrolü
```bash
GET /health
```

**Yanıt:**
```json
{
  "status": "healthy",
  "models": {
    "savasy": {
      "name": "savasy/bert-base-turkish-sentiment-cased",
      "status": "loaded"
    },
    "dbmdz": {
      "name": "dbmdz/bert-base-turkish-cased",
      "status": "loaded"
    }
  },
  "pipelines": ["savasy", "dbmdz"]
}
```

#### 5. API Dokümantasyonu
```bash
GET /docs  # Swagger UI
GET /openapi.json  # OpenAPI JSON
```

## 📁 Dosya Yapısı

```
MachineLearning/
├── sentiment_api.py              # FastAPI backend (çoklu model)
├── sentiment_tr.py               # Komut satırı aracı
├── requirements.txt              # Python bağımlılıkları
├── static/
│   └── index.html               # Web arayüzü
├── ornek_yorumlar.txt           # Genel örnek yorumlar (TXT)
├── ornek_yorumlar.csv           # Genel örnek yorumlar (CSV)
├── test_yorumlar_detayli.txt    # Detaylı test yorumları (TXT)
├── test_yorumlar_detayli.csv    # Detaylı test yorumları (CSV)
├── test_system.py               # Sistem test scripti
└── README.md                    # Bu dosya
```

## 🧪 Test

### 1. Web Arayüzü ile Test
1. `http://localhost:8000` adresini açın
2. **Tek Yorum Analizi** bölümünde örnek yorum yazın
3. **Toplu Yorum Analizi** bölümünde örnek dosyaları yükleyin

### 2. Komut Satırı ile Test
```bash
# Tek yorum
python sentiment_tr.py

# Demo yorumlar
python sentiment_tr.py --demo

# Dosyadan yorumlar
python sentiment_tr.py --file ornek_yorumlar.txt
```

### 3. Sistem Test Scripti
```bash
# Kapsamlı sistem testi
python test_system.py
```

### 4. API ile Test
```bash
# Tek yorum
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Bu ürün harika!"}'

# Dosya yükleme
curl -X POST "http://localhost:8000/upload" \
     -F "file=@ornek_yorumlar.txt"
```

## 📊 Analiz Sonuçları

### Sınıflandırma
- **Olumlu** 🟢: Pozitif duygular, memnuniyet
- **Olumsuz** 🔴: Negatif duygular, memnuniyetsizlik  
- **Nötr** 🟡: Tarafsız, kararsız ifadeler, iş yeri talepleri

### Güven Skoru
- **0.0 - 1.0** arasında (1.0 = %100 güven)
- Yüksek güven skorları daha kesin sonuçlar gösterir

### Analiz Yöntemleri
- **kural_tabanlı**: İş yeri talepleri için özel kurallar
- **multi_model**: Çoklu model analizi
- **hibrit_düzeltme**: Model + kural tabanlı düzeltme

## 🤖 Model Sistemi

### Yüklenen Modeller
1. **savasy/bert-base-turkish-sentiment-cased**
   - Türkçe sentiment analizi için özel eğitilmiş
   - Yüksek güven skorları
   - Sentiment sınıflandırması için optimize

2. **dbmdz/bert-base-turkish-cased**
   - Genel Türkçe BERT modeli
   - Geniş kelime hazinesi
   - Çok amaçlı kullanım

### Hibrit Yaklaşım
- **Kural Tabanlı**: İş yeri talepleri, öneriler, ricalar
- **Model Tabanlı**: Genel sentiment analizi
- **Akıllı Birleştirme**: En güvenilir sonucu seçme

## 📝 Örnek Dosyalar

### ornek_yorumlar.txt
```
# 🎭 TÜRKÇE DUYGU ANALİZİ - ÖRNEK YORUMLAR
# Bu dosya farklı türde yorumları test etmek için hazırlanmıştır

# ===== OLUMLU YORUMLAR =====
Bu ürün gerçekten harika! Çok memnun kaldım.
Müşteri hizmetleri çok iyi, teşekkür ederiz.
...

# ===== NÖTR YORUMLAR (İş Yeri Talepleri) =====
Personel dinlenme alanlarındaki kahve makinesinden lojmanada talep ediyoruz.
Lojmanda internet hızının artırılmasını talep ediyoruz.
...
```

### test_yorumlar_detayli.csv
```csv
test_no,yorum,kategori,beklenen_sonuç,açıklama
1,Personel dinlenme alanlarındaki kahve makinesinden...,Nötr,Nötr,İş yeri talebi - karmaşık yapı
2,Bu ürün gerçekten harika!...,Olumlu,Olumlu,Net olumlu yorum
...
```

## 🔧 Geliştirme

### Model Değiştirme
`sentiment_api.py` dosyasında `MODELS` sözlüğünü düzenleyin:
```python
MODELS = {
    "yeni_model": {
        "name": "yeni_model_adi",
        "description": "Model açıklaması"
    }
}
```

### Yeni Endpoint Ekleme
```python
@app.post("/yeni-endpoint")
async def yeni_fonksiyon():
    return {"message": "Yeni endpoint"}
```

### Frontend Özelleştirme
`static/index.html` dosyasını düzenleyin.

## 🚨 Sorun Giderme

### Port Hatası
```
[Errno 10048] error while attempting to bind on address
```
**Çözüm:** Farklı port kullanın
```bash
python -m uvicorn sentiment_api:app --port 8001
```

### Model İndirme Hatası
```
ModuleNotFoundError: No module named 'transformers'
```
**Çözüm:** Bağımlılıkları yeniden yükleyin
```bash
pip install -r requirements.txt --upgrade
```

### Dosya Yükleme Hatası
```
Form data requires "python-multipart" to be installed
```
**Çözüm:** Multipart paketini yükleyin
```bash
pip install python-multipart
```

### Model Yükleme Hatası
```
Some weights of BertForSequenceClassification were not initialized
```
**Bu normal bir uyarıdır, model çalışmaya devam eder.**

## 🎯 Test Senaryoları

### 1. **Basit Yorumlar**
- ✅ "Harika!" → Olumlu
- ✅ "Berbat!" → Olumsuz
- ✅ "İdare eder" → Nötr

### 2. **İş Yeri Talepleri**
- ✅ "Personel yemekhanesinde daha fazla çeşit yemek olmasını istiyoruz" → Nötr
- ✅ "Lojmanda spor salonu eklenmesini talep ediyoruz" → Nötr

### 3. **Karmaşık Yorumlar**
- ✅ "Herşey çok iyi fakat spor salonu eklenmesini istiyoruz" → Nötr
- ✅ "Ürün kaliteli ama pahalı" → Nötr

### 4. **Model Tutarlılığı**
- ✅ Her iki model de aynı sonucu verirse → Tutarlı
- ✅ Farklı sonuçlar verirse → En yüksek güven skorlu seçilir

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Commit yapın (`git commit -am 'Yeni özellik eklendi'`)
4. Push yapın (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

Sorularınız için issue açın veya pull request gönderin.

---

**Not:** İlk çalıştırmada modeller indirilecektir. Bu işlem internet bağlantısı gerektirir ve biraz zaman alabilir.
