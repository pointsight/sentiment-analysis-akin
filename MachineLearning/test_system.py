#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Çoklu Model Sentiment Analiz Sistemi Test Scripti
"""

import requests
import json

def test_single_comments():
    """Tek yorum testleri"""
    print("🧪 === TEK YORUM TESTLERİ ===\n")
    
    test_cases = [
        {
            "name": "NÖTR YORUM (İş Yeri Talebi)",
            "text": "Personel dinlenme alanlarındaki kahve makinesinden lojmanada talep ediyoruz. Sayın yönetim kuruluna Otel olsun lojman olsun herşey çok iyi sorunsuz fakat biz arkadaşlarımızla bulunmak istiyoruz lojmanda mescit yok bir lojmanda mescitin açılmasını istiyoruz sizden ricamız teşekkür ederiz hayırlı akşamlar dilerim"
        },
        {
            "name": "OLUMLU YORUM",
            "text": "Bu ürün gerçekten harika! Çok memnun kaldım, kalitesi mükemmel ve fiyatına göre çok uygun. Kesinlikle tavsiye ederim."
        },
        {
            "name": "OLUMSUZ YORUM",
            "text": "Bu ürün gerçekten berbat! Hiç memnun kalmadım, kalitesi çok düşük ve fiyatına hiç değmez. Kargo da çok geç geldi, çok kızgınım."
        },
        {
            "name": "KARMAŞIK NÖTR YORUM",
            "text": "Lojmanda yemekhane kalitesi iyi ama çeşit az. Spor salonu eklenmesini istiyoruz. İnternet hızı yeterli fakat otopark yetersiz. Genel olarak memnunuz ama iyileştirmeler yapılmasını talep ediyoruz."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Test {i}: {test_case['name']}")
        print(f"Yorum: {test_case['text'][:80]}...")
        
        try:
            response = requests.post(
                'http://127.0.0.1:8000/analyze',
                json={'text': test_case['text']}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Analiz: {result['analiz']}")
                print(f"✅ Yöntem: {result['yöntem']}")
                print(f"✅ Güven: {result['güven']}")
                
                if result.get('açıklama'):
                    print(f"✅ Açıklama: {result['açıklama']}")
                
                if result.get('model_sonuçları'):
                    model_data = result['model_sonuçları']
                    print(f"✅ Model: {model_data.get('model_used', 'N/A')}")
                    print(f"✅ Tutarlılık: {model_data.get('consistency', 'N/A')}")
                    
                    if 'all_results' in model_data:
                        print("   Model Sonuçları:")
                        for model_result in model_data['all_results']:
                            print(f"     - {model_result['model_id']}: {model_result['sentiment']} ({model_result['confidence']:.3f})")
                
            else:
                print(f"❌ Hata: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Hata: {e}")
        
        print("-" * 80)
        print()

def test_file_upload():
    """Dosya yükleme testi"""
    print("📁 === DOSYA YÜKLEME TESTİ ===\n")
    
    try:
        with open('test_yorumlar.txt', 'rb') as f:
            files = {'file': ('test_yorumlar.txt', f, 'text/plain')}
            
            response = requests.post('http://127.0.0.1:8000/upload', files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Dosya: {result['dosya_adi']}")
                print(f"✅ Yorum Sayısı: {result['yorum_sayisi']}")
                print("\n📊 Sonuçlar:")
                
                for i, comment_result in enumerate(result['sonuclar'], 1):
                    yorum = comment_result['yorum'][:50] + "..." if len(comment_result['yorum']) > 50 else comment_result['yorum']
                    print(f"  {i:2d}. {comment_result['analiz']:8s} ({comment_result['yöntem']:15s}) - {yorum}")
                
                # İstatistikler
                analizler = [r['analiz'] for r in result['sonuclar']]
                print(f"\n📈 İstatistikler:")
                print(f"  Olumlu: {analizler.count('Olumlu')}")
                print(f"  Olumsuz: {analizler.count('Olumsuz')}")
                print(f"  Nötr: {analizler.count('Nötr')}")
                
            else:
                print(f"❌ Hata: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    print("-" * 80)
    print()

def test_system_health():
    """Sistem sağlık kontrolü"""
    print("🏥 === SİSTEM SAĞLIK KONTROLÜ ===\n")
    
    try:
        response = requests.get('http://127.0.0.1:8000/health')
        
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Durum: {health['status']}")
            print(f"✅ Yüklenen Modeller:")
            
            for model_id, model_info in health['models'].items():
                print(f"  - {model_id}: {model_info['name']} ({model_info['status']})")
            
            print(f"✅ Pipeline'lar: {health['pipelines']}")
            
        else:
            print(f"❌ Hata: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    print("-" * 80)
    print()

if __name__ == "__main__":
    print("🚀 Çoklu Model Sentiment Analiz Sistemi Test Ediliyor...\n")
    
    # Sistem sağlık kontrolü
    test_system_health()
    
    # Tek yorum testleri
    test_single_comments()
    
    # Dosya yükleme testi
    test_file_upload()
    
    print("✨ Test tamamlandı!")
