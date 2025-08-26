#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment Analiz Sistemi İstatistik Analizörü
"""

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
from typing import Dict, List, Any

class SentimentStatisticsAnalyzer:
    def __init__(self, api_url: str = "http://127.0.0.1:8000"):
        self.api_url = api_url
        self.results_cache = {}
        
    def analyze_single_comment(self, comment: str) -> Dict[str, Any]:
        """Tek yorum analizi"""
        try:
            response = requests.post(
                f"{self.api_url}/analyze",
                json={'text': comment},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Dosya yükleme ve analiz"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path, f, 'text/plain')}
                response = requests.post(
                    f"{self.api_url}/upload",
                    files=files,
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Sistem sağlık kontrolü"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_test_files(self) -> Dict[str, Any]:
        """Tüm test dosyalarını analiz et"""
        print("🔍 Test dosyaları analiz ediliyor...")
        
        test_files = [
            "ornek_yorumlar.txt",
            "test_yorumlar_detayli.txt"
        ]
        
        all_results = {}
        
        for file_path in test_files:
            try:
                print(f"📁 {file_path} analiz ediliyor...")
                result = self.analyze_file(file_path)
                
                if "error" not in result:
                    all_results[file_path] = result
                    print(f"✅ {file_path}: {result['yorum_sayisi']} yorum analiz edildi")
                else:
                    print(f"❌ {file_path}: {result['error']}")
                    
            except Exception as e:
                print(f"❌ {file_path} hatası: {e}")
        
        return all_results
    
    def calculate_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Sonuçlardan istatistik hesapla"""
        if not results or "sonuclar" not in results:
            return {"error": "Geçerli sonuç bulunamadı"}
        
        comments = results["sonuclar"]
        
        # Temel istatistikler
        total_comments = len(comments)
        sentiments = [c["analiz"] for c in comments]
        methods = [c.get("yöntem", "bilinmiyor") for c in comments]
        confidences = [c.get("güven", 0) for c in comments]
        
        # Sentiment dağılımı
        sentiment_counts = Counter(sentiments)
        sentiment_percentages = {
            sentiment: (count / total_comments) * 100 
            for sentiment, count in sentiment_counts.items()
        }
        
        # Yöntem dağılımı
        method_counts = Counter(methods)
        method_percentages = {
            method: (count / total_comments) * 100 
            for method, count in method_counts.items()
        }
        
        # Güven skoru istatistikleri
        valid_confidences = [c for c in confidences if c > 0]
        confidence_stats = {
            "ortalama": sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0,
            "minimum": min(valid_confidences) if valid_confidences else 0,
            "maksimum": max(valid_confidences) if valid_confidences else 0,
            "standart_sapma": self._calculate_std(valid_confidences) if valid_confidences else 0
        }
        
        # Model tutarlılığı
        model_consistency = self._analyze_model_consistency(comments)
        
        return {
            "genel": {
                "toplam_yorum": total_comments,
                "dosya_adi": results.get("dosya_adi", "bilinmiyor")
            },
            "sentiment_dagilimi": {
                "sayilar": dict(sentiment_counts),
                "yuzdeler": sentiment_percentages
            },
            "yontem_dagilimi": {
                "sayilar": dict(method_counts),
                "yuzdeler": method_percentages
            },
            "guven_skorlari": confidence_stats,
            "model_tutarliligi": model_consistency
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Standart sapma hesapla"""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _analyze_model_consistency(self, comments: List[Dict]) -> Dict[str, Any]:
        """Model tutarlılığını analiz et"""
        model_results = []
        
        for comment in comments:
            if "model_sonuçları" in comment and comment["model_sonuçları"]:
                model_data = comment["model_sonuçları"]
                if "consistency" in model_data:
                    model_results.append({
                        "tutarlı": model_data["consistency"],
                        "model": model_data.get("model_used", "bilinmiyor"),
                        "güven": model_data.get("final_confidence", 0)
                    })
        
        if not model_results:
            return {"toplam_model_analizi": 0, "tutarlılık_oranı": 0}
        
        total_model_analysis = len(model_results)
        consistent_count = sum(1 for r in model_results if r["tutarlı"])
        consistency_rate = (consistent_count / total_model_analysis) * 100
        
        # Model bazında tutarlılık
        model_consistency = {}
        for result in model_results:
            model = result["model"]
            if model not in model_consistency:
                model_consistency[model] = {"tutarlı": 0, "toplam": 0}
            
            model_consistency[model]["toplam"] += 1
            if result["tutarlı"]:
                model_consistency[model]["tutarlı"] += 1
        
        # Yüzdelik hesapla
        for model in model_consistency:
            total = model_consistency[model]["toplam"]
            consistent = model_consistency[model]["tutarlı"]
            model_consistency[model]["yüzde"] = (consistent / total) * 100
        
        return {
            "toplam_model_analizi": total_model_analysis,
            "tutarlılık_oranı": consistency_rate,
            "model_bazında": model_consistency
        }
    
    def generate_detailed_report(self) -> str:
        """Detaylı rapor oluştur"""
        print("📊 Detaylı istatistik raporu oluşturuluyor...")
        
        # Sistem sağlığı
        health = self.get_system_health()
        
        # Test dosyalarını analiz et
        test_results = self.analyze_test_files()
        
        # Rapor oluştur
        report = []
        report.append("🎭 TÜRKÇE DUYGU ANALİZİ SİSTEMİ - İSTATİSTİK RAPORU")
        report.append("=" * 60)
        report.append("")
        
        # Sistem durumu
        report.append("🏥 SİSTEM DURUMU")
        report.append("-" * 20)
        if "error" not in health:
            report.append(f"✅ Durum: {health.get('status', 'bilinmiyor')}")
            report.append(f"✅ Yüklenen Modeller: {len(health.get('pipelines', []))}")
            for model_id, model_info in health.get('models', {}).items():
                report.append(f"   - {model_id}: {model_info.get('name', 'bilinmiyor')}")
        else:
            report.append(f"❌ Sistem hatası: {health['error']}")
        report.append("")
        
        # Test sonuçları
        if test_results:
            report.append("📁 TEST DOSYALARI ANALİZİ")
            report.append("-" * 30)
            
            for file_path, result in test_results.items():
                if "error" not in result:
                    stats = self.calculate_statistics(result)
                    if "error" not in stats:
                        report.append(f"📄 {file_path}")
                        report.append(f"   Toplam Yorum: {stats['genel']['toplam_yorum']}")
                        
                        # Sentiment dağılımı
                        report.append("   Sentiment Dağılımı:")
                        for sentiment, count in stats['sentiment_dagilimi']['sayilar'].items():
                            percentage = stats['sentiment_dagilimi']['yuzdeler'][sentiment]
                            report.append(f"     {sentiment}: {count} ({percentage:.1f}%)")
                        
                        # Yöntem dağılımı
                        report.append("   Yöntem Dağılımı:")
                        for method, count in stats['yontem_dagilimi']['sayilar'].items():
                            percentage = stats['yontem_dagilimi']['yuzdeler'][method]
                            report.append(f"     {method}: {count} ({percentage:.1f}%)")
                        
                        # Güven skorları
                        conf_stats = stats['guven_skorlari']
                        report.append("   Güven Skorları:")
                        report.append(f"     Ortalama: {conf_stats['ortalama']:.3f}")
                        report.append(f"     Min-Max: {conf_stats['minimum']:.3f} - {conf_stats['maksimum']:.3f}")
                        report.append(f"     Standart Sapma: {conf_stats['standart_sapma']:.3f}")
                        
                        # Model tutarlılığı
                        if stats['model_tutarliligi']['toplam_model_analizi'] > 0:
                            report.append("   Model Tutarlılığı:")
                            report.append(f"     Genel Tutarlılık: {stats['model_tutarliligi']['tutarlılık_oranı']:.1f}%")
                            for model, model_stats in stats['model_tutarliligi']['model_bazında'].items():
                                report.append(f"     {model}: {model_stats['yüzde']:.1f}% tutarlı")
                        
                        report.append("")
                    else:
                        report.append(f"❌ {file_path}: {stats['error']}")
                else:
                    report.append(f"❌ {file_path}: {result['error']}")
        else:
            report.append("❌ Test dosyaları analiz edilemedi")
        
        # Genel özet
        if test_results:
            report.append("📈 GENEL ÖZET")
            report.append("-" * 20)
            
            total_comments = 0
            all_sentiments = []
            all_methods = []
            
            for result in test_results.values():
                if "error" not in result:
                    total_comments += result.get('yorum_sayisi', 0)
                    all_sentiments.extend([c['analiz'] for c in result.get('sonuclar', [])])
                    all_methods.extend([c.get('yöntem', 'bilinmiyor') for c in result.get('sonuclar', [])])
            
            if total_comments > 0:
                sentiment_summary = Counter(all_sentiments)
                method_summary = Counter(all_methods)
                
                report.append(f"📊 Toplam Analiz Edilen Yorum: {total_comments}")
                report.append("🎯 Genel Sentiment Dağılımı:")
                for sentiment, count in sentiment_summary.most_common():
                    percentage = (count / total_comments) * 100
                    report.append(f"   {sentiment}: {count} ({percentage:.1f}%)")
                
                report.append("🔧 Genel Yöntem Dağılımı:")
                for method, count in method_summary.most_common():
                    percentage = (count / total_comments) * 100
                    report.append(f"   {method}: {count} ({percentage:.1f}%)")
        
        report.append("")
        report.append("✨ Rapor tamamlandı!")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_report_to_file(self, filename: str = "sentiment_analysis_report.txt"):
        """Raporu dosyaya kaydet"""
        report = self.generate_detailed_report()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Rapor {filename} dosyasına kaydedildi")
            return True
        except Exception as e:
            print(f"❌ Rapor kaydedilemedi: {e}")
            return False

def main():
    """Ana fonksiyon"""
    print("🚀 Sentiment Analiz İstatistik Analizörü Başlatılıyor...")
    
    # Analizör oluştur
    analyzer = SentimentStatisticsAnalyzer()
    
    # Rapor oluştur
    report = analyzer.generate_detailed_report()
    
    # Ekrana yazdır
    print("\n" + report)
    
    # Dosyaya kaydet
    analyzer.save_report_to_file()
    
    print("\n🎉 İstatistik analizi tamamlandı!")

if __name__ == "__main__":
    main()
