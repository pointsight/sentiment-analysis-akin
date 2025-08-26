#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Görsel Sentiment Analiz İstatistikleri
"""

import requests
import json
from collections import Counter
import time
from typing import Dict, List, Any

class VisualSentimentStatistics:
    def __init__(self, api_url: str = "http://127.0.0.1:8000"):
        self.api_url = api_url
        
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
    
    def generate_ascii_charts(self, results: Dict[str, Any]) -> str:
        """ASCII karakterlerle basit grafikler oluştur"""
        if not results or "sonuclar" not in results:
            return "❌ Geçerli sonuç bulunamadı"
        
        comments = results["sonuclar"]
        total_comments = len(comments)
        
        # Sentiment dağılımı
        sentiments = [c["analiz"] for c in comments]
        sentiment_counts = Counter(sentiments)
        
        # Yöntem dağılımı
        methods = [c.get("yöntem", "bilinmiyor") for c in comments]
        method_counts = Counter(methods)
        
        # Güven skorları
        confidences = [c.get("güven", 0) for c in comments if c.get("güven", 0) > 0]
        
        charts = []
        charts.append("📊 GÖRSEL İSTATİSTİKLER")
        charts.append("=" * 50)
        charts.append("")
        
        # 1. Sentiment Dağılımı Pasta Grafiği
        charts.append("🎯 SENTIMENT DAĞILIMI")
        charts.append("-" * 25)
        
        for sentiment, count in sentiment_counts.most_common():
            percentage = (count / total_comments) * 100
            bar_length = int((count / total_comments) * 30)  # 30 karakterlik bar
            bar = "█" * bar_length
            charts.append(f"{sentiment:15} | {bar} {count:2d} ({percentage:5.1f}%)")
        
        charts.append("")
        
        # 2. Yöntem Dağılımı
        charts.append("🔧 YÖNTEM DAĞILIMI")
        charts.append("-" * 25)
        
        for method, count in method_counts.most_common():
            percentage = (count / total_comments) * 100
            bar_length = int((count / total_comments) * 30)
            bar = "█" * bar_length
            charts.append(f"{method:15} | {bar} {count:2d} ({percentage:5.1f}%)")
        
        charts.append("")
        
        # 3. Güven Skoru Histogramı
        if confidences:
            charts.append("📈 GÜVEN SKORU DAĞILIMI")
            charts.append("-" * 25)
            
            # Güven skorlarını aralıklara böl
            ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
            range_counts = {}
            
            for low, high in ranges:
                count = sum(1 for c in confidences if low <= c < high)
                range_counts[f"{low:.1f}-{high:.1f}"] = count
            
            for range_name, count in range_counts.items():
                if count > 0:
                    percentage = (count / len(confidences)) * 100
                    bar_length = int((count / len(confidences)) * 30)
                    bar = "█" * bar_length
                    charts.append(f"{range_name:8} | {bar} {count:2d} ({percentage:5.1f}%)")
        
        charts.append("")
        
        # 4. Model Tutarlılığı
        charts.append("🤖 MODEL TUTARLILIĞI")
        charts.append("-" * 25)
        
        model_results = []
        for comment in comments:
            if "model_sonuçları" in comment and comment["model_sonuçları"]:
                model_data = comment["model_sonuçları"]
                if "consistency" in model_data:
                    model_results.append({
                        "tutarlı": model_data["consistency"],
                        "model": model_data.get("model_used", "bilinmiyor")
                    })
        
        if model_results:
            model_consistency = {}
            for result in model_results:
                model = result["model"]
                if model not in model_consistency:
                    model_consistency[model] = {"tutarlı": 0, "toplam": 0}
                
                model_consistency[model]["toplam"] += 1
                if result["tutarlı"]:
                    model_consistency[model]["tutarlı"] += 1
            
            for model, stats in model_consistency.items():
                if stats["toplam"] > 0:
                    percentage = (stats["tutarlı"] / stats["toplam"]) * 100
                    bar_length = int((stats["tutarlı"] / stats["toplam"]) * 30)
                    bar = "█" * bar_length
                    charts.append(f"{model:15} | {bar} {stats['tutarlı']:2d}/{stats['toplam']:2d} ({percentage:5.1f}%)")
        
        charts.append("")
        
        # 5. Performans Özeti
        charts.append("⚡ PERFORMANS ÖZETİ")
        charts.append("-" * 25)
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            
            charts.append(f"Ortalama Güven: {avg_confidence:.3f}")
            charts.append(f"En Düşük Güven: {min_confidence:.3f}")
            charts.append(f"En Yüksek Güven: {max_confidence:.3f}")
            
            # Güven skoru kategorileri
            high_confidence = sum(1 for c in confidences if c >= 0.9)
            medium_confidence = sum(1 for c in confidences if 0.7 <= c < 0.9)
            low_confidence = sum(1 for c in confidences if c < 0.7)
            
            charts.append("")
            charts.append("Güven Kategorileri:")
            charts.append(f"  Yüksek (≥0.9): {high_confidence:2d} ({high_confidence/len(confidences)*100:5.1f}%)")
            charts.append(f"  Orta (0.7-0.9): {medium_confidence:2d} ({medium_confidence/len(confidences)*100:5.1f}%)")
            charts.append(f"  Düşük (<0.7): {low_confidence:2d} ({low_confidence/len(confidences)*100:5.1f}%)")
        
        charts.append("")
        charts.append("=" * 50)
        
        return "\n".join(charts)
    
    def generate_comparison_chart(self, file1_results: Dict, file2_results: Dict) -> str:
        """İki dosya arasında karşılaştırma grafiği"""
        if not file1_results or not file2_results:
            return "❌ Karşılaştırma için yeterli veri yok"
        
        file1_name = file1_results.get("dosya_adi", "Dosya 1")
        file2_name = file2_results.get("dosya_adi", "Dosya 2")
        
        file1_sentiments = Counter([c["analiz"] for c in file1_results.get("sonuclar", [])])
        file2_sentiments = Counter([c["analiz"] for c in file2_results.get("sonuclar", [])])
        
        all_sentiments = set(file1_sentiments.keys()) | set(file2_sentiments.keys())
        
        comparison = []
        comparison.append("📊 DOSYA KARŞILAŞTIRMASI")
        comparison.append("=" * 50)
        comparison.append("")
        
        comparison.append(f"📄 {file1_name:30} | 📄 {file2_name}")
        comparison.append("-" * 50)
        
        for sentiment in sorted(all_sentiments):
            count1 = file1_sentiments.get(sentiment, 0)
            count2 = file2_sentiments.get(sentiment, 0)
            
            # Bar grafiği oluştur
            max_count = max(count1, count2)
            if max_count > 0:
                bar1_length = int((count1 / max_count) * 20)
                bar2_length = int((count2 / max_count) * 20)
                
                bar1 = "█" * bar1_length
                bar2 = "█" * bar2_length
                
                comparison.append(f"{sentiment:15} | {bar1:20} {count1:2d} | {bar2:20} {count2:2d}")
            else:
                comparison.append(f"{sentiment:15} | {'':20} {count1:2d} | {'':20} {count2:2d}")
        
        comparison.append("")
        comparison.append("=" * 50)
        
        return "\n".join(comparison)
    
    def generate_trend_analysis(self, results: Dict[str, Any]) -> str:
        """Trend analizi oluştur"""
        if not results or "sonuclar" not in results:
            return "❌ Trend analizi için yeterli veri yok"
        
        comments = results["sonuclar"]
        
        # Yorumları uzunluklarına göre analiz et
        length_sentiments = []
        for comment in comments:
            text = comment.get("yorum", "")
            sentiment = comment.get("analiz", "")
            confidence = comment.get("güven", 0)
            
            if text and sentiment and confidence > 0:
                length = len(text.split())  # Kelime sayısı
                length_sentiments.append({
                    "length": length,
                    "sentiment": sentiment,
                    "confidence": confidence
                })
        
        if not length_sentiments:
            return "❌ Trend analizi için yeterli veri yok"
        
        # Uzunluk kategorileri
        short_comments = [c for c in length_sentiments if c["length"] <= 10]
        medium_comments = [c for c in length_sentiments if 10 < c["length"] <= 25]
        long_comments = [c for c in length_sentiments if c["length"] > 25]
        
        trend = []
        trend.append("📈 TREND ANALİZİ")
        trend.append("=" * 50)
        trend.append("")
        
        # Uzunluk bazında sentiment dağılımı
        trend.append("📏 UZUNLUK BAZINDA SENTIMENT DAĞILIMI")
        trend.append("-" * 40)
        
        categories = [
            ("Kısa (≤10 kelime)", short_comments),
            ("Orta (11-25 kelime)", medium_comments),
            ("Uzun (>25 kelime)", long_comments)
        ]
        
        for category_name, category_comments in categories:
            if category_comments:
                trend.append(f"\n{category_name}:")
                sentiment_counts = Counter([c["sentiment"] for c in category_comments])
                total = len(category_comments)
                
                for sentiment, count in sentiment_counts.most_common():
                    percentage = (count / total) * 100
                    bar_length = int((count / total) * 20)
                    bar = "█" * bar_length
                    trend.append(f"  {sentiment:15} | {bar} {count:2d} ({percentage:5.1f}%)")
                
                # Ortalama güven
                avg_confidence = sum(c["confidence"] for c in category_comments) / len(category_comments)
                trend.append(f"  Ortalama Güven: {avg_confidence:.3f}")
        
        trend.append("")
        
        # Güven skoru trendi
        trend.append("🎯 GÜVEN SKORU TRENDİ")
        trend.append("-" * 40)
        
        # Güven skorlarını kategorilere ayır
        confidence_ranges = [
            (0.5, 0.6, "Düşük"),
            (0.6, 0.7, "Orta-Düşük"),
            (0.7, 0.8, "Orta"),
            (0.8, 0.9, "Orta-Yüksek"),
            (0.9, 1.0, "Yüksek")
        ]
        
        for low, high, label in confidence_ranges:
            count = sum(1 for c in length_sentiments if low <= c["confidence"] < high)
            if count > 0:
                percentage = (count / len(length_sentiments)) * 100
                bar_length = int((count / len(length_sentiments)) * 25)
                bar = "█" * bar_length
                trend.append(f"{label:15} | {bar} {count:2d} ({percentage:5.1f}%)")
        
        trend.append("")
        trend.append("=" * 50)
        
        return "\n".join(trend)
    
    def generate_full_visual_report(self) -> str:
        """Tam görsel rapor oluştur"""
        print("🎨 Görsel istatistik raporu oluşturuluyor...")
        
        # Test dosyalarını analiz et
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
        
        if not all_results:
            return "❌ Hiçbir dosya analiz edilemedi"
        
        # Rapor oluştur
        report = []
        report.append("🎨 GÖRSEL SENTIMENT ANALİZ İSTATİSTİKLERİ")
        report.append("=" * 60)
        report.append("")
        
        # Her dosya için görsel grafikler
        for file_path, result in all_results.items():
            report.append(f"📄 {file_path}")
            report.append("=" * (len(file_path) + 4))
            report.append("")
            
            charts = self.generate_ascii_charts(result)
            report.append(charts)
            report.append("")
        
        # Dosya karşılaştırması
        if len(all_results) >= 2:
            file_names = list(all_results.keys())
            comparison = self.generate_comparison_chart(
                all_results[file_names[0]], 
                all_results[file_names[1]]
            )
            report.append(comparison)
            report.append("")
        
        # Trend analizi (ilk dosya için)
        first_file = list(all_results.values())[0]
        trend = self.generate_trend_analysis(first_file)
        report.append(trend)
        
        return "\n".join(report)
    
    def save_visual_report(self, filename: str = "visual_statistics_report.txt"):
        """Görsel raporu dosyaya kaydet"""
        report = self.generate_full_visual_report()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Görsel rapor {filename} dosyasına kaydedildi")
            return True
        except Exception as e:
            print(f"❌ Görsel rapor kaydedilemedi: {e}")
            return False

def main():
    """Ana fonksiyon"""
    print("🎨 Görsel Sentiment Analiz İstatistikleri Başlatılıyor...")
    
    # Görsel analizör oluştur
    visual_analyzer = VisualSentimentStatistics()
    
    # Görsel rapor oluştur
    report = visual_analyzer.generate_full_visual_report()
    
    # Ekrana yazdır
    print("\n" + report)
    
    # Dosyaya kaydet
    visual_analyzer.save_visual_report()
    
    print("\n🎉 Görsel istatistik analizi tamamlandı!")

if __name__ == "__main__":
    main()
