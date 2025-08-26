"""
Basit Türkçe duygu analizi aracı.
- Hugging Face Transformers kullanır (çok dilli/Türkçe bir model).
- Tekil veya birden çok yorumu analiz eder.
- Çıktı formatı:
    - Yorum: "..."
    - Analiz: Olumlu/Olumsuz/Nötr

Kullanım:
    python sentiment_tr.py               # etkileşimli mod (kullanıcıdan girdi alır)
    python sentiment_tr.py --demo        # örnek yorum listesi üzerinde demo çalıştırır
    python sentiment_tr.py --file path   # bir dosyadaki yorumları satır satır okur

Notlar:
- Varsayılan model: savasy/bert-base-turkish-sentiment-cased (3 sınıf: neg/neu/pos)
"""
from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# Varsayılan model ve etiket eşlemeleri
DEFAULT_MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"
LABEL_ID_TO_NAME = {0: "Olumsuz", 1: "Nötr", 2: "Olumlu"}


def load_pipeline(model_name: str = DEFAULT_MODEL_NAME) -> TextClassificationPipeline:
	"""Duygu analizi için inference pipeline'ı yükler.

	Model ilk çalıştırmada indirilecektir. Sonraki çalıştırmalarda cache'den yüklenir.
	"""
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSequenceClassification.from_pretrained(model_name)
	return TextClassificationPipeline(model=model, tokenizer=tokenizer, task="text-classification", top_k=None)


def map_label_to_tr(label_str: str) -> str:
	"""Model etiketini Türkçe 3 sınıfa dönüştürür."""
	ls = label_str.strip().lower()
	if ls.startswith("label_"):
		try:
			idx = int(ls.split("_")[-1])
			return LABEL_ID_TO_NAME.get(idx, "Nötr")
		except Exception:
			return "Nötr"
	# Bazı modeller 'negative/neutral/positive' döndürebilir
	if "neg" in ls:
		return "Olumsuz"
	if "neu" in ls or "neutral" in ls:
		return "Nötr"
	if "pos" in ls or "positive" in ls:
		return "Olumlu"
	# Yıldız temelli olursa
	if "1" in ls or "2" in ls:
		return "Olumsuz"
	if "3" in ls:
		return "Nötr"
	if "4" in ls or "5" in ls:
		return "Olumlu"
	return "Nötr"


def predict_sentiment(pipeline: TextClassificationPipeline, texts: Iterable[str]) -> List[Tuple[str, str]]:
	"""Metinler için duygu analizi yapar ve (metin, etiket) döner."""
	outputs = pipeline(list(texts), truncation=True)
	results: List[Tuple[str, str]] = []
	for text, out in zip(texts, outputs):
		if isinstance(out, list):
			best = max(out, key=lambda x: float(x.get("score", 0.0)))
			pred_label = map_label_to_tr(str(best.get("label", "")))
		else:
			pred_label = map_label_to_tr(str(out.get("label", "")))
		results.append((text, pred_label))
	return results


def print_results(pairs: List[Tuple[str, str]]) -> None:
	"""İstenen formatta sonuçları yazdırır."""
	for text, label in pairs:
		print(f'- Yorum: "{text}"')
		print(f'- Analiz: {label}')


def interactive_loop(pipeline: TextClassificationPipeline) -> None:
	"""Kullanıcıdan bir veya birden fazla yorum alıp analiz eder. Çıkmak için boş enter."""
	print("Çıkmak için boş satır bırakıp Enter'a basın. Birden çok yorum için satır satır girin.")
	while True:
		try:
			text = input("Yorum: ").strip()
		except (EOFError, KeyboardInterrupt):
			print()
			break
		if not text:
			break
		results = predict_sentiment(pipeline, [text])
		print_results(results)


def read_lines(path: str) -> List[str]:
	"""Dosyadan satır satır yorumları okur (boş satırları atlar)."""
	with open(path, "r", encoding="utf-8") as f:
		lines = [ln.strip() for ln in f.readlines()]
	return [ln for ln in lines if ln]


def main() -> None:
	parser = argparse.ArgumentParser(description="Türkçe duygu analizi (Olumlu/Olumsuz/Nötr)")
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model adı veya yol")
	parser.add_argument("--file", type=str, default=None, help="Yorumları içeren dosya yolu (satır bazında)")
	parser.add_argument("--demo", action="store_true", help="Örnek yorum listesi üzerinde demo çalıştır")
	args = parser.parse_args()

	pipe = load_pipeline(args.model)

	if args.demo:
		demo_texts = [
			"Ürün harika, beklediğimden daha iyi!",
			"Hiç memnun kalmadım, kargo çok geç geldi.",
			"Ne iyi ne kötü, idare eder.",
			"Fiyatına göre performansı fena değil.",
			"Berbat bir deneyimdi, tekrar almam.",
			"Müşteri hizmetleri hızlı ve yardımcıydı.",
		]
		print_results(predict_sentiment(pipe, demo_texts))
		return

	if args.file:
		texts = read_lines(args.file)
		print_results(predict_sentiment(pipe, texts))
		return

	interactive_loop(pipe)


if __name__ == "__main__":
	main()

