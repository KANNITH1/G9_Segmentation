# Coin Segmentation Workshop

Workshop กลุ่ม: แบ่งส่วนวัตถุ (เหรียญ) ออกจากพื้นหลัง — Object = Positive Class, Background = Negative Class

## ทีม

| งาน | ผู้รับผิดชอบ | ขั้นตอน |
|---|---|---|
| Dataset & Ground Truth | เมธา | 1–2 |
| Segmentation Algorithm & Morphology | กานต์นิธิ | 3–4 |
| Evaluation & Presentation | ญานพัทธ์ | 5–6 |

## โครงสร้างโปรเจกต์

```
coin-segmentation-workshop/
├── dataset/
│   ├── images/        <- ภาพเหรียญต้นฉบับ (.jpg/.png) — งานของเมธา
│   └── masks/          <- ground truth mask (พิกเซลเหรียญ=255, พื้นหลัง=0) — งานของเมธา
├── src/
│   └── coin_segmentation.py   <- pipeline: threshold + morphology — งานของกานต์นิธิ
├── output/
│   ├── masks/          <- predicted mask ที่สคริปต์สร้างขึ้น (ใช้ต่อในขั้นตอน evaluation)
│   └── compare/         <- ภาพเปรียบเทียบ before/after morphology
├── requirements.txt
└── README.md
```

## วิธีติดตั้งและรัน

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/coin_segmentation.py
```

ผลลัพธ์จะถูกบันทึกไว้ใน `output/masks/` (predicted mask) และ `output/compare/` (ภาพ before/after morphology)

## ขั้นตอนถัดไป (สำหรับญานพัทธ์ — Evaluation)

ใช้ `output/masks/*.png` เทียบกับ `dataset/masks/*.png` (ground truth) แบบ pixel-wise เพื่อคำนวณ:
- Confusion Matrix (TP / TN / FP / FN)
- ROC Curve + AUC (ใช้ grayscale/score map จาก `segment_otsu()` เป็น score แทน mask ที่ threshold แล้ว)
- Precision, Recall, F1, IoU

## หมายเหตุ

- ปรับพารามิเตอร์ threshold/morphology ได้ในไฟล์ `src/coin_segmentation.py` (ตัวแปร `MORPH_KERNEL_SIZE`, `MORPH_ITERATIONS`)
- ถ้าเหรียญมีสีต่างจากพื้นหลังชัดเจน ลองสลับไปใช้ `method="hsv"` แทน `method="otsu"`
