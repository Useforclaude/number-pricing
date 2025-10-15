# คู่มือเทรนบน Paperspace

เอกสารนี้อธิบายขั้นตอนการรัน workflow ทำนายราคาหมายเลขโทรศัพท์บน Paperspace Gradient โดยตั้งค่าได้ทั้งหมดผ่าน `number_pricing/config.py` หรือ environment variable เท่านั้น

## 1. เตรียมพื้นที่ทำงาน
- เปิด Paperspace Gradient Notebook ที่ใช้ Python เวอร์ชัน 3.9 ขึ้นไป พร้อม RAM อย่างน้อย 8 GB
- โคลนหรือดึงอัปเดตจาก repository ล่าสุดไปยังพื้นที่จัดเก็บ เช่น `/storage/number-ML`
- ตรวจสอบให้แน่ใจว่ามีไฟล์ `data/raw/numberdata.csv` หากยังไม่มีให้คัดลอกไฟล์ไปไว้ที่ `/storage/number-ML/data/raw/`

## 2. ตั้งค่า Environment (แนะนำให้ทำ)
กำหนด environment variable ก่อนเรียก Python เพื่อให้ config ตรวจจับสภาพแวดล้อมได้ถูกต้อง
```bash
export NUMBER_PRICING_ENV=paperspace
export NUMBER_PRICING_PROJECT_ROOT=/storage/number-ML
export NUMBER_PRICING_ARTIFACT_DIR=/storage/number-ML/number_pricing_artifacts
export NUMBER_PRICING_DATASET_FILENAME=numberdata.csv
```
ปรับพาธหรือชื่อไฟล์ตามตำแหน่งที่คุณจัดเก็บข้อมูลจริง

## 3. ติดตั้งไลบรารี
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
สั่งติดตั้งจากโฟลเดอร์รากของโปรเจ็กต์ (`/storage/number-ML`) เพื่อให้โมดูลภายในนำเข้าได้ถูกต้อง

## 4. เริ่มเทรนโมเดล
เรียกสคริปต์ฝั่งเทรนตามนี้
```bash
python -m number_pricing.scripts.train
```
กระบวนการจะทำงานดังนี้
1. โหลดและตรวจสอบข้อมูลดิบตามกฎใน config รวมถึงล้างข้อมูลให้เรียบร้อย
2. ทำ cross-validation แบบ stratified เมื่อข้อมูลอนุญาต
3. เทรนโมเดลสุดท้าย ประเมินผลบน hold-out และบันทึก artefact ทุกชนิด

ไฟล์ log อยู่ใน `number_pricing_artifacts/logs/number_pricing.log` (หรือพาธที่กำหนดไว้ใน config)

## 5. ตรวจสอบผลลัพธ์
ไฟล์สำคัญจะถูกเก็บไว้ใต้โฟลเดอร์ artefact
- `models/<artifact_name>` – ไฟล์โมเดลที่เทรนแล้ว (ดีฟอลต์คือ `hist_gradient_boosting_number_pricing.joblib`)
- `reports/training_metrics.json` – รวม metric จาก hold-out และ cross-validation
- `reports/cross_validation_metrics.json` – รายละเอียด metric ราย fold
- `reports/holdout_predictions.csv` – ค่าเปรียบเทียบจริงกับค่าคาดการณ์สำหรับ hold-out
- `reports/oof_predictions.csv` – ค่า out-of-fold (เปิดปิดได้ใน config)

## 6. ปรับแต่งผ่าน config
- ปรับฟีเจอร์ โมเดล หรือพารามิเตอร์ cross-validation โดยแก้ไข `number_pricing/config.py`
- หลีกเลี่ยงการไปแก้ไขค่าคงที่ในไฟล์อื่น ใช้ config หรือ environment variable เท่านั้น
- หลังแก้ไข config ให้รันคำสั่งเทรนซ้ำ ระบบจะสร้าง artefact ชุดใหม่ในตำแหน่งเดิมอัตโนมัติ

## 7. (ทางเลือก) รันพยากรณ์แบบ batch
เมื่อเทรนเสร็จแล้ว สามารถพยากรณ์เบอร์ใหม่ได้ เช่น
```bash
python -m number_pricing.scripts.predict --numbers 0812345678 0991234567
```
หรือใช้ไฟล์ CSV
```bash
python -m number_pricing.scripts.predict \
  --input-file /storage/number-ML/data/raw/new_numbers.csv \
  --output /storage/number-ML/number_pricing_artifacts/reports/new_predictions.csv
```
ตรวจสอบให้คอลัมน์ชื่อตรงกับ config (ค่าเริ่มต้นคือ `phone_number`)

## 8. ข้อควรระวัง
- สำรองโฟลเดอร์ `number_pricing_artifacts/` ทุกครั้ง เผื่อเครื่อง Paperspace ถูกรีเซ็ต
- หากผลลัพธ์ดีขึ้นควรคอมมิตไฟล์รายงาน JSON เพื่อเก็บสถิติ
- ห้ามเก็บข้อมูลลับใน repository ให้ใช้ environment variable หรือไฟล์ `.env` ที่ไม่อยู่ใน git แทน
