import json
import os
from pathlib import Path
import lmstudio as lms

IMAGE = Path("data/raw/nota-sample2.jpeg")
MODEL = os.environ["LM_STUDIO_MODEL"]

image = lms.prepare_image(str(IMAGE))
model = lms.llm(MODEL)

chat = lms.Chat()
chat.add_user_message(
    "Baca nota pada gambar. "
    "Ekstrak merchant, tanggal, item, subtotal, pajak, dan total. "
    "Keluarkan HANYA JSON valid tanpa markdown atau penjelasan tambahan. "
    "Gunakan struktur berikut:\n"
    "{"
    '"merchant": "", '
    '"tanggal": "", '
    '"item": [], '
    '"subtotal": 0, '
    '"pajak": 0, '
    '"total": 0'
    "}\n"
    "Jika pajak tidak terlihat, isi 0. Jangan mengarang.",
    images=[image],
)

prediction = model.respond(chat)
response = prediction.content.strip()

# Menghapus markdown ```json ... ``` jika model masih menggunakannya
if response.startswith("```"):
    response = response.replace("```json", "").replace("```", "").strip()

# Mengambil bagian JSON dari response
start = response.find("{")
end = response.rfind("}")

if start == -1 or end == -1:
    print("Response model bukan JSON:")
    print(response)
    raise ValueError("Model tidak mengembalikan JSON yang valid.")

json_text = response[start:end + 1]

try:
    result = json.loads(json_text)
except json.JSONDecodeError:
    print("JSON dari model tidak valid:")
    print(response)
    raise

# Membuat folder reports jika belum ada
Path("reports").mkdir(parents=True, exist_ok=True)

# Menyimpan hasil ke file JSON
Path("reports/receipt.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

# Menampilkan hasil
print(json.dumps(result, indent=2, ensure_ascii=False))
