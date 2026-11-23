---
name: arac-3d
description: Bir aracın teknik çizimlerinden (blueprint: yan, üst, ön, arka görünüş) 3D yazıcı için ölçekli, detaylı ve su geçirmez STL maket üretir. Kullanıcı bir araba/araç maketi, "3D yazıcı için araba modeli", "blueprint'ten STL" ya da benzerini istediğinde kullan.
---

# Araç teknik çiziminden 3D baskı maketi

Bu yetenek, bir aracın dört dik görünüşünü (yan, üst, ön, arka) gerçek ölçülerle kalibre eder.
Gövdeyi bu siluetlerin kesişimi olarak kurar ve çizimdeki iç çizgileri yüzeye kazır: panel aralıkları, cam, far ve ızgara hatları.
Sonunda tek parça ve/veya kit (gövde + ayrı tekerlek) STL dosyaları üretir.
Ölçüler ve çizgiler doğrudan çizimden geldiği için model **uydurma değildir**.

Betik: `.claude/skills/arac-3d/scripts/arac3d.py`. Örnek ayar: `.claude/skills/arac-3d/ornek/astra_j_gtc.json`

## Gerekenler

```bash
pip install numpy scipy scikit-image pillow trimesh manifold3d matplotlib
```

## İş akışı

### 1. Çizimleri bul
Aracın **yan, üst, ön ve arka** görünüş çizimlerini topla. Arka görünüş isteğe bağlıdır; yoksa ön görünüş kullanılır.
- **Kaynaklar:** the-blueprints.com, getoutlines.com, carsized.com (yan/ön/arka siluet), drawingdatabase.com, smcars.net.
- **Ağ erişimi:** Ağ kapalıysa (proxy 403 veya "host not allowed") kullanıcıdan görselleri yüklemesini iste. İstersen bu siteleri ortamın izinli alan adlarına eklemesini de önerebilirsin.
- **Doğru kasa ve yıl:** Aynı modelin farklı kasalarını karıştırma; örneğin Astra H GTC ≠ Astra J GTC. Çizim başlığını, yılı ve kapı sayısını kontrol et.
- **Saklama:** Görselleri çalışma klasörüne (ör. scratchpad/arac-adi/) koy. Telif hakkı nedeniyle **depoya ekleme**.

### 2. Resmi ölçüleri topla
Ayara şu ölçüler girilir: boy, genişlik (aynasız), yükseklik, aks mesafesi, iz genişliği, yerden yükseklik, lastik ölçüsü.
- **Kaynak:** Üretici verisi ya da auto-data.net, ultimatespecs.
- **Lastik çapı:** jant(inç) × 25.4 + 2 × genişlik × oran. Örnek: 245/35 R20 = 508 + 171.5 = 679.5 mm.

### 3. Ayar dosyasını yaz
`ornek/astra_j_gtc.json` dosyasını kopyala. Her görünüş için:
- **`kutu`:** `[x0, y0, x1, y1]`, yani o görünüşün görseldeki piksel kırpma alanı. Tek görselde 4 görünüş varsa her biri için ayrı kutu ver. Görseli Read ile açıp koordinatları belirle.
- **Kutunun sınırları:** Kutu, aracı sıkıca çevrelemeli. Ölçü okları, yazılar ve zemin çizgisi dışarıda kalmalı.
- **`on`:** Yan ve üst görünüşte aracın önü hangi taraftaysa `"sol"` ya da `"sag"`.
- **`aynalar`:** Ayna merkezi `x, y, z` (gerçek mm). Aynalar siluetlerden otomatik ayıklanır ve buradaki parametrik şekille geri eklenir.
- **`bolgeler`** (isteğe bağlı): Tohum pikselinden başlayıp çizgilerle sınırlı bölgeyi oyar; camlar, farlar, ızgaralar için kullanılır. Alanlar:
  - `gorunum`
  - `px`: [sütun, satır], kırpılmış ve yönü düzeltilmiş görselde
  - `derinlik_mm`

### 4. Siluetleri kontrol et
```bash
python3 .claude/skills/arac-3d/scripts/arac3d.py siluet ayar.json
```
- **`cikti/siluetler.png`:** Mavi siluet aracı tam kaplamalı. Kırmızı iç çizgiler panel/cam/far hatları olmalı, yazı veya gölge taraması olmamalı.
- **Rapor:** Ölçülen yükseklik ve aks mesafesi resmi değerlerle ±%1 içinde olmalı. Değilse kırpma kutusunu, `on` yönünü veya `teker.on_aks_x` değerini düzelt.
- **Ayar düğmeleri:** `cizgi_esik`, `bos_esik`, `kapat_px`, `kenar_px`. Görünüş başına da verilebilir.

### 5. Modeli üret
```bash
python3 .claude/skills/arac-3d/scripts/arac3d.py model ayar.json --h 0.3 --parca tek   # hızlı deneme
python3 .claude/skills/arac-3d/scripts/arac3d.py model ayar.json --h 0.10             # sergi kalitesi
```
- **Ölçek:** `olcek` (24 → 1:24). `--h`, baskı ölçeğinde mm cinsinden ızgara aralığıdır.
- **0.10 mm:** 1:24'te yaklaşık 5 dk sürer. **Bellek:** iki üretimi aynı anda çalıştırma, bellek yetmez.
- **Çıktılar:** `cikti/<dosya_adi>_tek_parca.stl`, `_govde.stl`, `_teker.stl` (4 adet basılır, 1.75 mm filament aks).
- **Kalite kontrolü:** Hepsi su geçirmez olmalı; çıktıda `su geçirmez=True` yazar.

### 6. Doğrula ve denetlet
```bash
python3 .claude/skills/arac-3d/scripts/arac3d.py kontrol ayar.json
```
- **IoU ve bindirme:** Yan, üst, ön ve arka izdüşümlerin çizimle örtüşmesini (IoU) ve `cikti/kontrol.png` bindirmesini üretir. Kırmızı modelde eksik olanı, mavi fazla olanı gösterir.
- **Hedef:** Her görünüşte IoU ≥ 0.97.
- **Denetçi ajanlar:** Gerçekçilik için 2–3 denetçi ajan başlat; ön, yan ve arka ayrı kapsamlar olsun. Onlara `kontrol.png`'yi, modelin 3/4 görüntülerini ve çizimleri ver. Ayar değerleri (kırpma, eşikler, bölgeler, ayna konumu, `detay` parametreleri) için somut düzeltme iste.

## Detay ayarları (`detay`)

| Ayar | Varsayılan | Anlamı |
|---|---|---|
| `cizgi_derinlik_mm` | 0.35 | Kazınan çizgi derinliği (baskı mm) |
| `cizgi_genislik_mm` | 0.35 | Kazınan çizgi genişliği; FDM 0.4 nozulda 0.5 önerilir |
| `normal_esik` | 0.55 | Bir yüzeyin hangi görünüşten okunacağı için normal eşiği |
| `yuvarlama_mm` | 1.5 | Gövde kenarlarındaki yuvarlatma |

## Sınırlar
- **Gövde yaklaşımı:** Gövde, siluetlerin kesişimi ve ön/arka kesit biçimiyle oluşturulur.
  - Dış hatlar çizimle birebir örtüşür.
  - Görünüşler arasında kalan çift eğrilikli yüzeyler (çamurluk kabarıklığı, kaput sırtı) yaklaşıktır.
  - Bunlar için düzeltmeleri ayara ya da betiğe ekle.
- **Yazılar:** Rozet ve yazılar çizimde varsa çizgi olarak kazınır. Logolar marka sahibinindir; ticari kullanımda izin gerekir.
- **Çizim kalitesi:** Kalitesiz, perspektifli veya filigranlı çizimler siluet ve çizgileri bozar. Mümkün olan en temiz vektör çizimi kullan.
