# Opel Astra J GTC (OPC) – 3D Baskı Maketi, 1:24

> **Durum:** Önceki STL dosyaları kaldırıldı; tahmini eğrilerle yapılan model gerçeğe yeterince yakın değildi.
> Yeni yöntem: `.claude/skills/arac-3d` yeteneği. Aracı gerçek teknik çizimlerden (blueprint) ölçerek kurar.
> Astra J GTC için hazır ayar dosyası `.claude/skills/arac-3d/ornek/astra_j_gtc.json`.
> Yan/üst/ön/arka çizimler yüklendiğinde yeni STL'ler bu yetenekle üretilecek.
> Aşağıdaki bölüm önceki `astra_gtc_olustur.py` sürümünü anlatır.


![Genel görünüm](onizleme.png)
![Detaylar](detaylar.png)

Astra J GTC (2011–2018, 3 kapı) OPC donanımlı. Gerçek aracın ölçüleriyle modellenip 1:24 ölçeğe küçültüldü.

| Ölçü | Gerçek araç | Maket (1:24) |
|---|---|---|
| Boy | 4466 mm | 185.2 mm |
| Genişlik (aynasız / aynalı) | 1840 / 2020 mm | 76.5 / 84.0 mm |
| Yükseklik | 1482 mm | ~61 mm |
| Aks mesafesi | 2695 mm | 112.3 mm |
| İz genişliği (ön / arka) | 1584 / 1588 mm | 66 mm |
| Lastik | 245/35 R20 (OPC) | Ø 28.3 mm |

## Dosyalar

| Dosya | Ne için |
|---|---|
| `astra_gtc_tek_parca.stl` | Tekerlekler gövdeye bağlı, tek parça (1.67 M üçgen) |
| `astra_gtc_govde.stl` | **Kit:** tekerleksiz gövde, alt yüzeyi düz, aks delikli |
| `astra_gtc_teker.stl` | **Kit:** tek tekerlek. **4 adet** basılır |
| `astra_gtc_olustur.py` | Tüm STL'leri üreten betik (ölçek ve çözünürlük ayarlanabilir) |

Tüm dosyalar su geçirmez (manifold) ve tek gövdedir; dilimleyicide onarım gerekmez.
Yüzey **0.10 mm** ızgarada çıkarıldı, sadeleştirmede en fazla 0.005 mm sapma var.

## Modeldeki detaylar

- **Ön:**
  - kanat şeklindeki krom çıta ve ortasında **Opel şimşekli rozet**
  - iki yatay çıtalı üst ızgara
  - bumerang farlar: gündüz farı çizgisi, iki projektör merceği, çamurluğa sarılan yan kısım
  - kaput ortasındaki sırt ve kaput kapağı çizgisi
  - plaka yeri
  - **bal peteği ağlı** büyük alt ızgara
  - petek ağlı köşe hava girişleri ve çevrelerindeki "bıçak" kanatları
  - splitter dudağı
- **Yan:**
  - GTC'ye özgü, kapının önünden arka çamurluğa yükselen **"bıçak" çizgisi**
  - omuz çizgisi
  - geniş arka çamurluk kabarıklıkları
  - OPC marşpiyel
  - kapı aralıkları ve kapı kolu
  - sağ arkada yakıt kapağı
  - ön kapı camı, B direği ve sivri biten arka yan cam
  - sinyal çizgili büyük aynalar
- **Üst:**
  - girintili ön ve arka camlar
  - park konumunda iki silecek
  - köpekbalığı anten
  - OPC tavan spoyleri
- **Arka:**
  - kanat grafikli, yana sarılan stoplar
  - aralarındaki krom şerit ve **Opel şimşekli rozet**
  - bagaj kapağı çizgisi ve plaka yeri
  - kanatçıklı difüzör
  - **çift trapez egzoz** çıkışı
  - reflektörler
- **Tekerlek:**
  - 20 inç, 5 çift kollu jant, bijonlar ve göbek kapağı
  - **delikli fren diski** ve büyük kaliper
  - diş kanalları olan lastik

Ön ve arka rozetlerde Opel şimşeği kabartma olarak işlendi: halka 0.4 mm, şimşek rozet zemininden 0.3 mm yüksek. Yazılar (ASTRA, GTC, OPC) eklenmedi.

> Opel adı ve şimşek logosu Opel Automobile GmbH / Stellantis'in tescilli markasıdır. Logolu maket kişisel kullanım ve kâr amacı gütmeyen sergileme içindir; satış veya ticari kullanım için marka sahibinden izin gerekir.

## Hangi sürümü basmalı?

**Kit önerilir.** Tekerlekler jant yüzü yukarı bakacak şekilde basılınca kollar, bijonlar ve disk delikleri çok daha net çıkar. Gövde de alt yüzeyi tablaya oturduğu için neredeyse desteksiz basılır.

### Kit montajı

1. Gövdeyi ve 4 tekerleği basın.
2. **1.75 mm filamentten 2 parça aks** kesin, her biri yaklaşık **62 mm**.
3. Aksı gövdedeki delikten geçirin (Ø 2.1 mm, aks serbest döner).
4. Tekerlekleri iki uca bastırın. Tekerlekteki kör delik Ø 1.8 mm olduğu için sıkı geçer; gevşekse bir damla japon yapıştırıcısı yeterli.

Tekerlekler döner.

## Baskı ayarları

### FDM (filamentli)
| Ayar | Gövde | Tekerlek |
|---|---|---|
| Nozul | **0.2 mm** (en iyi) / 0.4 mm | **0.2 mm** önerilir |
| Katman | 0.06–0.08 mm | 0.05–0.08 mm |
| Çevre | 3–4 | 3 |
| Doluluk | %15 gyroid | %30 |
| Hız | Dış duvar 25–30 mm/s | 20 mm/s |
| Destek | Ağaç destek, **yalnız tabladan**: aynaların altı ve spoyler ucu | Gerekmez |
| Yönelim | Alt yüzey tablada | Jant yüzü yukarı, arka yüz tablada |

- **0.4 mm nozulda:** bal peteği ağı, disk delikleri ve kapı aralıkları yumuşar ama görünür kalır. Bu detaylar için 0.2 mm nozul ya da reçine yazıcı fark yaratır.
- **"Arachne" duvar üretecini** (Cura/Prusa/Orca'da varsayılan) açık tutun; ince çıtaları daha iyi basar.
- **Tek parça sürüm:** tekerleklerin altındaki 0.25 mm düzlük tablaya oturur. Gövdenin altı tabladan ~4.7 mm yüksekte olduğundan **"yalnız tabladan" destek** gerekir.

### Reçine (MSLA) — en detaylı sonuç
- Katman 0.03–0.05 mm.
- Gövdeyi dilimleyicide **2 mm et kalınlığıyla oyun**, iç tarafa 2 adet Ø3 mm tahliye deliği açın.
- Arkası aşağıda, 30–45° eğik yerleştirip hafif destek kullanın.
- Tekerlekleri arka yüzleri tabla tarafında düz basabilirsiniz.

## Boyama önerisi
1. **Astar:** Gri astar, 400 ve 800 zımpara.
2. **Gövde ve tavan:** Gövde rengi; OPC için çoğunlukla mavi, beyaz ya da kırmızı. Tavan siyah da olabilir.
3. **Siyah kısımlar:** Cam girintileri parlak siyah. Bal peteği ızgaralar, köşe girişleri, difüzör ve B direği mat siyah.
4. **Krom kısımlar:** Kanat çıtası, rozet halkaları ve şimşekler, arka krom şerit ve egzoz uçları krom/gümüş kalem. Rozet zeminini siyaha boyayıp şimşeği ve halkayı kromla boyarsanız logo çok net okunur.
5. **Lambalar:** Farlar gümüş, üzerine şeffaf vernik. Stoplar şeffaf kırmızı.
6. **Tekerlekler:** Lastikler mat siyah, kaliperler kırmızı ya da sarı (OPC'de Brembo).
7. **Vernik:** En son parlak vernik.

## Yeniden üretmek / ölçek değiştirmek

```bash
pip install numpy scipy scikit-image trimesh manifold3d
python3 astra_gtc_olustur.py --h 0.10        # sergi kalitesi, 1:24 (~5 dk)
python3 astra_gtc_olustur.py                 # 0.15 mm ızgara, daha hızlı
python3 astra_gtc_olustur.py --olcek 18      # 1:18 (~248 mm)
```

Aks delikleri baskı ölçüsünde sabittir (1.75 mm filament), ölçek değişse de aynı kalır.
1:18'de aks boyu ≈ 62 × 24 / 18 ≈ 83 mm olur.

## Denetim süreci

Model iki tur boyunca üç ayrı denetçi ajan tarafından incelendi: ön yüz, yan profil ve arka/üst görünüş.
Her tur standart açılardan görüntü alındı. Ajanlar bunları gerçek Astra J GTC ile karşılaştırıp milimetre cinsinden düzeltmeler önerdi ve her düzeltmeyi ayrı bir kopyada deneyerek doğruladı.

- **Oranlar ve kabin:**
  - Genişlik 1836 mm'ye indirildi.
  - Kabin üstten bakınca daralıyor, arka çamurluk üstlerinde basamak oluşuyor.
  - C direği içe yatıyor.
  - Kapı ile cam arasında yaklaşık 50 mm'lik omuz basamağı var.
- **Yan:**
  - GTC'ye özgü, arkaya sivrilen yan cam.
  - Keskin kenarı üstte olan "bıçak" çizgisi.
  - Kapı kolu omuz çizgisinin üzerinde.
- **Ön:**
  - Kaputun altına giren büyük bumerang farlar.
  - İnce, sivri kaput sırtı.
  - Gülümseme şeklinde petek ızgara.
  - Bumerang şeklinde köşe hava girişleri.
  - Gerçek boyutta rozet halkası.
  - Düz üstlü, ince kollu aynalar.
- **Arka:**
  - Dik arka kapak ve küçük "ördek kuyruğu".
  - Daha yatık ve uzun arka cam.
  - Tavanın devamı gibi duran spoyler.
  - Kanat şeklinde stoplar ve gövdeyle uyumlu bagaj kapağı çizgisi.
  - OPC difüzörü ve köşe hava girişleri.

## Kaynak ve doğruluk

- **Kaynak:** Gövde oranları üretici verilerine dayanır: boy, genişlik, yükseklik, aks mesafesi, iz genişliği, yerden yükseklik ve lastik ölçüsü.
- **Yaklaşık kısımlar:** Teknik çizim (blueprint) indirilemedi. Bu yüzden şunlar aracın bilinen tasarımı ve referans fotoğrafı esas alınarak modellendi:
  - yüzey kıvrımları,
  - cam ve far çizgileri,
  - tampon şekilleri.
- **Daha doğru sonuç için:** Ölçü kontrolü yapmak isterseniz `astra_gtc_olustur.py` içindeki profil tabloları (`Z_ALT_UST`, `Z_KABIN`, `ON_PLAN` …) gerçek milimetre cinsindendir. Bir teknik çizimle karşılaştırıp düzeltmek kolaydır.
