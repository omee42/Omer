---
name: arac-3d
description: Gerçek bir aracın internetteki normal fotoğraflarından (serbest lisanslı) ve resmi ölçülerinden 3D yazıcı için ölçekli, çok detaylı, su geçirmez STL maket üretir (tek parça + kit). Kullanıcı bir araba/araç maketi, "3D yazıcı için araba modeli", "aracın STL'i", "şu arabayı bas" ya da benzerini istediğinde kullan.
---

# Fotoğraftan araç maketi (STL)

Amaç: bakan kişi "aracın aynısını basmışsınız" demeli. Model uydurma olmaz. Her çizgi, kavis ve detay aracın
gerçek fotoğraflarından ölçülür ve resmi ölçülere oturtulur. Su geçirmezlik gerekli ama yetmez; asıl ölçüt
detayın inceliği, çokluğu ve fotoğrafla birebir uyumdur.

Hazır örnekler (aynı yöntemle yapıldı, şablon olarak oku):
- `3d-model/astra_gtc_olustur.py`: Opel Astra J GTC OPC (3 kapı). En olgun örnek: kalibre fotoğraf ölçümleri,
  bıçak çizgisi, OPC detayları, logolar, Brembo fren. Çekirdeği kendi içinde taşır.
- `3d-model/astra_sedan_olustur.py`: Opel Astra J Sedan (4 kapı). Ortak çekirdeği (`scripts/arac_cekirdek.py`)
  kullanan temiz yapı. Yeni araç için bunu kopyalayarak başla.

## Araçlar (`.claude/skills/arac-3d/scripts/`)

| Betik | İşi |
|---|---|
| `arac_cekirdek.py` | Araçtan bağımsız çekirdek: SDF yardımcıları (Egri/PCHIP, smin/smax, sstep5, poligon/çizgi/kutu, YPoligon, petek), çizgi yazı tipi, `maskeli_yumusat`, `Blok`, çok çekirdekli `govde_mesh`, su geçirmez `sadelestir` + `yuzey_duzenle`, `kaydet`, `teker_mesh` |
| `commons_indir.py` | Wikimedia Commons'tan dosyaları lisans/yazar kaydıyla indirir (429'da Retry-After'a uyar) |
| `kamera.py` | Fotoğraf kalibrasyonu: 2B-3B nokta eşlerinden iğne deliği kamera (`fit`, `fit_sev`), modeli fotoğrafa bindirme (`bindir`), ızgaralı kırpma (`izgara`), pikselden 3B nokta (`geri3d`) |
| `gorsel.py` | Dik izdüşüm gölgeli + **zebra** (yansıma çizgisi) görüntü; `pafta` ile genel/detay önizleme |
| `bolge_ag.py` | Modelin bir bölgesini 5-20 dakikada ağa çevirir (tam üretim 1-2 saat); düzeltme denemeleri için |
| `arac3d.py` | (Eski, yedek) Yalnız temiz teknik çizim varsa: 4 görünüşten siluet kesişimi. Fotoğraf yöntemi her zaman önceliklidir |

Gerekenler: `pip install numpy scipy scikit-image pillow trimesh manifold3d matplotlib`

## Kurallar (kullanıcıyla anlaşma)

- **Kaynaklar yalnız ücretsiz ve yasal:** Wikimedia Commons (CC lisanslı), kullanıcının kendi yüklediği
  fotoğraflar, resmi ölçü sayfaları (auto-data.net, ultimatespecs, üretici).
- **Üyelik, giriş, ödeme yok.** Kişisel bilgi gönderme. Bir site bunları isterse dur, kullanıcıya söyle.
- **Siteleri kandırma.** Tarayıcı taklidi, engel aşma, sahte başlık kullanma. Bir kaynak hız sınırı koyarsa
  bekle (Retry-After) ya da kullanıcıdan fotoğrafları yüklemesini iste.
- **Fotoğraflar depoya girmez.** Scratchpad'de kalır; lisanslar `lisanslar.json` içinde tutulur.
- **Teknik çizim (blueprint) arama.** Kullanıcı normal fotoğraflardan çıkarılmasını istiyor.
- **Logolar marka sahibinindir.** Kişisel maket için kabartma olarak eklenir; ticari kullanımda izin gerekir.

## İş akışı

### 1. Doğru kasayı ve ölçüleri sabitle
- **Kasa:** Kuşak (Astra H ≠ J ≠ K), kasa tipi (3 kapı / 5 kapı / sedan / SW), makyaj öncesi/sonrası, donanım
  (jant, tampon, krom) ve yılı netleştir. Belirsizse en yaygın donanımı seç ve kullanıcıya söyle.
- **Ölçüler:** boy, genişlik (aynasız), yükseklik, aks mesafesi, ön/arka iz, lastik ölçüsü.
  Lastik çapı = jant(inç) × 25.4 + 2 × genişlik × oran (225/50 R17 = 431.8 + 225 = 656.8 mm).

### 2. Fotoğrafları topla (en az 10-15)
- **Gerekli açılar:** tam yan profil (en değerlisi: aks mesafesiyle ölçeklenir), ön 3/4, arka 3/4, tam ön, tam
  arka, alçak açı, yakın planlar (far, stop, ızgara, jant, ayna, kapı kolu).
- **Kaynak:** Commons kategorisi (ör. `Category:Opel Astra J`) API ile listelenir, `commons_indir.py` ile indirilir.
  Aynı aracın farklı renkleri iyidir: açık renk yüzey eğriliğini, koyu renk yansımayı gösterir.

### 3. Modeli kur (araca özgü betik)
Koordinatlar gerçek mm: x önden arkaya (ön tampon x≈0), +y sağ, z yukarı (yer z=0). Model yarım hacimde (y ≥ 0)
tanımlanır, sol yarı aynadır; tek tarafta olanlar (yakıt kapağı, sürücü kilidi, logo şimşeği) `asimetrik()`'te.

Sıra:
1. **Yan profil:** tavan/kaput/bagaj üst çizgisi, alt çizgi, cam hattı (DLO), kapı çizgileri, teker kemerleri.
   Tam yan fotoğrafı aks merkezleriyle ölçekleyip noktaları ızgaralı kırpmadan oku (`kamera.izgara`).
2. **Plan ve kesitler:** x'e göre yarı genişlik, yüksekliğe göre kesit biçimi (omuz, bel, eteğe dönüş),
   kabin daralması (tumblehome), ön/arka plan kavisi.
3. **Gövde SDF'i:** profillerden pürüzsüz alanlar; birleşimler `smin/smax` ve `sstep5` ile (C2 geçiş).
4. **Detaylar:** panel aralıkları (oluk), camlar (girinti + krom/siyah çerçeve), farlar, stoplar, ızgara,
   sis farları, plaka, rozet/logo, aynalar, kapı kolları, silecekler, anten, egzoz, jant + lastik yazısı + fren.
5. **Kamera kalibrasyonu:** her önemli fotoğraf için 8-15 nokta eşi (teker merkezleri, jant çemberi, kapı
   köşeleri, ayna, far uçları) ile kamera oturt, modeli bindir, farkları ölç, tabloları düzelt. Tekrarla.

### 4. Kalite döngüsü (her turda, her bölge için)
Kullanıcının GTC'de yakaladığı kusurlar ezbere bakınca kaçtı. Her turda şunları **yakından** yap:
- **Zebra:** `gorsel.py zebra` ile her bölgeyi 2-3 açıdan. Bantlar düzgün akmalı; halka, kıvrım, kopukluk = kusur.
- **Bağlantı noktaları:** kapı-çamurluk, ön-arka çamurluk/kapı geçişi, tampon-gövde, cam-direk birleşimleri.
  Panel aralığının iki yanı aynı yüzeyde olmalı (basamak yok).
- **Fotoğraf bindirme:** siluet ve çizgiler her kalibre fotoğrafta örtüşmeli; cam hatları (ön/arka kapı camları
  aynı hizada mı?), far ve sis farı biçimi, stop eğimi, bagaj/arka cam eğimi.
- **Kesitler:** şüpheli yerde polar/düz kesit al; düz eğik yüz, sivri sırt, ani eğim değişimi ara.
- **Bağımsız denetçi ajanlar:** 2-3 ajan (ön / yan / arka). Onlara gerçek fotoğrafları ve model görüntülerini
  ver; sonucunu değil, kodu ve görüntüyü ver. Somut, ölçülü kusur listesi iste. Kendi kanaatini doğrulatma.
- **Yerel yumuşatma** (`maskeli_yumusat`) son çare: şablon (±1.7a) keskin öğeye (kemer dudağı, stop kırımı, cam
  kenarı) değmemeli; panel aralığının iki yanı aynı ağırlıkta olmalı; geçiş bandı ≥ 4a. Aksi halde 1-4 mm kırışık.
  Uyguladıktan sonra yüzey kayma haritası çıkar ve tam ağda tekrar zebraya bak.

### 5. Üret
- **Komut:** `python3 <betik> --parca kit --h 0.07` ve `--parca tek --h 0.08` **ayrı** arka plan işleri olarak.
  Her biri ~1-1.5 saat (4 çekirdek). Arka plan süre sınırı 2 saattir; daha uzun sürecekse bölge sayısını azalt
  ya da ızgarayı büyüt.
- **Paralellik ve bellek:** `govde_mesh` tüm çekirdekleri kullanır (`ARAC_NP`). İki üretimi aynı anda çalıştırma.
- **Kontroller:** `su geçirmez=True`, ölçüler (boy, genişlik, yükseklik) resmi değerlere uyuyor mu.
- **Önizleme:** `gorsel.py pafta model.stl onizleme.png genel` ve `... detay`.
- **Büyük dosya:** 50 MB'tan büyük tek parça STL'i zip'le (`zip -9`). GitHub 100 MB'ı reddeder.

### 6. Teslim
- **Görseller:** önizlemeleri kullanıcıya gönder; düzeltilen yerlerin önce/sonra yakın planlarını da gönder.
- **Dürüstlük:** kalan kusurları açıkça yaz. "Hafif" deyip geçme; kullanıcı yakından bakıyor.
- **Depo:** STL'leri, betiği ve önizlemeleri commit/push et.

## GTC'den öğrenilenler (aynı hataları tekrarlama)
- **Arka çeyrek:** üst üste binen 8-9 düzeltme tablosu topak ve sivri omuz yaptı. Her bölgeyi tek, pürüzsüz bir
  tanımla kur; düzeltmeyi yeni tablo ekleyerek değil, var olanı düzelterek yap.
- **Cam hizası:** ön ve arka kapı camlarının alt/üst çizgileri fotoğraflarda tek akışta; ayrı ayrı uydurma.
- **Far ve sis farı:** biçimi yakın plandan, gerçek konturla al; simetrik varsayma (far iç ucu, dış sarma).
- **Logo:** Opel şimşeği nokta simetrik; ayna simetrisi bozar, `asimetrik()`'te işle.
- **Önizleme yanıltır:** uzak açıdan "iyi" görünen yüzey yakından topak çıkar. Önce zebra, sonra karar.
- **Tam üretim pahalı:** kararları `bolge_ag.py` ile bölge ağında ver, tam üretimi doğrulama için çalıştır.
