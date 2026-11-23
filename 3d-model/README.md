# Opel Tarzı Hatchback – 3D Baskı Dosyası

![Önizleme](onizleme.png)

Astra benzeri oranlarda, tek parça basılan bir hatchback maket.

| Dosya | Açıklama |
|---|---|
| `opel_tarzi_hatchback.stl` | Doğrudan dilimleyiciye (Cura, PrusaSlicer, Bambu Studio, OrcaSlicer) atılacak dosya |
| `araba_olustur.py` | STL'i üreten betik. Ölçüyü değiştirip yeniden üretmek için |
| `onizleme.png` | Dört açıdan görünüm |

## Model bilgileri

- **Boyut:** 98.8 × 46.4 × 37.5 mm (boy × en × yükseklik), yaklaşık **1:43 ölçek**
- **Hacim:** ~94 cm³ (dolulukla birlikte ~25–35 g filament)
- **Tek parça:** gövde, tekerlekler ve aynalar birleşik. Model su geçirmez (manifold), onarım gerektirmez.
- **Detaylar:** girintili camlar (ön, arka, 4 yan cam, A/B/C direkleri), ince köşeli farlar, yatay ızgara,
  alt hava girişi, stop lambaları, plaka yeri, kapı çizgileri, kapı kolları, yan omuz çizgisi,
  5 kollu jantlar, yan aynalar
- **Logo:** Marka logosu modele eklenmedi. Ön ızgaranın ortasında **boş yuvarlak bir rozet yeri** var.
  İsterseniz buraya bir çıkartma yapıştırabilir ya da küçük bir rozet takabilirsiniz.

## Önerilen baskı ayarları (FDM)

| Ayar | Değer |
|---|---|
| Malzeme | PLA (en kolayı) veya PETG |
| Nozul | 0.4 mm |
| Katman yüksekliği | 0.12–0.16 mm (detaylar için ince katman daha iyi) |
| Duvar / çevre | 3 çevre (1.2 mm) |
| Doluluk | %15, gyroid |
| Destek | **Açık, yalnızca baskı tablasına değen** (“touching buildplate”) |
| Destek tipi | Ağaç (tree / organic) önerilir, sökmesi daha kolay |
| Tabla yapışması | Brim 3–5 mm (tekerlekler küçük bir yüzeyle tabladan kalkar) |
| Yönelim | Tekerlekler tablada, araba normal duruşunda (dosyada zaten böyle) |

**Neden destek?** Araba tekerlekleri üzerinde durur. Gövdenin altı tabladan yaklaşık 5 mm yüksekte
ve düz olduğu için orada destek gerekir. Desteği yalnızca tabladan başlatırsanız camlara ve
çamurluk içlerine destek girmez.

**Reçine (SLA/MSLA) yazıcı için:** Arabayı arkası aşağı gelecek şekilde ~30–45° yatırın,
hafif destekler kullanın ve 0.05 mm katman seçin. Model içi dolu olduğu için reçine tasarrufu
isterseniz dilimleyicide “hollow” (oyma) yapıp 2 mm et kalınlığı ve tahliye delikleri ekleyin.

## Boyutu değiştirmek

En kolayı dilimleyicide ölçeklemektir. Örneğin %150 ≈ 148 mm boy, yaklaşık 1:29 ölçek.
%60'ın altına inmeyin, kapı çizgileri ve jant kolları kaybolur.

İsterseniz betikle yeniden de üretebilirsiniz:

```bash
pip install manifold3d trimesh numpy
python3 araba_olustur.py --olcek 1.5 --cikti buyuk_araba.stl
```

Betikteki ölçüler (`BOY`, `YARI_GEN`, `TEKER_R`, aks konumları, cam derinliği) dosyanın başında
toplu durur. Bunları değiştirip farklı oranlarda bir araba elde edebilirsiniz.

## Boyama ipuçları

1. Gri astar atın, ardından gövde rengini sprey veya fırçayla boyayın.
2. Cam girintilerini koyu gri veya siyah boyayın. Girintiler 1 mm derin, fırça taşmasın diye kenarları sınır görevi görür.
3. Farları gümüş, stop lambalarını kırmızı, lastikleri mat siyah boyayın.
4. Son kat olarak parlak vernik atın.
