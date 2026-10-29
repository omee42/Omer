# Senaryo Yazım Rehberi

Her bölüm `bolumler/YYYY-AA-GG-konu.json` adlı tek bir JSON dosyasıdır. Video, ses, müzik,
altyazı, kapak görseli ve YouTube metni bu dosyadan otomatik üretilir.

Tam bir örnek: [`bolumler/2026-10-29-cumhuriyet-bayrami.json`](../bolumler/2026-10-29-cumhuriyet-bayrami.json)

## Hızlı kontrol komutları

```bash
python3 -m pofi validate bolumler/2026-10-30-sonbahar.json     # kurallara uyuyor mu?
python3 -m pofi preview  bolumler/2026-10-30-sonbahar.json --hizli   # her sahneden kare içeren PNG
python3 -m pofi render   bolumler/2026-10-30-sonbahar.json     # tam video + Shorts + kapak + YouTube metni
```

`preview --hizli` seslendirme yapmaz; süreleri tahmin eder ve `cikti/<id>/onizleme.png` dosyasını üretir.

## Hedef kitle ve üslup (en önemli kısım)

- **Yaş: 3–7.** Cümleler kısa olsun: ideal 4–10 kelime, en fazla ~14 kelime.
- **Bir bölüm = tek bir konu**, 3–6 yeni kavram. Az ama sağlam öğret.
- **Tekrar öğretir.** Yeni kelimeyi en az 2–3 kez farklı sahnelerde kullan (tanıt → örnekle → bilmece → tekrar).
- **Etkileşim kur:** “Sen de söyle!”, “Hadi birlikte sayalım!”, “Düşün bakalım!”, “Ekranda gösterebilir misin?”
- **Olumlu ve sıcak dil:** “Aferin!”, “Harika!”, “Çok güzel!”. Çocuğu asla küçümseme, korkutma, utandırma.
- Maskotun adı için `{maskot}` yaz (ayarlardan değiştirilebilir). Kanal adı için `{kanal}`.
- Çocuğa “sen” diye hitap et, sıcak ve neşeli ol.
- **Doğruluk:** Bilgiler kesin ve yaşa uygun olmalı. Emin olmadığın bilgiyi yazma. Basitleştir ama yanlış söyleme
  (ör. “Güneş bir yıldızdır” doğru; “Güneş ateş topudur” yanlış).
- **Güvenlik:** Tehlikeli davranış gösterme/özendirme yok (ateş, bıçak, priz, ilaç, yüksekten atlama...).
  Gerekirse güvenlik notu ekle (“büyüklerimizle birlikte”).
- **Kapsayıcılık:** Kız/erkek kalıp yargısı yok (ör. “doktor” her iki cinsiyet), farklı aileler ve çocuklar.
- **Yasak:** marka/ürün adı, reklam, “abone ol/beğen” çağrısı, korkutucu ya da şiddet içeren içerik, siyasi/
  tartışmalı içerik, din ayrımı. Dini bayramlarda kültürel ve kapsayıcı (paylaşma, aile ziyareti, tatlı) anlat.
- Ulusal günlerde (23 Nisan, 19 Mayıs, 29 Ekim, 30 Ağustos) saygılı ve coşkulu, **10 Kasım’da sakin ve saygılı**
  bir üslup kullan ve `"mood": "calm"` seç (hareket molası, konfeti, “yaşasın” yok).

## Seslendirme için yazım ipuçları

- Rakam yazabilirsin; otomatik okunur: `5` → “beş”, `1923'te` → “bin dokuz yüz yirmi üçte”, `2 + 3 = 5` olur.
- Sıra sayılarını yazıyla yaz: “birinci”, “üçüncü” (`3.` yazma).
- Kısaltma kullanma (`vb.`, `TBMM`, `kg` → açık yaz).
- Şapkalı harf (â, î, û) kullanma; düz yaz (rengarenk, hala).
- Ünlem ve soru işareti tonlamaya yardım eder; kullan.
- Uzun, iç içe cümle yerine iki kısa cümle yaz.
- Ünsüzle biten bir kelimeyi, ünlüyle başlayan bir kelimeyle sık sık art arda getirme; ses bazen birleştirir.
  Önemli anahtar kelimeyi cümle sonuna ya da tek başına koymak ses motorunda daha net duyulur.

## Dosya yapısı

```json
{
  "id": "2026-10-30-sonbahar",          // dosya adıyla aynı (uzantısız)
  "date": "2026-10-30",                 // yayın günü
  "category": "mevsimler",              // takvim.yaml'deki kategori (renk temasını da seçer)
  "title": "Sonbahar Geldi",            // ekranda görünen kısa başlık (en fazla 40 karakter)
  "mood": "happy",                      // müzik: happy | calm | energetic
  "youtube": {
    "title": "...",                     // en fazla 100 karakter, arama dostu, emoji olabilir
    "short_title": "...",               // Shorts başlığı (isteğe bağlı)
    "description": "...",               // 2-4 cümle; neler öğrenileceğini anlat
    "tags": ["sonbahar", "mevsimler"],  // 5-12 etiket
    "thumbnail": {"text": "SONBAHAR", "emoji": ["🍂", "🍁"]}   // kapak: en fazla 18 karakter, 1-3 görsel
  },
  "scenes": [ ... ]                     // ilk sahne intro, son sahne outro
}
```

İsteğe bağlı: `"theme"` (renk teması adı: şeker, gökyüzü, orman, gün batımı, lavanta, deniz, uzay, limon, bayram).

### Görseller

Bir görsel şu biçimlerden biri olabilir:

- **Emoji:** `"🍎"` — Microsoft Fluent 3D çizimleri kullanılır. `validate` olmayan emojiyi söyler.
- **Şekil:** `{"shape": "üçgen", "color": "#3EC1D3"}` — daire, kare, üçgen, dikdörtgen, yıldız, kalp, oval,
  baklava, beşgen, altıgen, yarım daire, hilal.
- **Bayrak:** `{"flag": "tr"}` — resmi oranlarla çizilmiş Türk bayrağı.
- **Sembol/Yazı:** `{"symbol": "A", "color": "#7C6CF2"}` — harf, rakam veya kısa işaret (en fazla 6 karakter).

Sahnelerde görsel `"emoji": "🍎"` ya da doğrudan `"shape": ...`, `"flag": ...`, `"symbol": ...` ya da
`"visual": {...}` olarak yazılabilir.

### Ortak alanlar

- `lines`: Sahnenin başında söylenecek cümleler (liste, en fazla 8).
- `after`: Sahnenin sonunda söylenecek cümleler (isteğe bağlı).
- `chapter`: YouTube bölüm başlığı (isteğe bağlı; intro/quiz/recap/outro gibi sahnelerde otomatik).

## Sahne türleri

| Tür | Ne işe yarar | Zorunlu alanlar |
|---|---|---|
| `intro` | Başlık kartı, maskot el sallar | `title`, `lines`; isteğe bağlı `subtitle`, `emoji` (1-5 görsel listesi) |
| `talk` | Maskot konuşur, 1-3 büyük görsel ve kısa yazı | `lines`; isteğe bağlı `emoji` (1-3), `text` (≤60) |
| `item` | Tek bir nesne/kavramı büyük tanıtır, kelime ekranda | görsel, `word` (≤16), `lines`; isteğe bağlı `color` (#RRGGBB) |
| `gallery` | Öğeler tek tek belirir, her biri kendi cümlesiyle | `items`: 1-6 × `{görsel, label (≤22), line}`; isteğe bağlı `title` |
| `recap` | Hepsi birden görünür, sırayla vurgulanır (tekrar) | `items`: 1-6 × `{görsel, label, line?}`; isteğe bağlı `title` |
| `count` | Nesneleri tek tek sayar (sayılar otomatik okunur) | görsel, `n` (1-20); isteğe bağlı `label`, `lines`, `after` |
| `letter` | Büyük/küçük harf ve o harfle başlayan kelimeler | `letter`, `words`: 1-4 × `{word, görsel, line}` |
| `compare` | İki şeyi yan yana karşılaştırır | `left`/`right`: `{görsel, label, line, scale?}`; isteğe bağlı `title` |
| `fact` | “Biliyor muydun?” bilgi kartı | görsel, `text` (≤90), `lines` |
| `quiz` | Soru + 2-4 seçenek + 3-2-1 geri sayım + doğru cevap kutlaması | `question` (≤50), `options`: `{görsel, label}`, `answer` (0'dan başlayan sıra); `lines` (soruyu sesli sor), `after` (cevabı açıkla) |
| `chant` | Tekerleme / ritimli tekrar: her satır ekranda büyük | `items`: 1-8 × `{line (≤60), görsel?}`; isteğe bağlı `title` |
| `story` | Sayfa sayfa minik hikâye | `panels`: 1-6 × `{emoji: [1-3 görsel], caption? (≤50), lines (1-4)}`; isteğe bağlı `title` |
| `math` | Toplama/çıkarma görsellerle (otomatik okunur: “İki artı üç eşittir beş!”) | görsel, `a`, `b` (0-10), `op` (`+`/`-`) |
| `action` | Hareket molası: zıpla/alkışla/dön + geri sayım | görsel, `text` (≤18, ör. “ZIPLA!”), `seconds` (2-12), `lines` |
| `outro` | Kapanış, maskot el sallar, sıradaki bölüm kartı | `lines`; isteğe bağlı `title`, `next`: `{text (≤30), emoji}` |

Notlar:
- `quiz` için `lines` yoksa soru okunur; `after` yoksa “Doğru cevap: …!” söylenir. Doğru cevabın yerini
  bölüm içinde değiştir (hep ilk seçenek olmasın).
- `count` sırasında sayılar otomatik okunur (“Bir! İki! ...”); `lines` içinde sayıları tekrar yazma.
- `math` sırasında işlem otomatik okunur; `lines` sadece giriş, `after` sadece övgü/açıklama olsun.
- `letter` kelimeleri o harfle başlamalı (validate kontrol eder).
- `gallery` öğe cümlesi (`line`) o öğeyi anlatmalı: öğe tam o cümle başlarken ekrana gelir.

## Önerilen bölüm akışı (yaklaşık 3–4 dakika)

1. `intro` — selam + bugünün konusu (2-3 cümle)
2. `talk` veya `item` — konuya giriş, merak uyandır
3. Öğretim sahneleri: `gallery` / `item` / `letter` / `count` / `compare` / `story` / `math` (3-5 sahne)
4. `fact` — ilginç ve doğru bir bilgi
5. `quiz` — 1-2 bilmece
6. `chant` veya `action` — tekrar ya da hareket molası
7. `recap` — bugün öğrendiklerimiz
8. `outro` — veda + yarınki konu (`next`)

Toplam 10–14 sahne, 35–60 cümle hedefle. Bölüm 2,5 dakikadan kısa ve 5 dakikadan uzun olmasın.

## YouTube metinleri

- Başlık: konu + “çocuklar için” gibi aranan ifadeler, merak uyandıran, abartısız. Büyük harf yağmuru yok.
- Açıklama: Bölümde neler öğrenileceğini 2-4 cümleyle anlat. Bölümler (chapters), yaş bilgisi ve emeği geçenler
  otomatik eklenir.
- Etiketler: Türkçe arama terimleri (ör. “renkleri öğreniyorum”, “okul öncesi sayılar”).
- “Abone ol”, “beğen” gibi çağrılar ve dış bağlantı yazma (çocuklara özel içerik kuralları).
