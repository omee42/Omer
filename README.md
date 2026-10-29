# Pofi ile Öğreniyorum — Günlük Çocuk Eğitici Video Fabrikası

3–7 yaş çocuklar için **her gün farklı bir konuda** eğitici video üreten otomatik sistem.
Baykuş maskot **Pofi** konuşur, göz kırpar, el sallar; sahneler 3D emoji çizimleri, Türkçe seslendirme,
neşeli arka plan müziği, ses efektleri ve altyazılarla otomatik canlandırılır.

Her bölüm için üretilenler:

| Dosya | Ne işe yarar |
|---|---|
| `video.mp4` | 1920×1080, 2–4 dakikalık ana video (YouTube) |
| `shorts.mp4` | 1080×1920, ~30 saniyelik dikey Shorts (bilmece bölümü) |
| `kapak.jpg` | 1280×720 YouTube kapak görseli |
| `altyazi.srt` | Türkçe altyazı dosyası |
| `youtube.txt` | Kopyala-yapıştır başlık, açıklama (bölüm zaman damgalarıyla), etiketler |
| `shorts.txt` | Shorts başlık ve açıklaması |
| `ses_raporu.txt` | Seslendirmede net duyulmayabilecek cümlelerin listesi |

## Her gün nasıl çalışır?

1. `takvim.yaml` bir yıllık konu planıdır (Pazartesi sayılar, Salı harfler, Çarşamba hayvanlar,
   Perşembe bilim, Cuma renkler/sanat, Cumartesi değerler, Pazar günlük yaşam + özel günler).
2. Her günün senaryosu `bolumler/YYYY-AA-GG-konu.json` dosyasındadır.
3. **GitHub Actions** her sabah 06:17'de (Türkiye saati) o günün senaryosunu bulur, videoyu üretir ve
   **Releases** sayfasında `bolum-YYYY-AA-GG` adıyla yayınlar. O güne senaryo yoksa `bolumler/yedek/`
   klasöründeki her zaman geçerli bölümlerden biri kullanılır.
4. İsterseniz video YouTube'a otomatik yüklenir (aşağıya bakın); istemezseniz Releases sayfasından indirip
   kendiniz yüklersiniz.

### Videoyu YouTube'a elle yükleme (telefondan da olur)

1. GitHub'da depo → **Releases** → bugünün bölümü → `video.mp4`, `kapak.jpg`, `youtube.txt` dosyalarını indirin.
2. YouTube Studio → **Oluştur → Video yükle** → `video.mp4`.
3. Başlık, açıklama ve etiketleri `youtube.txt` dosyasından kopyalayın; kapak olarak `kapak.jpg` seçin.
4. **Kitle: “Evet, çocuklara özel”** seçin (yasal zorunluluk).
5. Altyazılar → `altyazi.srt` yükleyin (Türkçe).
6. Aynısını `shorts.mp4` ve `shorts.txt` ile Shorts için yapın.

### Elle üretim (isteğe bağlı)

Actions sekmesi → **Günlük video** → **Run workflow** → tarih girin (ör. `2026-11-05`). Birkaç dakika sonra
Releases sayfasında hazır olur.

## Otomatik YouTube yükleme (isteğe bağlı)

1. [Google Cloud Console](https://console.cloud.google.com/)'da bir proje açın ve **YouTube Data API v3**'ü etkinleştirin.
2. **OAuth izin ekranı** oluşturun (Harici). Yayın durumunu **"Üretimde" (In production)** yapın; "Test" durumunda
   alınan anahtarlar 7 günde geçersiz olur. Kendi kullanımınız için uygulamanın doğrulanması gerekmez.
3. **OAuth istemci kimliği** oluşturun: tür **Web uygulaması**, yetkili yönlendirme adresi
   `https://developers.google.com/oauthplayground`.
4. [OAuth 2.0 Playground](https://developers.google.com/oauthplayground/) → sağ üstteki ayardan "Use your own OAuth
   credentials" ile kendi kimliğinizi girin → `https://www.googleapis.com/auth/youtube.upload` ve
   `https://www.googleapis.com/auth/youtube.force-ssl` izinlerini seçip kanal hesabınızla onaylayın →
   **refresh token** değerini kopyalayın.
5. GitHub depo → Settings → Secrets and variables → Actions: `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN`.
6. `config.yaml` → `youtube.otomatik_yukle: true`. `gizlilik: "public"` seçerseniz video `yayin_saati`
   (varsayılan 17:00) saatinde yayına girer; `private`/`unlisted` seçerseniz o gizlilikle yüklenir ve siz
   YouTube Studio'dan kontrol edip yayınlarsınız (önerilen başlangıç).

Aynı gün iş akışı tekrar çalıştırılırsa yüklenen videolar `youtube_linkler.json` kaydı sayesinde ikinci kez yüklenmez.

> Not: Google'ın denetlemediği API projeleriyle yüklenen videolar YouTube tarafından **gizli** kilitlenebilir;
> herkese açık otomatik yayın için projenin YouTube API denetiminden geçmesi gerekir. O zamana kadar elle
> yükleme daha pratiktir.

## Yeni bölüm yazmak

Rehber: [`docs/SENARYO_REHBERI.md`](docs/SENARYO_REHBERI.md). Kısaca:

```bash
pip install -r requirements.txt          # ayrıca ffmpeg kurulu olmalı
python -m pofi setup                     # çizimler, ses ve kalite kontrol modeli (ilk sefer)
python -m pofi validate                  # tüm senaryoları kontrol et
python -m pofi preview bolumler/2026-10-30-sonbahar-yapraklari.json --hizli   # hızlı önizleme PNG
python -m pofi render  bolumler/2026-10-30-sonbahar-yapraklari.json           # tam üretim → cikti/
python -m pofi takvim --gun 30           # önümüzdeki 30 günün senaryo durumu
```

## Ayarlar (`config.yaml`)

- `kanal.ad`, `kanal.maskot` — kanal ve maskot adı (senaryolarda `{maskot}` bu isimle değişir).
- `ses.motor` — `piper:fettah` (varsayılan) veya `piper:fahrettin`. İnternetli ortamda `edge:tr-TR-EmelNeural`
  gibi bir Microsoft sesi denenebilir (lisans koşullarını kendiniz değerlendirin).
- `ses.hiz`, `ses.deneme` (her cümle kaç kez seslendirilip en nettinin seçileceği), `ses.sozluk` (okunuş düzeltmeleri).
- `video.altyazi_goster`, `video.muzik_seviyesi`, `shorts.aktif`, `youtube.*`.

## Lisanslar ve kaynaklar

- **Çizimler:** [Microsoft Fluent Emoji](https://github.com/microsoft/fluentui-emoji) 3D — MIT lisansı.
- **Yazı tipleri:** Baloo 2 ve Nunito — SIL Open Font License (`assets/fonts`).
- **Ses:** Piper Türkçe sesleri `fettah` ve `fahrettin` — CC0 (ticari kullanım serbest). `dfki` sesi ticari
  olmayan lisanslı olduğu için bilinçli olarak kullanılmaz.
- **Müzik ve efektler:** Program her bölüm için kendisi besteler (telif yok).
- **Maskot Pofi:** Bu projede kodla çizilmiştir.

## YouTube politikaları hakkında önemli not

- Çocuk videolarında **“çocuklara özel”** işaretini mutlaka açık tutun (COPPA). Sistem bunu otomatik ayarlar.
- YouTube, para kazanma (YPP) için **seri üretilmiş/tekrarlayan** içeriği kısıtlayabiliyor. Kanalın özgün
  kalması için senaryoları gözden geçirip kişisel dokunuşlar ekleyin, formatları çeşitlendirin; isterseniz
  seslendirmeyi kendi sesinizle yapın (senaryo metni `youtube.txt` ve JSON dosyasında hazır).
- Bilgilerin doğruluğu bir kez daha bir yetişkin tarafından kontrol edilmelidir; özellikle sağlık ve
  güvenlik konularında.
- GitHub, herkese açık depolarda 60 gün hiç değişiklik olmazsa zamanlanmış iş akışlarını durdurur. Depoya
  düzenli olarak yeni senaryo eklendikçe bu sorun olmaz; durursa Actions sekmesinden yeniden etkinleştirin.
