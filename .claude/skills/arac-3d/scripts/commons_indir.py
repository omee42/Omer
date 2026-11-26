#!/usr/bin/env python3
"""Wikimedia Commons dosyalarını lisans bilgisiyle indirir (üyelik gerekmez).
    python3 -I commons_indir.py <hedef_klasor> "Dosya adı 1.jpg" "Dosya adı 2.jpg" ...
Her dosyanın lisansı, yazarı ve sayfası <hedef>/lisanslar.json içine yazılır. 429 (hız sınırı) gelirse bekleyip
yeniden dener; istekler arasında 2 sn bekler. Görseller referans içindir, depoya eklenmez."""
import json, sys, urllib.parse, urllib.request, urllib.error, os, time
UA = "arac3d-reference-fetch/1.0 (scale-model reference; python-urllib)"


def al(url, deneme=10):
    bekle = 5
    for i in range(deneme):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == deneme - 1:
                raise
            ra = e.headers.get("Retry-After")
            time.sleep(int(ra) if ra and ra.isdigit() else bekle)
            bekle *= 2
dosyalar = sys.argv[2:]
out = sys.argv[1]
meta = {}
for ad in dosyalar:
    q = urllib.parse.urlencode({"action": "query", "titles": "File:" + ad, "prop": "imageinfo",
                                "iiprop": "url|size|extmetadata", "iiurlwidth": 1800, "format": "json"})
    time.sleep(2)
    d = json.loads(al("https://commons.wikimedia.org/w/api.php?" + q))
    sayfa = next(iter(d["query"]["pages"].values()))
    ii = sayfa["imageinfo"][0]
    url = ii.get("thumburl") or ii["url"]
    # küçük boy görseller thumb.wikimedia.org'dan gelir (ortamın izinli alan adlarında olmalı); özgün dosyalar
    # upload.wikimedia.org'da paylaşılan bulut IP'lerine sık sık 429 + Retry-After 600 döner
    em = ii.get("extmetadata", {})
    lis = em.get("LicenseShortName", {}).get("value", "?")
    yazar = em.get("Artist", {}).get("value", "?")
    hedef = os.path.join(out, ad.replace(" ", "_").replace("/", "_"))
    time.sleep(2)
    try:
        veri = al(url)
    except Exception as e:
        print("  HATA", url, e)
        continue
    with open(hedef, "wb") as f:
        f.write(veri)
    meta[ad] = {"lisans": lis, "yazar": yazar, "sayfa": "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(ad.replace(" ", "_")), "yerel": hedef}
    print(f"{lis:14s} {ad}")
json.dump(meta, open(os.path.join(out, "lisanslar.json"), "w"), indent=1, ensure_ascii=False)
