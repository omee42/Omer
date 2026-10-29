"""Optional YouTube upload through the YouTube Data API v3 (standard library only).

Needs three secrets (see README): YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN.
Uploads the long video (and the Short), sets the thumbnail and uploads Turkish subtitles.
"""
import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
THUMB_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={vid}&uploadType=media"
CAPTION_URL = "https://www.googleapis.com/upload/youtube/v3/captions?uploadType=multipart&part=snippet"


def _access_token():
    data = urllib.parse.urlencode({
        "client_id": os.environ["YT_CLIENT_ID"],
        "client_secret": os.environ["YT_CLIENT_SECRET"],
        "refresh_token": os.environ["YT_REFRESH_TOKEN"],
        "grant_type": "refresh_token",
    }).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=data), timeout=60) as r:
            return json.loads(r.read())["access_token"]
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if "invalid_grant" in body:
            raise SystemExit("YouTube yenileme anahtarının (YT_REFRESH_TOKEN) süresi dolmuş ya da iptal edilmiş. "
                             "README'deki adımlarla yeni bir anahtar alıp GitHub gizli anahtarını güncelleyin.")
        raise SystemExit(f"YouTube erişim anahtarı alınamadı: {e.code} {body}")


def _publish_at(cfg, date):
    """Scheduled publish time (only used when the configured privacy is public)."""
    if cfg["youtube"].get("gizlilik", "private") != "public":
        return None
    v = cfg["youtube"].get("yayin_saati")
    if v in (None, ""):
        return None
    if isinstance(v, int):  # unquoted 17:00 in YAML 1.1 is read as 1020 (minutes)
        hh, mm = divmod(v, 60)
    else:
        hh, mm = (int(x) for x in str(v).strip().split(":"))
    local = dt.datetime.fromisoformat(date).replace(hour=hh, minute=mm)
    utc = local - dt.timedelta(hours=3)  # Türkiye saati UTC+3
    if utc <= dt.datetime.utcnow() + dt.timedelta(minutes=20):
        return None
    return utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def upload_video(token, video: Path, meta: dict, publish_at=None):
    status = {"privacyStatus": meta["privacyStatus"], "selfDeclaredMadeForKids": meta["madeForKids"],
              "embeddable": True}
    if publish_at:  # YouTube publishes a scheduled video only from private + publishAt
        status.update(privacyStatus="private", publishAt=publish_at)
    body = {
        "snippet": {"title": meta["title"], "description": meta["description"], "tags": meta["tags"],
                    "categoryId": meta["categoryId"], "defaultLanguage": "tr", "defaultAudioLanguage": "tr"},
        "status": status,
    }
    size = video.stat().st_size
    req = urllib.request.Request(UPLOAD_URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)})
    with urllib.request.urlopen(req, timeout=120) as r:
        location = r.headers["Location"]
    with open(video, "rb") as fh:
        req = urllib.request.Request(location, data=fh.read(), method="PUT", headers={
            "Authorization": f"Bearer {token}", "Content-Type": "video/mp4"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["id"]


def set_thumbnail(token, vid, path: Path):
    req = urllib.request.Request(THUMB_URL.format(vid=vid), data=path.read_bytes(), method="POST",
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "image/jpeg"})
    urllib.request.urlopen(req, timeout=120).read()


def upload_caption(token, vid, srt: Path):
    boundary = "pofi-caption-boundary"
    meta = json.dumps({"snippet": {"videoId": vid, "language": "tr", "name": "Türkçe", "isDraft": False}})
    body = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{meta}\r\n"
            f"--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n").encode() + srt.read_bytes() + \
        f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(CAPTION_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": f"multipart/related; boundary={boundary}"})
    urllib.request.urlopen(req, timeout=120).read()


def upload_dir(out_dir: Path, cfg: dict, date: str):
    """Upload the long video and the Short; skips whatever youtube_linkler.json says is already up."""
    token = _access_token()
    record_path = out_dir / "youtube_linkler.json"
    results = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}

    def save():
        record_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if "video" not in results:
        meta = json.loads((out_dir / "youtube.json").read_text(encoding="utf-8"))
        vid = upload_video(token, out_dir / "video.mp4", meta, _publish_at(cfg, date))
        results["video"] = f"https://youtu.be/{vid}"
        save()
        for step, fn in (("kapak", lambda: set_thumbnail(token, vid, out_dir / "kapak.jpg")),
                         ("altyazi", lambda: upload_caption(token, vid, out_dir / "altyazi.srt"))):
            try:
                fn()
            except Exception as e:  # thumbnails need a verified channel; keep the upload anyway
                print(f"[youtube] {step} yüklenemedi: {e}")
    if "shorts" not in results and (out_dir / "shorts.mp4").exists() and (out_dir / "shorts.json").exists():
        smeta = json.loads((out_dir / "shorts.json").read_text(encoding="utf-8"))
        sid = upload_video(token, out_dir / "shorts.mp4", smeta, _publish_at(cfg, date))
        results["shorts"] = f"https://youtube.com/shorts/{sid}"
        save()
    print("[youtube] yüklendi:", results)
    return results
