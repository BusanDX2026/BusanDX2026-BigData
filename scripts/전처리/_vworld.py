# -*- coding: utf-8 -*-
"""브이월드 지오코딩 공용 헬퍼. 인증키는 코드에 두지 않음 → secrets/vworld_key.txt 또는 환경변수 VWORLD_KEY."""
import os, time, json, urllib.parse, urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

def get_key():
    k = os.environ.get("VWORLD_KEY")
    if k:
        return k.strip()
    f = _ROOT / "secrets" / "vworld_key.txt"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    raise RuntimeError("VWORLD 키 없음: 환경변수 VWORLD_KEY 또는 secrets/vworld_key.txt 필요")

BUSAN_BBOX = (128.70, 34.90, 129.40, 35.40)  # lon0, lat0, lon1, lat1

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "busan-flood-preproc/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))

def _in_busan(lon, lat):
    return BUSAN_BBOX[0] <= lon <= BUSAN_BBOX[2] and BUSAN_BBOX[1] <= lat <= BUSAN_BBOX[3]

def geocode_address(addr, key, kind="parcel"):
    """지번(parcel) 또는 도로명(road) 주소 → (lon, lat) or None"""
    q = urllib.parse.urlencode({
        "service": "address", "request": "getcoord", "version": "2.0",
        "crs": "epsg:4326", "type": kind, "address": addr,
        "format": "json", "errorformat": "json", "key": key,
    })
    try:
        d = _get("https://api.vworld.kr/req/address?" + q)
        if d.get("response", {}).get("status") == "OK":
            p = d["response"]["result"]["point"]
            lon, lat = float(p["x"]), float(p["y"])
            if _in_busan(lon, lat):
                return lon, lat
    except Exception:
        pass
    return None

def search_place(query, key, category=""):
    """지명/POI 검색 → (lon, lat) or None (부산 범위 내 첫 결과)"""
    params = {
        "service": "search", "request": "search", "version": "2.0",
        "crs": "EPSG:4326", "size": "10", "page": "1", "query": query,
        "type": "place", "format": "json", "errorformat": "json", "key": key,
    }
    if category:
        params["category"] = category
    try:
        d = _get("https://api.vworld.kr/req/search?" + urllib.parse.urlencode(params))
        res = d.get("response", {})
        if res.get("status") == "OK":
            for it in res.get("result", {}).get("items", []):
                p = it.get("point", {})
                lon, lat = float(p["x"]), float(p["y"])
                if _in_busan(lon, lat):
                    return lon, lat
    except Exception:
        pass
    return None

def polite():
    time.sleep(0.35)
