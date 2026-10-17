"""Weekly refresh: Delphi FluView -> docs/data.json (last 26 contiguous weeks per state)."""
import json, statistics
from datetime import datetime, timedelta, timezone
import requests

API = "https://api.delphi.cmu.edu/epidata/fluview/"
REGIONS = ["al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in",
  "ia","ks","ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv","nh",
  "nj","nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn","tx","ut",
  "vt","va","wa","wv","wi","wy","dc","pr","vi","gu","as","mp"]

def epiweek_to_date(ew):
    y, w = ew // 100, ew % 100
    jan4 = datetime(y, 1, 4)
    s = jan4 - timedelta(days=(jan4.weekday() + 1) % 7)
    pj = datetime(y - 1, 1, 1)
    leap = ((y - 1) % 4 == 0 and (y - 1) % 100 != 0) or ((y - 1) % 400 == 0)
    if pj.weekday() == 6 or (leap and pj.weekday() == 5):
        s += timedelta(weeks=1)
    return s + timedelta(weeks=w - 1)

def next_ew(ew):
    y, w = ew // 100, ew % 100
    jan1 = datetime(y, 1, 1)
    leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
    wiy = 53 if (jan1.weekday() == 6 or (leap and jan1.weekday() == 5)) else 52
    return (y + 1) * 100 + 1 if w >= wiy else ew + 1

def pick(rows):
    blocks, cur = [], [rows[0]] if rows else []
    for r in rows[1:]:
        if next_ew(cur[-1]["epiweek"]) == r["epiweek"]:
            cur.append(r)
        else:
            blocks.append(cur); cur = [r]
    if cur: blocks.append(cur)
    for b in reversed(blocks):
        if len(b) >= 26: return b[-26:]
    return max(blocks, key=len) if blocks else []

out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "source": "CDC FluView via CMU Delphi API",
       "states": {}, "warnings": []}
for reg in REGIONS:
    try:
        now = datetime.now(timezone.utc)
        start = now - timedelta(weeks=60)
        r = requests.get(API, params={"regions": reg,
            "epiweeks": f"{start.year}01-{now.year}52"}, timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get("result") != 1: raise ValueError(j.get("message"))
        raw = [d for d in j.get("epidata", [])
               if d.get("num_ili") is not None
               and (d.get("num_patients") or 0) > 0
               and (d.get("num_providers") or 0) > 0]
        if raw:
            med_pat = statistics.median([d["num_patients"] for d in raw])
            med_prov = statistics.median([d["num_providers"] for d in raw])
            raw = [d for d in raw
                   if d["num_patients"] >= 0.5 * med_pat
                   and d["num_providers"] >= 0.75 * med_prov]
        rows = [{"epiweek": int(d["epiweek"]),
                 "week_start": epiweek_to_date(int(d["epiweek"])).strftime("%Y-%m-%d"),
                 "ilitotal": int(d["num_ili"]),
                 "total_patients": d.get("num_patients"),
                 "percent_ili": d.get("ili")} for d in raw]
        rows.sort(key=lambda x: x["epiweek"])
        weeks = pick(rows)
        out["states"][reg.upper()] = weeks
        if len(weeks) < 26:
            out["warnings"].append(f"{reg.upper()}: only {len(weeks)} weeks in best block")
    except Exception as e:
        out["states"][reg.upper()] = []
        out["warnings"].append(f"{reg.upper()}: {e}")

with open("docs/data.json", "w") as f:
    json.dump(out, f, indent=2)
print("warnings:", out["warnings"])
