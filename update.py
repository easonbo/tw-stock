#!/usr/bin/env python3
"""
台股上市櫃 每日技術指標更新
- 清單：TWSE OpenAPI t187ap03_L（上市）＋ 公開資訊觀測站 t187ap03_O.csv（上櫃）
- 價量：Yahoo Finance（上市 .TW／上櫃 .TWO），透過 yfinance
- 籌碼：證交所 T86（上市三大法人）、MI_MARGN（上市融資融券）；
        櫃買中心 insti/dailyTrade（上櫃三大法人）、margin/balance（上櫃融資融券）
- 輸出：site/data/snapshot.json（每檔最新指標）＋ site/data/px/<代號>.json（近 180 日線圖資料）
- 歷史價量存在 cache/hist.pkl（由 GitHub Actions cache 保留），每天只補抓最近一個月
本機測試：TEST_MODE=1 python update.py（用模擬資料，不連網）
"""
import io, json, os, re, shutil, sys, time, pickle, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(line_buffering=True)          # 讓 GitHub Actions 即時顯示進度
START = time.time()
CHIP_TIME_LIMIT = 35 * 60                             # 籌碼回補最多用到開始後 35 分鐘（避免超過 60 分鐘上限）

ROOT  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
OUT   = os.path.join(ROOT, "site", "data")
PXDIR = os.path.join(OUT, "px")
TEST  = os.environ.get("TEST_MODE") == "1"
INTRADAY = os.environ.get("INTRADAY") == "1"          # 盤中模式：只更新股價，籌碼沿用快取
TZ    = ZoneInfo("Asia/Taipei")
KEEP_ROWS, CHART_ROWS, MIN_ROWS = 320, 250, 80

IND_MAP = {
  "01":"水泥工業","02":"食品工業","03":"塑膠工業","04":"紡織纖維","05":"電機機械",
  "06":"電器電纜","08":"玻璃陶瓷","09":"造紙工業","10":"鋼鐵工業","11":"橡膠工業",
  "12":"汽車工業","14":"建材營造","15":"航運業","16":"觀光餐旅","17":"金融保險",
  "18":"貿易百貨","20":"其他業","21":"化學工業","22":"生技醫療業","23":"油電燃氣業",
  "24":"半導體業","25":"電腦及週邊設備業","26":"光電業","27":"通信網路業",
  "28":"電子零組件業","29":"電子通路業","30":"資訊服務業","31":"其他電子業",
  "32":"文化創意業","33":"農業科技業","34":"電子商務",
  "35":"綠能環保","36":"數位雲端","37":"運動休閒","38":"居家生活",
}

def to_num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return np.nan

# --------------------------------------------------------------------------- 清單
SHARES_COL = "已發行普通股數或TDR原股發行股數"

def _clean(rows, market):
    out = []
    for row in rows:
        code, name, ind = row[:3]
        shares = to_num(row[3]) if len(row) > 3 else np.nan
        code = str(code).strip()
        if not re.fullmatch(r"[1-9]\d{3}", code):      # 只留一般股
            continue
        ind = str(ind).strip().zfill(2)
        out.append(dict(code=code, name=str(name).strip(), industry=IND_MAP.get(ind, ind), market=market,
                        shares=None if np.isnan(shares) or shares <= 0 else shares))
    return out

def fetch_tse():
    r = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", timeout=60)
    r.raise_for_status()
    return _clean([(x["公司代號"], x["公司簡稱"], x["產業別"], x.get(SHARES_COL)) for x in r.json()], "上市")

def fetch_otc():
    r = requests.get("https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv", timeout=60)
    r.raise_for_status()
    for enc in ("utf-8-sig", "cp950"):
        try:
            txt = r.content.decode(enc); break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(txt), dtype=str)
    sh = df[SHARES_COL] if SHARES_COL in df.columns else [None] * len(df)
    return _clean(list(zip(df["公司代號"], df["公司簡稱"], df["產業別"], sh)), "上櫃")

def get_list():
    path = os.path.join(CACHE, "listed.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    res = []
    for market, fn in (("上市", fetch_tse), ("上櫃", fetch_otc)):
        try:
            rows = fn()
            if len(rows) < 100:
                raise ValueError(f"只拿到 {len(rows)} 筆")
            print(f"{market}清單 {len(rows)} 檔")
        except Exception as e:
            rows = [x for x in old if x["market"] == market]
            print(f"⚠ {market}清單下載失敗（{e}），改用上次快取 {len(rows)} 檔")
        res += rows
    seen, out = set(), []
    for x in res:
        if x["code"] not in seen:
            seen.add(x["code"]); out.append(x)
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return out

# --------------------------------------------------------------------------- 價量
def _naive(df):
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df

def yf_download(tickers, **kw):
    import yfinance as yf
    res = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(chunk, group_by="ticker", auto_adjust=False, actions=False,
                                 threads=True, progress=False, **kw)
                break
            except Exception as e:
                print("  重試", attempt + 1, e); time.sleep(20 * (attempt + 1))
        if df is None or df.empty:
            continue
        for t in chunk:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            except KeyError:
                continue
            sub = sub[sub["Volume"] > 0]
            if len(sub):
                res[t] = _naive(sub.astype(float))
        print(f"  已下載 {min(i + 100, len(tickers))}/{len(tickers)}")
        time.sleep(2)
    return res

def fake_prices(listed):
    rng = np.random.default_rng(1)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=KEEP_ROWS)
    out = {}
    for k, s in enumerate(listed):
        n = KEEP_ROWS if k % 17 else 60
        c = 100 * np.cumprod(1 + rng.normal(0, .025, n)); o = c * (1 + rng.normal(0, .01, n))
        out[s["code"]] = pd.DataFrame({
            "Open": o, "High": np.maximum(o, c) * (1 + abs(rng.normal(0, .015, n))),
            "Low": np.minimum(o, c) * (1 - abs(rng.normal(0, .015, n))), "Close": c,
            "Volume": rng.uniform(5e5, 3e7, n).round()}, index=days[-n:])
    return out

def update_history(listed):
    path = os.path.join(CACHE, "hist.pkl")
    hist = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    if TEST:
        return fake_prices(listed)
    tick = {s["code"]: s["code"] + (".TWO" if s["market"] == "上櫃" else ".TW") for s in listed}
    have = [c for c in tick if c in hist and len(hist[c]) >= MIN_ROWS]
    need = [c for c in tick if c not in have]
    print(f"增量更新 {len(have)} 檔，完整下載 {len(need)} 檔")
    for codes, period in ((have, "5d" if INTRADAY else "1mo"), (need, "18mo")):
        if not codes:
            continue
        got = yf_download([tick[c] for c in codes], period=period)
        for c in codes:
            new = got.get(tick[c])
            if new is None:
                continue
            old = hist.get(c)
            df = new if old is None else pd.concat([old, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            hist[c] = df.tail(KEEP_ROWS)
    hist = {c: hist[c] for c in tick if c in hist}
    pickle.dump(hist, open(path, "wb"))
    return hist

# --------------------------------------------------------------------------- 籌碼
CHIP_DAYS = 245                      # 保留／回補最近幾個交易日（約 1 年）
MAX_CHIP_REQ = 300                   # 每次執行最多發幾個籌碼請求（每 3 秒 1 個，約 20 分鐘）
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36", "Accept": "application/json"}
TPEX_WWW = "https://www.tpex.org.tw/www/zh-tw"

HOST_FAILS = {}                      # 連線失敗計數：同一網站連續失敗 3 次就停止本次請求

def get_json(url, params=None, timeout=30):
    host = url.split("/")[2]
    if HOST_FAILS.get(host, 0) >= 3:
        return None
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                HOST_FAILS[host] = 0
                try:
                    return r.json()
                except ValueError:
                    return None                # 非 JSON（例如假日的說明頁）
            print(f"  {host} HTTP {r.status_code}")
        except Exception as e:
            print(f"  {host} 連線失敗：{e}")
        time.sleep(5)
    HOST_FAILS[host] = HOST_FAILS.get(host, 0) + 1
    if HOST_FAILS[host] >= 3:
        print(f"⚠ {host} 連續失敗，本次略過該網站的籌碼資料")
    return None

def _nz(x):
    return 0.0 if np.isnan(x) else x

def _same_day(resp_date, day):
    """回應日期與要求日期一致才接受（相容 20261001 / 115/10/01 / 2026/10/01）。"""
    d = re.sub(r"\D", "", str(resp_date or ""))
    if len(d) == 8:
        return d == day.strftime("%Y%m%d")
    if len(d) == 7:                                    # 民國年
        return int(d[:3]) + 1911 == day.year and d[3:] == day.strftime("%m%d")
    return True                                        # 沒有日期欄位就不檢查

def _idx(fields, name, fallback, nth=0):
    hits = [i for i, f in enumerate(fields) if str(f).strip() == name]
    return hits[nth] if len(hits) > nth else fallback

def _first_table(j):
    t = (j or {}).get("tables") or []
    return t[0] if t and isinstance(t[0], dict) else {}

def twse_inst(day):
    j = get_json("https://www.twse.com.tw/rwd/zh/fund/T86",
                 {"date": day.strftime("%Y%m%d"), "selectType": "ALLBUT0999", "response": "json"})
    if not j or j.get("stat") != "OK" or not j.get("data") or not _same_day(j.get("date"), day):
        return None
    f = j.get("fields") or []
    i_f1 = _idx(f, "外陸資買賣超股數(不含外資自營商)", 4); i_f2 = _idx(f, "外資自營商買賣超股數", 7)
    i_t = _idx(f, "投信買賣超股數", 10); i_d = _idx(f, "自營商買賣超股數", 11); i_s = _idx(f, "三大法人買賣超股數", 18)
    out = {}
    for r in j["data"]:
        if len(r) <= max(i_f1, i_f2, i_t, i_d, i_s):
            continue
        fv = _nz(to_num(r[i_f1])) + _nz(to_num(r[i_f2]))
        out[str(r[0]).strip()] = [fv / 1000, to_num(r[i_t]) / 1000, to_num(r[i_d]) / 1000, to_num(r[i_s]) / 1000]
    return out or None

def tpex_inst(day):
    j = get_json(f"{TPEX_WWW}/insti/dailyTrade",
                 {"type": "Daily", "sect": "EW", "date": day.strftime("%Y/%m/%d"), "response": "json"})
    rows = _first_table(j).get("data") or []
    if not rows or not _same_day((j or {}).get("date"), day):
        return None
    # 欄位：代號,名稱, 外資(不含自營)[2:5], 外資自營[5:8], 外資合計[8:11], 投信[11:14],
    #       自營自行[14:17], 自營避險[17:20], 自營合計[20:23], 三大法人合計[23]
    out = {str(r[0]).strip(): [to_num(r[10]) / 1000, to_num(r[13]) / 1000, to_num(r[22]) / 1000, to_num(r[23]) / 1000]
           for r in rows if len(r) > 23}
    return out or None

def twse_margin(day):
    j = get_json("https://www.twse.com.tw/exchangeReport/MI_MARGN",
                 {"date": day.strftime("%Y%m%d"), "selectType": "ALL", "response": "json"})
    if not j or j.get("stat") != "OK" or not _same_day(j.get("date"), day):
        return None
    tables = j.get("tables") or []
    if len(tables) < 2 or not tables[1].get("data"):
        return None
    f = tables[1].get("fields") or []
    # 欄位：代號,名稱, 融資(買進,賣出,現金償還,前日餘額,今日餘額,限額), 融券(買進,賣出,現券償還,前日餘額,今日餘額,限額), 資券互抵, 註記
    mp, mb = _idx(f, "前日餘額", 5, 0), _idx(f, "今日餘額", 6, 0)
    sp, sb = _idx(f, "前日餘額", 11, 1), _idx(f, "今日餘額", 12, 1)
    out = {str(r[0]).strip(): [to_num(r[mb]), to_num(r[mp]), to_num(r[sb]), to_num(r[sp])]
           for r in tables[1]["data"] if len(r) > max(mb, sb)}
    return out or None

def tpex_margin(day):
    j = get_json(f"{TPEX_WWW}/margin/balance", {"date": day.strftime("%Y/%m/%d"), "response": "json"})
    rows = _first_table(j).get("data") or []
    if not rows or not _same_day((j or {}).get("date"), day):
        return None
    # 欄位：代號,名稱,前資餘額,資買,資賣,現償,資餘額,…,前券餘額[10],券賣,券買,券償,券餘額[14],…
    out = {str(r[0]).strip(): [to_num(r[6]), to_num(r[2]), to_num(r[14]), to_num(r[10])]
           for r in rows if len(r) > 14}
    return out or None

CHIP_SOURCES = {"inst": {"tse": twse_inst, "otc": tpex_inst},
                "margin": {"tse": twse_margin, "otc": tpex_margin}}

def trading_days(hist):
    cnt = {}
    for df in hist.values():
        for d in df.index[-CHIP_DAYS - 5:]:
            cnt[d] = cnt.get(d, 0) + 1
    n = max(cnt.values()) if cnt else 0
    return sorted(d for d, c in cnt.items() if c >= 0.3 * n)[-CHIP_DAYS:]

def fake_chips(days, codes):
    rng = np.random.default_rng(2); ch = {"inst": {}, "margin": {}}
    bal = {c: rng.uniform(500, 20000) for c in codes}
    for d in days:
        k = d.strftime("%Y-%m-%d"); ch["inst"][k] = {}; ch["margin"][k] = {}
        for mkt in ("tse", "otc"):
            ch["inst"][k][mkt] = {c: list(rng.normal(0, 300, 3)) + [0] for c in codes}
            for v in ch["inst"][k][mkt].values():
                v[3] = sum(v[:3])
            m = {}
            for c in codes:
                prev = bal[c]; bal[c] = max(0, prev + rng.normal(0, 200))
                m[c] = [bal[c], prev, bal[c] * 0.1, prev * 0.1]
            ch["margin"][k][mkt] = m
        ch.setdefault("warrant", {})[k] = {"tse": {c: [rng.uniform(0, 5e6) * (8 if rng.random() < .05 else 1),
                                                       rng.uniform(0, 1e6)] for c in codes}}
    return ch

# ---- 權證：證交所 OpenAPI t187ap37_L（基本資料：權證→標的）、t187ap42_L（每日成交：最新一日）
OPENAPI = "https://openapi.twse.com.tw/v1"

def _iso(d):
    d = re.sub(r"\D", "", str(d or ""))
    if len(d) == 7:
        d = str(int(d[:3]) + 1911) + d[3:]
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else None

def warrant_map(listed):
    """權證代號 → (標的代號, 'C' 認購 / 'P' 認售)，每週更新一次。"""
    path = os.path.join(CACHE, "warrant_map.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 6 * 86400:
        return json.load(open(path, encoding="utf-8"))
    data = get_json(f"{OPENAPI}/opendata/t187ap37_L", timeout=180)
    if not isinstance(data, list) or not data:
        return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    by_name = {s["name"]: s["code"] for s in listed}
    out = {}
    for w in data:
        under = str(w.get("標的證券/指數", "")).strip()
        m = re.search(r"(?<!\d)(\d{4,6})(?!\d)", under)
        code = m.group(1) if m else by_name.get(re.sub(r"[\s\d]", "", under))
        if not code:
            continue
        kind = str(w.get("權證類型", "")) + str(w.get("類別", ""))
        out[str(w.get("權證代號", "")).strip()] = [code, "P" if ("售" in kind or "熊" in kind) else "C"]
    json.dump(out, open(path, "w", encoding="utf-8"))
    print(f"權證基本資料：{len(out)} 檔可對應到標的")
    return out

def warrant_today(wmap):
    """最新一日：{標的代號: [認購成交金額, 認售成交金額]}（元），回傳 (日期, 資料)。"""
    data = get_json(f"{OPENAPI}/opendata/t187ap42_L", timeout=60)
    if not isinstance(data, list) or not data:
        return None, {}
    day, agg = None, {}
    for w in data:
        info = wmap.get(str(w.get("權證代號", "")).strip())
        if not info:
            continue
        day = day or _iso(w.get("交易日期"))
        amt = to_num(w.get("成交金額"))
        if np.isnan(amt):
            continue
        a = agg.setdefault(info[0], [0.0, 0.0])
        a[0 if info[1] == "C" else 1] += amt
    return day, agg

def update_warrants(ch, listed):
    if TEST:
        return
    try:
        wmap = warrant_map(listed)
        day, agg = warrant_today(wmap) if wmap else (None, {})
        if day and agg:
            ch.setdefault("warrant", {})[day] = {"tse": agg}
            print(f"權證成交：{day}，{len(agg)} 檔標的有權證交易")
        else:
            print("⚠ 權證資料沒有更新")
    except Exception as e:
        print(f"⚠ 權證資料失敗：{e}")

def update_chips(days, codes):
    if TEST:
        return fake_chips(days, codes)
    path = os.path.join(CACHE, "chips.pkl")
    ch = pickle.load(open(path, "rb")) if os.path.exists(path) else {"inst": {}, "margin": {}}
    keep = {d.strftime("%Y-%m-%d") for d in days}
    budget, fails = MAX_CHIP_REQ, 0
    for kind in CHIP_SOURCES:
        ch[kind] = {k: v for k, v in ch[kind].items() if k in keep}
    ch["warrant"] = {k: v for k, v in ch.get("warrant", {}).items() if k in keep}
    for d in reversed(days):                          # 由新到舊回補，法人與資券交錯進行
        k = d.strftime("%Y-%m-%d")
        for kind, srcs in CHIP_SOURCES.items():
            slot = ch[kind].setdefault(k, {})
            for mkt, fn in srcs.items():
                if slot.get(mkt) or budget <= 0 or time.time() - START > CHIP_TIME_LIMIT:
                    continue
                budget -= 1
                got = fn(d)
                time.sleep(3)                         # 證交所限制請求頻率
                if got:
                    slot[mkt] = got
                else:
                    fails += 1
                used = MAX_CHIP_REQ - budget
                if used % 40 == 0:                    # 定期存檔＋顯示進度，中途被中斷也不會白跑
                    pickle.dump(ch, open(path, "wb"))
                    print(f"  籌碼回補進度：{used} 個請求，目前補到 {k}")
    pickle.dump(ch, open(path, "wb"))
    for kind in ch:
        have = sorted(k for k, v in ch[kind].items() if v.get("tse") or v.get("otc"))
        print(f"籌碼 {kind}: {len(have)} 天" + (f"（{have[0]} ~ {have[-1]}）" if have else ""))
    if fails:
        print(f"  其中 {fails} 個請求沒有資料（多半是尚未公布，下次執行會再補）")
    return ch

def merged(ch, kind):
    """{日期: {代號: [...]}}，上市上櫃合併。"""
    return {k: {**(v.get("tse") or {}), **(v.get("otc") or {})} for k, v in sorted(ch[kind].items())
            if v.get("tse") or v.get("otc")}

def chip_stats(code, inst, marg, last_price_date, shares=None, vol_lots=None, turnover=None, warr=None):
    s = {}
    idays = [k for k in inst if k <= last_price_date]
    rows = [inst[k].get(code) for k in idays]
    if idays and rows[-1] is not None:
        arr = np.array([r if r is not None else [np.nan] * 4 for r in rows], dtype=float)
        last = arr[-1]
        s.update(inst_date=idays[-1], f1=last[0], t1=last[1], d1=last[2], tot1=last[3])
        for n in (5, 20, 60, 120, 240):
            tail = arr[-n:]
            s[f"f{n}"], s[f"t{n}"], s[f"tot{n}"] = (float(np.nansum(tail[:, i])) for i in (0, 1, 3))
        def streak(col):
            k = 0; sign = 0
            for v in arr[::-1, col]:
                sg = 0 if np.isnan(v) or v == 0 else (1 if v > 0 else -1)
                if sg == 0 or (sign and sg != sign):
                    break
                sign = sg; k += 1
            return k * sign
        s["f_streak"], s["t_streak"] = streak(0), streak(1)
        if shares:                                    # 買賣超佔已發行股數（‰，千分比）
            lots = shares / 1000
            for key in ("f1", "t1", "f5", "t5", "f20", "t20", "tot5"):
                s[key + "_cap"] = s[key] / lots * 1000 if not np.isnan(s[key]) else np.nan
        if vol_lots:                                  # 三大法人買賣超佔當日成交量 %
            s["inst_vol_pct"] = last[3] / vol_lots * 100
        prev = arr[-10:-5]
        s["f5_prev"], s["t5_prev"] = ((float(np.nansum(prev[:, i])) if len(prev) else np.nan) for i in (0, 1))
    mdays = [k for k in marg if k <= last_price_date]
    mrows = [marg[k].get(code) for k in mdays]
    if mdays and mrows[-1] is not None:
        mb, mp, sb, sp = mrows[-1]
        s.update(margin_date=mdays[-1], m_bal=mb, m_chg=mb - mp, s_bal=sb, s_chg=sb - sp,
                 ms_ratio=(sb / mb * 100) if mb and mb > 0 else None)
        back = [r for r in mrows[-5:] if r is not None]
        s["m_chg5"] = mb - back[0][1] if back else None
    if warr:
        wdays = [k for k in warr if k <= last_price_date]
        if wdays:
            cur = warr[wdays[-1]].get(code)
            if cur is not None:
                s.update(w_date=wdays[-1], w_call=cur[0] / 1e4, w_put=cur[1] / 1e4,     # 萬元
                         w_pc=cur[1] / cur[0] if cur[0] > 0 else None,
                         w_ratio=cur[0] / turnover * 100 if turnover else None)
                hist = [warr[k].get(code, [0.0, 0.0])[0] for k in wdays[-11:-1]]
                if len(hist) >= 3 and np.mean(hist) > 0:
                    s["w_surge"] = cur[0] / np.mean(hist)
    return {k: (round(float(v), 2) if isinstance(v, (float, np.floating)) and not np.isnan(v)
                else (None if isinstance(v, float) else v)) for k, v in s.items()}

# --------------------------------------------------------------------------- 指標
def wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()

def tw_kd(h, l, c, n=9):
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    rsv = ((c - ll) / (hh - ll) * 100).replace([np.inf, -np.inf], np.nan).to_numpy()
    K = np.full(len(c), np.nan); D = K.copy(); k = d = 50.0
    for i, v in enumerate(rsv):
        if np.isnan(v):
            continue
        k = 2 / 3 * k + v / 3; d = 2 / 3 * d + k / 3
        K[i], D[i] = k, d
    return pd.Series(K, c.index), pd.Series(D, c.index)

def indicators(df):
    o, h, l, c, v = (df[x] for x in ("Open", "High", "Low", "Close", "Volume"))
    x = pd.DataFrame(index=df.index)
    for n in (5, 10, 20, 60, 120, 240):
        x[f"ma{n}"] = c.rolling(n).mean()
    d = c.diff()
    x["rsi"] = 100 - 100 / (1 + wilder(d.clip(lower=0), 14) / wilder(-d.clip(upper=0), 14))
    x["dif"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    x["macd"] = x["dif"].ewm(span=9, adjust=False).mean()
    x["mid"] = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
    x["up"], x["dn"] = x["mid"] + 2 * sd, x["mid"] - 2 * sd
    x["K"], x["D"] = tw_kd(h, l, c)
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    x["atr"] = wilder(tr, 14)
    upm, dnm = h.diff(), -l.diff()
    pdm = upm.where((upm > dnm) & (upm > 0), 0.0); mdm = dnm.where((dnm > upm) & (dnm > 0), 0.0)
    x["pdi"] = 100 * wilder(pdm, 14) / x["atr"]; x["mdi"] = 100 * wilder(mdm, 14) / x["atr"]
    x["adx"] = wilder(100 * (x["pdi"] - x["mdi"]).abs() / (x["pdi"] + x["mdi"]), 14)
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    x["wr"] = -100 * (hh - c) / (hh - ll)
    tp = (h + l + c) / 3
    md = tp.rolling(20).apply(lambda a: np.abs(a - a.mean()).mean(), raw=True)
    x["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * md)
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(14).sum(); neg = mf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    x["mfi"] = 100 - 100 / (1 + pos / neg)
    x["obv"] = (np.sign(d).fillna(0) * v).cumsum(); x["obv_ma20"] = x["obv"].rolling(20).mean()
    x["vma5"], x["vma20"] = v.rolling(5).mean(), v.rolling(20).mean()
    x["bias20"] = (c / x["ma20"] - 1) * 100
    x["bbw"] = (x["up"] - x["dn"]) / x["mid"] * 100
    x["bbw60"] = 4 * c.rolling(60).std(ddof=0) / x["ma60"] * 100            # 季線布林帶寬 %
    x["amp"] = (h - l) / pc * 100
    return x.replace([np.inf, -np.inf], np.nan)

def num(a):
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(a) else a

def gt(a, b):
    a, b = num(a), num(b)
    return None if a is None or b is None else a > b

def both(*xs):
    if any(x is False for x in xs): return False
    if any(x is None for x in xs): return None
    return True

def wslope(series, end=0, n=5):
    """近 n 日加權斜率：每日變化量以 1..n 加權（越近越重），end=5 表示往前推 5 天的窗口。"""
    a = series.to_numpy(dtype=float)
    stop = len(a) - end
    seg = a[stop - n - 1: stop]
    if len(seg) < n + 1 or np.isnan(seg).any():
        return None
    w = np.arange(1, n + 1)
    return float((np.diff(seg) * w).sum() / w.sum())

def snapshot(df, x):
    n = len(df); c = df["Close"]; L = lambda col, i=-1: num(x[col].iloc[i])
    C = lambda i=-1: num(c.iloc[i])
    close = C()
    mas = [L("ma5"), L("ma10"), L("ma20")]
    hist = x["dif"] - x["macd"]
    kgold = both(gt(L("D", -2), L("K", -2)), gt(L("K"), L("D")))
    s = dict(
        date=df.index[-1].strftime("%Y-%m-%d"), close=close,
        chg_pct=(close / C(-2) - 1) * 100, ret20=(close / C(-21) - 1) * 100,
        ret60=(close / C(-61) - 1) * 100,
        vol_lots=num(df["Volume"].iloc[-1]) / 1000,
        vol_ratio=num(df["Volume"].iloc[-1] / x["vma20"].iloc[-2]),
        rsi=L("rsi"), rsi_prev=L("rsi", -2), K=L("K"), D=L("D"), wr=L("wr"), cci=L("cci"),
        mfi=L("mfi"), bias20=L("bias20"), adx=L("adx"), pdi=L("pdi"), mdi=L("mdi"),
        atr_pct=num(L("atr") / close * 100) if L("atr") else None,
        bb_pos=num((close - x["dn"].iloc[-1]) / (x["up"].iloc[-1] - x["dn"].iloc[-1])),
        bbw=L("bbw"),
        ma_bull=both(gt(L("ma5"), L("ma20")), gt(L("ma20"), L("ma60"))),
        above_ma20=gt(close, L("ma20")), above_ma60=gt(close, L("ma60")),
        above_ma120=gt(close, L("ma120")), above_ma240=gt(close, L("ma240")),
        ma_tangle=None if None in mas else (max(mas) - min(mas)) / close < 0.02,
        kd_golden=kgold, kd_low_gold=both(kgold, gt(30, L("K"))),
        macd_golden=both(gt(L("macd", -2), L("dif", -2)), gt(L("dif"), L("macd"))),
        macd_pos=gt(L("dif"), L("macd")), dmi_bull=gt(L("pdi"), L("mdi")),
        obv_up=gt(L("obv"), L("obv_ma20")),
        breakout20=close > df["High"].iloc[-21:-1].max(),
        near_high52=close >= 0.95 * df["High"].tail(250).max(),
        bb_squeeze=gt(1.1 * np.nanmin(x["bbw"].tail(120)) + 1e-12, L("bbw")),
        bb_up_break=gt(close, x["up"].iloc[-1]), bb_dn_break=gt(x["dn"].iloc[-1], close),
        up3=bool((c.tail(4).diff().dropna() > 0).all()),
        vol_expand=gt(L("vma5"), L("vma20")),
        low60_dist=(close / df["Low"].tail(60).min() - 1) * 100,
        rsi_min10=num(x["rsi"].tail(10).min()), k_min5=num(x["K"].tail(5).min()),
        hist=num(hist.iloc[-1]), hist_prev=num(hist.iloc[-2]),
        red=close > num(df["Open"].iloc[-1]),
        reclaim_ma5=both(gt(L("ma5", -2), C(-2)), gt(close, L("ma5"))),
        trend_up=both(gt(L("ma20"), L("ma60")), gt(L("ma60"), L("ma60", -21))),
        amp20=num(x["amp"].tail(20).mean()),
        # ---- 均線／布林斜率（近 5 日加權，%／日；帶寬為百分點／日）
        ma5_s=_pct(wslope(x["ma5"]), close), ma10_s=_pct(wslope(x["ma10"]), close),
        ma20_s=_pct(wslope(x["ma20"]), close), ma60_s=_pct(wslope(x["ma60"]), close),
        ma20_s_prev=_pct(wslope(x["ma20"], end=5), close),
        bbw_s=wslope(x["bbw"]), bbw60_s=wslope(x["bbw60"]),
        # ---- 當沖：CDP 逆勢操作價位（以今日高低收推算下一個交易日）
        **cdp_levels(num(df["High"].iloc[-1]), num(df["Low"].iloc[-1]), close),
        day_pos=num((close - df["Low"].iloc[-1]) / (df["High"].iloc[-1] - df["Low"].iloc[-1]))
               if df["High"].iloc[-1] > df["Low"].iloc[-1] else 0.5,
        # ---- 做多支撐：10日線、月線、10日低點中，位於股價下方且最接近者
        **long_support(close, L("ma10"), L("ma20"), num(df["Low"].tail(10).min())),
        vma20_lots=num(x["vma20"].iloc[-1] / 1000),
        turnover20=num((c * df["Volume"]).tail(20).mean() / 1e8),
    )
    return {k: (round(v, 3) if isinstance(v, float) else (bool(v) if isinstance(v, (bool, np.bool_)) else v))
            for k, v in s.items()}

def _pct(v, close):
    return None if v is None or not close else v / close * 100

def cdp_levels(h, l, c):
    cdp = (h + l + 2 * c) / 4
    return dict(cdp=cdp, ah=cdp + (h - l), nh=2 * cdp - l, nl=2 * cdp - h, al=cdp - (h - l))

def long_support(close, ma10, ma20, low10):
    cands = [(v, name) for v, name in ((ma10, "10日線"), (ma20, "月線"), (low10, "10日低點"))
             if v is not None and v <= close]
    if not cands:
        return dict(sup=None, sup_name=None, sup_dist=None)
    v, name = max(cands)
    return dict(sup=v, sup_name=name, sup_dist=(close / v - 1) * 100)

CHART_COLS = ["ma5","ma10","ma20","ma60","ma120","ma240","up","dn","K","D","rsi","dif","macd",
              "pdi","mdi","adx","obv","obv_ma20","wr","cci","mfi","bias20"]

def arr(s, nd=2):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]

def chart_json(df, x, code=None, inst=None, marg=None):
    d, y = df.tail(CHART_ROWS), x.tail(CHART_ROWS)
    days = [i.strftime("%Y-%m-%d") for i in d.index]
    out = {"d": days,
           "o": arr(d["Open"]), "h": arr(d["High"]), "l": arr(d["Low"]), "c": arr(d["Close"]),
           "v": arr(d["Volume"] / 1000, 0)}
    for col in CHART_COLS:
        out[col] = arr(y[col] / 1000 if col.startswith("obv") else y[col])
    if inst:
        for j, key in enumerate(("fi", "ti", "di")):
            out[key] = [None if (inst.get(k) or {}).get(code) is None else round(inst[k][code][j]) for k in days]
    if marg:
        for j, key in ((0, "mb"), (2, "sb")):
            out[key] = [None if (marg.get(k) or {}).get(code) is None or np.isnan(marg[k][code][j])
                        else round(marg[k][code][j]) for k in days]
    return out

# --------------------------------------------------------------------------- main
def main():
    os.makedirs(CACHE, exist_ok=True)
    shutil.rmtree(PXDIR, ignore_errors=True); os.makedirs(PXDIR, exist_ok=True)
    if TEST:
        listed = [dict(code=str(1101 + i), name=f"測試{i}", industry=["半導體業", "航運業", "金融保險"][i % 3],
                       market="上市" if i % 2 else "上櫃", shares=float(np.random.default_rng(i).uniform(2e7, 3e9)))
                  for i in range(60)]
    else:
        listed = get_list()
    hist = update_history(listed)
    try:
        if INTRADAY:                                       # 盤中不抓籌碼（尚未公布），直接用快取
            cp = os.path.join(CACHE, "chips.pkl")
            ch = pickle.load(open(cp, "rb")) if os.path.exists(cp) else {"inst": {}, "margin": {}}
            print("盤中模式：只更新股價")
        else:
            ch = update_chips(trading_days(hist), [s["code"] for s in listed])
            update_warrants(ch, listed)
        if not TEST and not INTRADAY:
            pickle.dump(ch, open(os.path.join(CACHE, "chips.pkl"), "wb"))
        inst, marg = merged(ch, "inst"), merged(ch, "margin")
        warr = {k: v.get("tse") or {} for k, v in sorted(ch.get("warrant", {}).items())}
    except Exception as e:                                  # 籌碼失敗不影響技術面
        print(f"⚠ 籌碼資料更新失敗：{e}")
        inst, marg, warr = {}, {}, {}
    rows, errs = [], []
    for s in listed:
        df = hist.get(s["code"])
        if df is None or len(df) < MIN_ROWS:
            continue
        try:
            x = indicators(df)
            snap = snapshot(df, x)
            turnover = snap["close"] * snap["vol_lots"] * 1000 if snap["close"] and snap["vol_lots"] else None
            rows.append({**{k: v for k, v in s.items() if k != "shares"}, **snap,
                         **chip_stats(s["code"], inst, marg, snap["date"], s.get("shares"),
                                      snap["vol_lots"], turnover, warr)})
            json.dump(chart_json(df, x, s["code"], inst, marg), open(os.path.join(PXDIR, s["code"] + ".json"), "w"),
                      separators=(",", ":"))
        except Exception as e:
            errs.append(f"{s['code']}: {e}")
    if errs:
        print(f"⚠ 指標計算失敗 {len(errs)} 檔，例如 {errs[:3]}")
    if not rows:
        sys.exit("沒有任何股票算出指標，停止（避免把網站覆蓋成空的）")
    last = max(r["date"] for r in rows)
    meta = dict(updated=dt.datetime.now(TZ).strftime("%Y-%m-%d %H:%M"), last_date=last, intraday=INTRADAY,
                inst_date=max((r.get("inst_date") or "" for r in rows), default="") or None,
                margin_date=max((r.get("margin_date") or "" for r in rows), default="") or None,
                warrant_date=max((r.get("w_date") or "" for r in rows), default="") or None,
                n_list=len(listed), n_rows=len(rows), n_stale=sum(r["date"] < last for r in rows), rows=rows)
    json.dump(meta, open(os.path.join(OUT, "snapshot.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"完成：{len(rows)} 檔，資料日期 {last}")

if __name__ == "__main__":
    main()
