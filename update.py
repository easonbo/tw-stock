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
CHIP_TIME_LIMIT = 35 * 60                             # 籌碼回補最多用到開始後 35 分鐘
BT_TIME_LIMIT = 60 * 60                               # 開始後超過 60 分鐘就跳過回測，先把最新股價發佈出去

ROOT  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
OUT   = os.path.join(ROOT, "site", "data")
PXDIR = os.path.join(OUT, "px")
TEST  = os.environ.get("TEST_MODE") == "1"
INTRADAY = os.environ.get("INTRADAY") == "1"          # 盤中模式：只更新股價，籌碼沿用快取
TZ    = ZoneInfo("Asia/Taipei")
KEEP_ROWS, CHART_ROWS, MIN_ROWS = 1560, 250, 80   # 保留約 6 年日K（5 年回測＋指標暖身）

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

# --------------------------------------------------------------------------- 永豐 Shioaji 行情快照（選用）
# 在 GitHub Secrets 設定 SJ_API_KEY / SJ_SECRET_KEY 後啟用；只用來「讀行情」，不下單。
# 一次最多 500 檔快照，盤中與收盤後用它更新「今天」這根 K 棒；失敗時自動退回 Yahoo。
SJ_KEY, SJ_SECRET = os.environ.get("SJ_API_KEY", "").strip(), os.environ.get("SJ_SECRET_KEY", "").strip()
SJ_INDEX = {}

# --------------------------------------------------------------------------- 大盤指數
INDEX_DEF = [("TAIEX", "加權指數", "^TWII"), ("TPEX", "櫃買指數", "^TWOII")]

def fetch_indices():
    """回傳 [{key, name, close, chg, chg_pct, date, spark}]；永豐盤中快照優先，其餘用 Yahoo。"""
    out = []
    if TEST:
        rng = np.random.default_rng(7)
        for key, name, _ in INDEX_DEF:
            base = 22000 if key == "TAIEX" else 250
            sp = list(np.round(base * np.cumprod(1 + rng.normal(0, .01, 30)), 2))
            out.append(dict(key=key, name=name, close=sp[-1], chg=round(sp[-1] - sp[-2], 2),
                            chg_pct=round((sp[-1] / sp[-2] - 1) * 100, 2), date=dt.date.today().isoformat(), spark=sp))
        return out
    try:
        import yfinance as yf
        df = yf.download([t for _, _, t in INDEX_DEF], period="3mo", group_by="ticker", auto_adjust=False, progress=False)
    except Exception as e:
        print(f"⚠ 指數下載失敗：{e}"); df = None
    for key, name, t in INDEX_DEF:
        try:
            c = _naive(df[t][["Close"]].dropna())["Close"] if df is not None else pd.Series(dtype=float)
        except Exception:
            c = pd.Series(dtype=float)
        item = dict(key=key, name=name, close=None, chg=None, chg_pct=None, date=None, spark=[round(float(v), 2) for v in c.tail(30)])
        if len(c) >= 2:
            item.update(close=round(float(c.iloc[-1]), 2), chg=round(float(c.iloc[-1] - c.iloc[-2]), 2),
                        chg_pct=round(float(c.iloc[-1] / c.iloc[-2] - 1) * 100, 2), date=c.index[-1].strftime("%Y-%m-%d"))
        sj = SJ_INDEX.get(key)
        if sj:                                                    # 永豐即時快照覆蓋最新值
            item.update(close=round(sj["close"], 2), chg=round(sj["chg"], 2), chg_pct=round(sj["chg_pct"], 2),
                        date=dt.datetime.now(TZ).date().isoformat(), time=sj["time"], source="永豐")
            if item["spark"]:
                item["spark"][-1] = item["close"]
        out.append(item)
    return out

def sj_today_bars(codes):
    """回傳 {代號: 今天的 OHLCV DataFrame（1 列）}；沒有金鑰、登入失敗或今天沒交易時回傳空 dict。"""
    if not (SJ_KEY and SJ_SECRET) or TEST:
        return {}
    try:
        import shioaji as sj
    except Exception as e:
        print(f"⚠ 未安裝 shioaji：{e}"); return {}
    api = sj.Shioaji(simulation=False)
    try:
        api.login(api_key=SJ_KEY, secret_key=SJ_SECRET)
    except Exception as e:
        print(f"⚠ 永豐登入失敗，改用 Yahoo：{e}"); return {}
    out, today = {}, pd.Timestamp(dt.datetime.now(TZ).date())
    try:
        contracts = []
        for c in codes:
            try:
                k = api.Contracts.Stocks[c]
                if k is not None:
                    contracts.append(k)
            except Exception:
                pass
        for i in range(0, len(contracts), 500):                  # 官方限制：每次最多 500 檔
            for sn in api.snapshots(contracts[i:i + 500]):
                day = pd.Timestamp(sn.ts).normalize()             # ts 為台灣當地時間
                if day != today or not sn.total_volume or not sn.open:
                    continue
                out[sn.code] = pd.DataFrame(
                    {"Open": [float(sn.open)], "High": [float(sn.high)], "Low": [float(sn.low)],
                     "Close": [float(sn.close)], "Volume": [float(sn.total_volume) * 1000]},   # total_volume 單位為張
                    index=pd.DatetimeIndex([today]))
            time.sleep(1)
        print(f"永豐行情：取得 {len(out)} 檔今日快照（共 {len(contracts)} 檔合約）")
        try:                                                      # 加權指數（TSE 001）、櫃買指數（OTC 101）
            idx = [api.Contracts.Indexs.TSE["001"], api.Contracts.Indexs.OTC["101"]]
            for key, sn in zip(("TAIEX", "TPEX"), api.snapshots([k for k in idx if k is not None])):
                if pd.Timestamp(sn.ts).normalize() == today and sn.close:
                    SJ_INDEX[key] = dict(close=float(sn.close), chg=float(sn.change_price), chg_pct=float(sn.change_rate),
                                         time=pd.Timestamp(sn.ts).strftime("%H:%M"))
        except Exception as e:
            print(f"  永豐指數快照略過：{e}")
    except Exception as e:
        print(f"⚠ 永豐快照失敗，改用 Yahoo：{e}")
    finally:
        try:
            api.logout()
        except Exception:
            pass
    return out

def update_history(listed):
    path = os.path.join(CACHE, "hist.pkl")
    hist = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    if TEST:
        return fake_prices(listed)
    tick = {s["code"]: s["code"] + (".TWO" if s["market"] == "上櫃" else ".TW") for s in listed}
    marker = os.path.join(CACHE, "hist_5y.done")         # 第一次升級到 5 年歷史時，全部重抓 7 年
    if not os.path.exists(marker) and not INTRADAY:
        hist = {}
    have = [c for c in tick if c in hist and len(hist[c]) >= MIN_ROWS]
    need = [c for c in tick if c not in have]
    print(f"增量更新 {len(have)} 檔，完整下載 {len(need)} 檔")
    sj_bars = sj_today_bars(list(tick))
    jobs = [(need, "7y")]
    if not (INTRADAY and len(sj_bars) > 0.8 * len(have)):        # 盤中若永豐快照成功，就不必再抓 Yahoo
        jobs.insert(0, (have, "5d" if INTRADAY else "1mo"))
    for codes, period in jobs:
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
    for c, bar in sj_bars.items():                                # 用永豐快照覆蓋／補上今天這根 K 棒
        if c in hist:
            df = pd.concat([hist[c], bar])
            hist[c] = df[~df.index.duplicated(keep="last")].sort_index().tail(KEEP_ROWS)
    hist = {c: hist[c] for c in tick if c in hist}
    pickle.dump(hist, open(path, "wb"))
    if len(hist) > 0.8 * len(tick):
        open(marker, "w").close()
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
        # ---- 布林上軌近 3 日斜率：上軌 U(t-2), U(t-1), U(t) → 兩段斜率取平均（元／日；另換算 %／日）
        **upper_slope3(x["up"], x["ma20"], close),
        # ---- 關鍵價位（情境整理用）
        ma5v=L("ma5"), ma10v=L("ma10"), ma20v=L("ma20"), ma60v=L("ma60"), ma240v=L("ma240"),
        high20=num(df["High"].iloc[-21:-1].max()), low20=num(df["Low"].iloc[-21:-1].min()),
        # ---- 當沖：CDP 逆勢操作價位（以今日高低收推算下一個交易日）
        **cdp_levels(num(df["High"].iloc[-1]), num(df["Low"].iloc[-1]), close),
        day_pos=num((close - df["Low"].iloc[-1]) / (df["High"].iloc[-1] - df["Low"].iloc[-1]))
               if df["High"].iloc[-1] > df["Low"].iloc[-1] else 0.5,
        # ---- 做多支撐：10日線、月線、10日低點中，位於股價下方且最接近者
        **long_support(close, L("ma10"), L("ma20"), num(df["Low"].tail(10).min())),
        vma20_lots=num(x["vma20"].iloc[-1] / 1000),
        turnover20=num((c * df["Volume"]).tail(20).mean() / 1e8),
    )
    return {k: (round(v, 4 if k.startswith("bb_up") else 3) if isinstance(v, float) else (bool(v) if isinstance(v, (bool, np.bool_)) else v))
            for k, v in s.items()}

def upper_slope3(up, mid, close):
    u = up.tail(3).to_numpy(dtype=float); m = mid.tail(3).to_numpy(dtype=float)
    if len(u) < 3 or np.isnan(u).any() or np.isnan(m).any():
        return dict(bb_up=None, bb_up_s1=None, bb_up_s2=None, bb_up_s3=None, bb_up_s3_pct=None, ma20_nodown3=None)
    s1, s2 = u[1] - u[0], u[2] - u[1]
    avg = (s1 + s2) / 2
    dm = np.diff(m)
    return dict(bb_up=round(float(u[2]), 4), bb_up_s1=float(s1), bb_up_s2=float(s2), bb_up_s3=float(avg),
                bb_up_s3_pct=float(avg / close * 100) if close else None,
                ma20_nodown3=bool((dm >= 0).all()))            # 近 3 日月線（中軌）沒有遞減

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

# --------------------------------------------------------------------------- 回測
# 問題：收盤時出現某訊號的股票，之後 10 個交易日內「收盤價曾比今天高 10% 以上」的機率，
#       是否明顯高於同期所有股票的平均（基準）？倍數 lift = 訊號機率 ÷ 基準機率。
BT_HORIZONS = (5, 10, 20)                   # 期間（交易日）
BT_THRESHOLDS = (5, 10, 15)                 # 漲幅門檻（%）
BT_DEFAULT = "h10_t10"
BT_MIN_VOL, BT_MIN_PRICE = 500, 10          # 只用 20 日均量 ≥ 500 張、股價 ≥ 10 元的日子，避免冷門股雜訊

SIG_NAMES = {
    # 技術面（key 與網頁勾選項相同，點了可以直接套用）
    "ma_bull": ("均線多頭排列", "均線"), "above_ma20": ("站上月線", "均線"), "above_ma60": ("站上季線", "均線"),
    "above_ma240": ("站上年線", "均線"), "ma20_bend": ("月線上彎", "均線"), "ma20_turn": ("月線剛翻揚", "均線"),
    "ma_tangle": ("均線糾結", "均線"),
    "kd_golden": ("KD 黃金交叉", "動能"), "kd_low_gold": ("KD 低檔黃金交叉", "動能"),
    "macd_golden": ("MACD 黃金交叉", "動能"), "macd_pos": ("DIF 在 MACD 之上", "動能"),
    "rsi_rebound": ("RSI 脫離超賣（站回 30）", "動能"), "dmi_bull": ("+DI > -DI", "動能"),
    "adx25": ("ADX ≥ 25", "動能"), "obv_up": ("OBV 在均線上", "動能"),
    "breakout20": ("突破 20 日高點", "型態"), "near_high52": ("距 52 週高 5% 內", "型態"),
    "bb_squeeze": ("布林收斂", "布林"), "bb_up_break": ("突破布林上軌", "布林"), "bb_open": ("布林開口變大", "布林"),
    "bb_up3_good": ("上軌 3 日斜率向上且月線未遞減", "布林"), "bb_mid_up": ("布林中軌上揚", "布林"),
    "vol15": ("量比 ≥ 1.5", "量能"), "vol_expand": ("5 日均量 > 20 日均量", "量能"), "up3": ("連 3 日收紅", "型態"),
    "red_vol": ("帶量紅 K（量比 ≥ 1.2）", "量能"),
    # 籌碼
    "c_f_buy": ("外資今日買超", "籌碼"), "c_t_buy": ("投信今日買超", "籌碼"), "c_all_buy": ("三大法人同步買超", "籌碼"),
    "c_f_streak": ("外資連買 ≥ 3 日", "籌碼"), "c_t_streak": ("投信連買 ≥ 3 日", "籌碼"), "c_tot5": ("法人 5 日買超", "籌碼"),
    "c_f_cap": ("外資 5 日買超 ≥ 1‰ 股本", "籌碼"), "c_t_cap": ("投信 5 日買超 ≥ 1‰ 股本", "籌碼"),
    "c_t_cap20": ("投信 20 日買超 ≥ 5‰ 股本", "籌碼"), "c_f_flip": ("外資翻多", "籌碼"), "c_t_flip": ("投信翻多", "籌碼"),
    "c_m_down5": ("融資 5 日減少", "籌碼"),
    # 策略（網頁左上角的策略選股，只用前提條件）
    "P_bottom": ("策略：低檔轉強", "策略"), "P_breakout": ("策略：量價突破", "策略"), "P_pullback": ("策略：多頭回檔", "策略"),
    "P_squeeze": ("策略：布林收斂待發", "策略"), "P_bbup": ("策略：布林上軌擴張", "策略"),
    "P_maflip": ("策略：月線上彎＋法人翻多", "策略"), "P_trust": ("策略：投信認養", "策略"), "P_chips": ("策略：法人買、散戶退", "策略"),
}
CHIP_KEYS = {k for k in SIG_NAMES if k.startswith("c_")} | {"P_maflip", "P_trust", "P_chips"}

def _ws(sr, n=5):
    d = sr.diff(); w = np.arange(n, 0, -1)
    return sum(w[i] * d.shift(i) for i in range(n)) / w.sum()

def bt_signals(df, x, code, inst, marg, shares):
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    S = {}
    ma5, ma10, ma20, ma60 = x["ma5"], x["ma10"], x["ma20"], x["ma60"]
    S["ma_bull"] = (ma5 > ma20) & (ma20 > ma60); S["above_ma20"] = c > ma20; S["above_ma60"] = c > ma60
    S["above_ma240"] = c > x["ma240"]
    s20 = _ws(ma20) / c; s20p = s20.shift(5)
    S["ma20_bend"] = (s20 > 0) & (s20 > s20p); S["ma20_turn"] = (s20 > 0) & (s20p <= 0)
    mx = pd.concat([ma5, ma10, ma20], axis=1); S["ma_tangle"] = (mx.max(axis=1) - mx.min(axis=1)) / c < 0.02
    K, D = x["K"], x["D"]; kg = (K.shift() < D.shift()) & (K > D)
    S["kd_golden"] = kg; S["kd_low_gold"] = kg & (K < 30)
    S["macd_golden"] = (x["dif"].shift() < x["macd"].shift()) & (x["dif"] > x["macd"]); S["macd_pos"] = x["dif"] > x["macd"]
    S["rsi_rebound"] = (x["rsi"].shift() <= 30) & (x["rsi"] > 30)
    S["dmi_bull"] = x["pdi"] > x["mdi"]; S["adx25"] = x["adx"] >= 25; S["obv_up"] = x["obv"] > x["obv_ma20"]
    S["breakout20"] = c > h.shift().rolling(20).max()
    S["near_high52"] = c >= 0.95 * h.rolling(250, min_periods=120).max()
    S["bb_squeeze"] = x["bbw"] <= 1.1 * x["bbw"].rolling(120, min_periods=60).min()
    S["bb_up_break"] = c > x["up"]; S["bb_open"] = _ws(x["bbw"]) > 0; S["bb_mid_up"] = s20 > 0
    S["bb_up3_good"] = ((x["up"] - x["up"].shift(2)) > 0) & ((ma20.diff() >= 0).rolling(2).sum() == 2)
    vr = v / x["vma20"].shift(); red = c > o
    S["vol15"] = vr >= 1.5; S["vol_expand"] = x["vma5"] > x["vma20"]; S["up3"] = (c.diff() > 0).rolling(3).sum() == 3
    S["red_vol"] = red & (vr >= 1.2)
    # 策略前提
    hist = x["dif"] - x["macd"]
    low60 = c / l.rolling(60).min() - 1; ret60 = c / c.shift(60) - 1
    bot_base = (low60 <= 0.15) & ((c < ma60) | (ret60 < -0.10)) & ((x["rsi"].rolling(10).min() < 35) | (K.rolling(5).min() < 25))
    bot_sc = (kg & (K < 40)).astype(int) + ((hist.shift() < 0) & (hist > hist.shift())).astype(int) \
        + ((c.shift() < ma5.shift()) & (c > ma5)).astype(int) + S["red_vol"].astype(int) + S["rsi_rebound"].astype(int) \
        + S["obv_up"].astype(int)
    S["P_bottom"] = bot_base & (bot_sc >= 2)
    S["P_breakout"] = S["breakout20"] & (vr >= 1.5) & red
    S["P_pullback"] = (ma20 > ma60) & (ma60 > ma60.shift(20)) & (x["bias20"] >= -3) & (x["bias20"] <= 2) & (K < 50)
    S["P_squeeze"] = S["bb_squeeze"]; S["P_bbup"] = S["bb_up3_good"]
    # 籌碼（以日期對齊；沒有籌碼資料的日子為 NaN → 不列入籌碼類統計）
    days = [d.strftime("%Y-%m-%d") for d in df.index]
    ser = lambda j: pd.Series([(inst.get(d) or {}).get(code, [np.nan] * 4)[j] for d in days], index=df.index, dtype=float)
    f, t, dd = ser(0), ser(1), ser(2)
    has = f.notna()
    f5, t5, t20 = f.rolling(5).sum(), t.rolling(5).sum(), t.rolling(20).sum()
    lots = shares / 1000 if shares else np.nan
    S["c_f_buy"] = f > 0; S["c_t_buy"] = t > 0; S["c_all_buy"] = (f > 0) & (t > 0) & (dd > 0)
    S["c_f_streak"] = (f > 0).rolling(3).sum() == 3; S["c_t_streak"] = (t > 0).rolling(3).sum() == 3
    S["c_tot5"] = (f + t + dd).rolling(5).sum() > 0
    S["c_f_cap"] = f5 / lots * 1000 >= 1; S["c_t_cap"] = t5 / lots * 1000 >= 1; S["c_t_cap20"] = t20 / lots * 1000 >= 5
    S["c_f_flip"] = (f5 > 0) & ~(f5.shift(5) > 0); S["c_t_flip"] = (t5 > 0) & ~(t5.shift(5) > 0)
    mb = pd.Series([(marg.get(d) or {}).get(code, [np.nan] * 4)[0] for d in days], index=df.index, dtype=float)
    S["c_m_down5"] = (mb - mb.shift(5)) < 0
    S["P_maflip"] = S["ma20_bend"] & (S["c_f_flip"] | S["c_t_flip"] | ((f > 0).rolling(2).sum() == 2) | ((t > 0).rolling(2).sum() == 2))
    S["P_trust"] = S["c_t_streak"] & (t5 > 0)
    S["P_chips"] = S["c_tot5"] & S["c_m_down5"]
    sig = pd.DataFrame(S).fillna(False).astype(bool)
    # 結果：未來 h 日內最高收盤漲幅（h = 5／10／20），以及 5／20 日報酬
    out = pd.DataFrame({"r5": c.shift(-5) / c - 1, "r20": c.shift(-20) / c - 1,
                        "okb": (x["vma20"] / 1000 >= BT_MIN_VOL) & (c >= BT_MIN_PRICE) & x["ma60"].notna(),
                        "chip": has & f5.notna(), "date": df.index})
    for h in BT_HORIZONS:
        out[f"fmax{h}"] = (c[::-1].rolling(h).max()[::-1].shift(-1) / c - 1).astype("float32")
    return sig, out

FEATS = [k for k in SIG_NAMES if not k.startswith("P_")]

def fit_logit(X, y, lam=2.0, iters=30):
    """L2 正則化邏輯斯迴歸（IRLS），回傳 [截距, 各特徵權重]。"""
    X1 = np.hstack([np.ones((len(X), 1), dtype=np.float32), X.astype(np.float32)]); w = np.zeros(X1.shape[1])
    R = lam * np.eye(X1.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -30, 30))); W = p * (1 - p)
        H = (X1.T @ (X1 * W[:, None].astype(np.float32))).astype(float) + R
        g = (X1.T @ (y - p).astype(np.float32)).astype(float) - R @ w
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-6:
            break
    return w

def predict(w, X):
    return 1 / (1 + np.exp(-np.clip(w[0] + X @ w[1:], -30, 30)))

def auc(pred, y):
    r = pd.Series(pred).rank().to_numpy(); npos = y.sum(); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None

def feat_matrix(sig, chip):
    X = sig[FEATS].to_numpy(dtype=np.float32)
    return np.hstack([X, chip.reshape(-1, 1).astype(np.float32)])     # 最後一欄：是否有籌碼資料（float32 節省記憶體）

def build_model(sig, out, ok, big):
    y = big.astype(float)
    X = feat_matrix(sig, out["chip"].to_numpy())
    dates = out["date"].to_numpy()
    Xo, yo, do = X[ok], y[ok], dates[ok]
    if len(yo) < 2000 or yo.sum() < 100:
        return None
    ds = pd.Series(pd.to_datetime(do)); cut = ds.quantile(0.7)
    tr = (ds <= cut).to_numpy(); te = ~tr
    if tr.sum() < 1000 or te.sum() < 300:
        return None
    w_tr = fit_logit(Xo[tr], yo[tr])
    pt = predict(w_tr, Xo[te])
    # 校準：把測試期依預測機率分 5 組，看實際發生率
    calib = []
    if te.sum() > 500:
        qs = np.quantile(pt, [0, .2, .4, .6, .8, 1])
        for i in range(5):
            m = (pt >= qs[i]) & (pt <= qs[i + 1] if i == 4 else pt < qs[i + 1])
            if m.sum():
                calib.append(dict(pred=round(float(pt[m].mean()) * 100, 1), actual=round(float(yo[te][m].mean()) * 100, 1),
                                  n=int(m.sum())))
    w = fit_logit(Xo, yo)                                               # 用全部資料重新估計，給今天打分數
    return dict(features=FEATS + ["has_chip"], coef=[round(float(v), 5) for v in w[1:]], intercept=round(float(w[0]), 5),
                auc_test=round(auc(pt, yo[te]), 3) if te.sum() else None, calib=calib,
                n_train=int(tr.sum()), n_test=int(te.sum()), test_from=cut.strftime("%Y-%m-%d"),
                base_rate=round(float(yo.mean()) * 100, 1))

BT_PERIODS = (1, 3, 5)                                   # 回測期間（年）

def run_backtest(frames):
    if not frames:
        return None
    sig = pd.concat([f[0] for f in frames], ignore_index=True)
    out = pd.concat([f[1] for f in frames], ignore_index=True)
    last = out.loc[out["okb"], "date"].max()
    periods = {}
    for yrs in BT_PERIODS:
        m = (out["date"] > last - pd.DateOffset(years=yrs)).to_numpy()
        sg, oc = sig[m].reset_index(drop=True), out[m].reset_index(drop=True)
        periods[f"{yrs}y"] = {}
        for h in BT_HORIZONS:
            for t in BT_THRESHOLDS:
                res = _bt_core(sg, oc, len(frames), h, t)
                if res:
                    res["years"] = yrs
                    periods[f"{yrs}y"][f"h{h}_t{t}"] = res
                    print(f"回測 {yrs} 年｜{h} 日漲 {t}%：基準 {res['baseline']['p_big']}%，"
                          f"模型 AUC {res['model']['auc_test'] if res.get('model') else '–'}")
    return dict(generated=dt.datetime.now(TZ).strftime("%Y-%m-%d %H:%M"), default_period="1y",
                default_target=BT_DEFAULT, horizons=list(BT_HORIZONS), thresholds=list(BT_THRESHOLDS), periods=periods)

def _bt_core(sig, out, n_stocks, h, t):
    fm = out[f"fmax{h}"].to_numpy()
    ok = out["okb"].to_numpy() & ~np.isnan(fm); big = fm >= t / 100
    r5 = out["r5"].to_numpy(); r20 = out["r20"].to_numpy()
    chip = out["chip"].to_numpy()
    dates = out["date"]; mid = dates[ok].quantile(0.5) if ok.any() else None
    h1 = (dates <= mid).to_numpy() if mid is not None else ok
    def stat(mask, base_mask):
        m = mask & base_mask
        n = int(m.sum())
        if n == 0:
            return None
        p = big[m].mean(); pb = big[base_mask].mean()
        r20v = r20[m]; r20v = r20v[~np.isnan(r20v)]
        lift_half = []
        for half in (h1, ~h1):
            mm, bb = m & half, base_mask & half
            lift_half.append(round(float(big[mm].mean() / big[bb].mean()), 2) if mm.sum() >= 30 and big[bb].mean() > 0 else None)
        return dict(n=n, p_big=round(float(p) * 100, 1), lift=round(float(p / pb), 2) if pb > 0 else None,
                    r5=round(float(np.nanmean(r5[m])) * 100, 2), r20=round(float(np.nanmean(r20v)) * 100, 2) if len(r20v) else None,
                    win20=round(float((r20v > 0).mean()) * 100, 1) if len(r20v) else None,
                    lift_h1=lift_half[0], lift_h2=lift_half[1])
    base_all, base_chip = ok, ok & chip
    res = []
    for k, (name, grp) in SIG_NAMES.items():
        st = stat(sig[k].to_numpy(), base_chip if k in CHIP_KEYS else base_all)
        if st:
            res.append(dict(key=k, name=name, group=grp, chip=k in CHIP_KEYS, **st))
    # 兩兩組合：取樣本足夠、倍數最高的 12 個單一訊號（排除策略）互相搭配
    top = [r for r in sorted(res, key=lambda r: -(r["lift"] or 0)) if r["n"] >= 300 and not r["key"].startswith("P_")][:12]
    combos = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i]["key"], top[j]["key"]
            st = stat(sig[a].to_numpy() & sig[b].to_numpy(), base_chip if (a in CHIP_KEYS or b in CHIP_KEYS) else base_all)
            if st and st["n"] >= 100:
                combos.append(dict(keys=[a, b], name=f"{SIG_NAMES[a][0]} ＋ {SIG_NAMES[b][0]}", **st))
    combos.sort(key=lambda r: -(r["lift"] or 0))
    okd = dates[ok]
    def base_stat(mask):
        return dict(n=int(mask.sum()), p_big=round(float(big[mask].mean()) * 100, 1) if mask.any() else None,
                    r20=round(float(np.nanmean(r20[mask])) * 100, 2) if mask.any() else None)
    try:
        model = build_model(sig, out, ok, big)
    except Exception as e:
        print(f"⚠ 機率模型失敗：{e}"); model = None
    return dict(model=model, generated=dt.datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),
                period=[okd.min().strftime("%Y-%m-%d"), okd.max().strftime("%Y-%m-%d")] if len(okd) else None,
                chip_period=[dates[base_chip].min().strftime("%Y-%m-%d"), dates[base_chip].max().strftime("%Y-%m-%d")] if base_chip.any() else None,
                horizon=h, big=t, min_vol=BT_MIN_VOL,
                baseline=base_stat(base_all), baseline_chip=base_stat(base_chip),
                n_stocks=n_stocks, signals=res, combos=combos[:20])

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
    bt_path = os.path.join(CACHE, "backtest.json")
    do_bt = not INTRADAY and (TEST or not os.path.exists(bt_path) or
                              dt.date.fromtimestamp(os.path.getmtime(bt_path)) != dt.date.today())
    bt_frames, last_feat = [], {}
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
            try:
                sg, oc = bt_signals(df, x, s["code"], inst, marg, s.get("shares"))
                last_feat[s["code"]] = (sg.iloc[[-1]], oc["chip"].to_numpy()[-1:])
                if do_bt:
                    bt_frames.append((sg, oc))
            except Exception as e:
                errs.append(f"{s['code']} 回測: {e}")
        except Exception as e:
            errs.append(f"{s['code']}: {e}")
    if errs:
        print(f"⚠ 指標計算失敗 {len(errs)} 檔，例如 {errs[:3]}")
    if not rows:
        sys.exit("沒有任何股票算出指標，停止（避免把網站覆蓋成空的）")
    if do_bt and time.time() - START > BT_TIME_LIMIT:
        print("⚠ 本次執行時間已長，先跳過回測（下次執行再算），優先更新股價")
        do_bt = False
    if do_bt:
        bt = run_backtest(bt_frames)
        if bt:
            json.dump(bt, open(bt_path, "w", encoding="utf-8"), ensure_ascii=False)
            print("回測完成")
    if os.path.exists(bt_path):
        shutil.copy(bt_path, os.path.join(OUT, "backtest.json"))
        # 用模型替每一檔打分數：10 日內漲 ≥10% 的估計機率，並列出目前成立的訊號
        try:
            # 機率由網頁依「期間／漲幅／天數」選項，用模型係數即時計算
            for r in rows:
                lf = last_feat.get(r["code"])
                if lf is None:
                    continue
                sg, chip = lf
                r["sig_on"] = [k for k in SIG_NAMES if bool(sg[k].iloc[0])]
                r["has_chip"] = bool(chip[0])
        except Exception as e:
            print(f"⚠ 機率打分失敗：{e}")
    last = max(r["date"] for r in rows)
    meta = dict(updated=dt.datetime.now(TZ).strftime("%Y-%m-%d %H:%M"), last_date=last, intraday=INTRADAY,
                price_source="永豐 Shioaji" if (SJ_KEY and SJ_SECRET) else "Yahoo Finance",
                indices=fetch_indices(),
                inst_date=max((r.get("inst_date") or "" for r in rows), default="") or None,
                margin_date=max((r.get("margin_date") or "" for r in rows), default="") or None,
                warrant_date=max((r.get("w_date") or "" for r in rows), default="") or None,
                n_list=len(listed), n_rows=len(rows), n_stale=sum(r["date"] < last for r in rows), rows=rows)
    json.dump(meta, open(os.path.join(OUT, "snapshot.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"完成：{len(rows)} 檔，資料日期 {last}")

if __name__ == "__main__":
    main()
