#!/usr/bin/env python3
"""
台股上市櫃 每日技術指標更新
- 清單：TWSE OpenAPI t187ap03_L（上市）＋ 公開資訊觀測站 t187ap03_O.csv（上櫃）
- 價量：Yahoo Finance（上市 .TW／上櫃 .TWO），透過 yfinance
- 輸出：site/data/snapshot.json（每檔最新指標）＋ site/data/px/<代號>.json（近 180 日線圖資料）
- 歷史價量存在 cache/hist.pkl（由 GitHub Actions cache 保留），每天只補抓最近一個月
本機測試：TEST_MODE=1 python update.py（用模擬資料，不連網）
"""
import io, json, os, re, shutil, sys, time, pickle, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

ROOT  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
OUT   = os.path.join(ROOT, "site", "data")
PXDIR = os.path.join(OUT, "px")
TEST  = os.environ.get("TEST_MODE") == "1"
TZ    = ZoneInfo("Asia/Taipei")
KEEP_ROWS, CHART_ROWS, MIN_ROWS = 320, 180, 80

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

# --------------------------------------------------------------------------- 清單
def _clean(rows, market):
    out = []
    for code, name, ind in rows:
        code = str(code).strip()
        if not re.fullmatch(r"[1-9]\d{3}", code):      # 只留一般股
            continue
        ind = str(ind).strip().zfill(2)
        out.append(dict(code=code, name=str(name).strip(), industry=IND_MAP.get(ind, ind), market=market))
    return out

def fetch_tse():
    r = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", timeout=60)
    r.raise_for_status()
    return _clean([(x["公司代號"], x["公司簡稱"], x["產業別"]) for x in r.json()], "上市")

def fetch_otc():
    r = requests.get("https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv", timeout=60)
    r.raise_for_status()
    for enc in ("utf-8-sig", "cp950"):
        try:
            txt = r.content.decode(enc); break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(txt), dtype=str)
    return _clean(zip(df["公司代號"], df["公司簡稱"], df["產業別"]), "上櫃")

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
    for codes, period in ((have, "1mo"), (need, "18mo")):
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
        vma20_lots=num(x["vma20"].iloc[-1] / 1000),
        turnover20=num((c * df["Volume"]).tail(20).mean() / 1e8),
    )
    return {k: (round(v, 3) if isinstance(v, float) else (bool(v) if isinstance(v, (bool, np.bool_)) else v))
            for k, v in s.items()}

CHART_COLS = ["ma5","ma10","ma20","ma60","ma120","ma240","up","dn","K","D","rsi","dif","macd",
              "pdi","mdi","adx","obv","obv_ma20","wr","cci","mfi","bias20"]

def arr(s, nd=2):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]

def chart_json(df, x):
    d, y = df.tail(CHART_ROWS), x.tail(CHART_ROWS)
    out = {"d": [i.strftime("%Y-%m-%d") for i in d.index],
           "o": arr(d["Open"]), "h": arr(d["High"]), "l": arr(d["Low"]), "c": arr(d["Close"]),
           "v": arr(d["Volume"] / 1000, 0)}
    for col in CHART_COLS:
        out[col] = arr(y[col] / 1000 if col.startswith("obv") else y[col])
    return out

# --------------------------------------------------------------------------- main
def main():
    os.makedirs(CACHE, exist_ok=True)
    shutil.rmtree(PXDIR, ignore_errors=True); os.makedirs(PXDIR, exist_ok=True)
    if TEST:
        listed = [dict(code=str(1101 + i), name=f"測試{i}", industry=["半導體業", "航運業", "金融保險"][i % 3],
                       market="上市" if i % 2 else "上櫃") for i in range(60)]
    else:
        listed = get_list()
    hist = update_history(listed)
    rows, errs = [], []
    for s in listed:
        df = hist.get(s["code"])
        if df is None or len(df) < MIN_ROWS:
            continue
        try:
            x = indicators(df)
            rows.append({**s, **snapshot(df, x)})
            json.dump(chart_json(df, x), open(os.path.join(PXDIR, s["code"] + ".json"), "w"),
                      separators=(",", ":"))
        except Exception as e:
            errs.append(f"{s['code']}: {e}")
    if errs:
        print(f"⚠ 指標計算失敗 {len(errs)} 檔，例如 {errs[:3]}")
    if not rows:
        sys.exit("沒有任何股票算出指標，停止（避免把網站覆蓋成空的）")
    last = max(r["date"] for r in rows)
    meta = dict(updated=dt.datetime.now(TZ).strftime("%Y-%m-%d %H:%M"), last_date=last,
                n_list=len(listed), n_rows=len(rows), n_stale=sum(r["date"] < last for r in rows), rows=rows)
    json.dump(meta, open(os.path.join(OUT, "snapshot.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"完成：{len(rows)} 檔，資料日期 {last}")

if __name__ == "__main__":
    main()
