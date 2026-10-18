"""竞价修复系统(RELAY_v1):全市场 9:25 竞价 + D 日数据,预测「T 开盘买、T+1 收盘卖仍赚钱」,每天前 5 名等权买入。

依据 docs/REALIZABLE.md 第 8 节:只用 D 日数据找不到接力票(AUC ≈ 0.51);加竞价跳空(相对 T 日参考价)后,
每天前 5 名等权 2022 ~ 2026 每年为正(第一名单买 2024 年接近 0,所以用前 5 分散)。
回测按开盘价成交,实盘看到竞价后只能 9:30 之后买,所以本系统同时记录 9:30 之后的可成交价,用前向数据检验买入价吃掉多少。

每日流程(北京时间):
- D 日盘后 `relay-train --date D`:用卖出日 ≤ D 的全部样本重训模型,冻结 D 日全部可交易主板股票的特征快照(只增不改)。
- T 日 9:26 `relay-pick --date T`:拉快照里全部股票的腾讯行情(约 3.5 秒),竞价到涨停价 / 无竞价的跳过,打分,出前 5。
- T 日 9:31 后 `relay-fill --date T`:再拉一次前 5 的现价,记作「9:30 后可成交价」。
- 每天盘后 `relay-settle`:T+1 收盘后结算(T+1 收盘卖;收盘跌停或停牌顺延到能卖的开盘),分别按开盘价和 9:31 价算净收益,写台账。
"""
from __future__ import annotations

import json
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import TZ, Config
from ..data.auction import validate_auction_record
from ..data.calendar import TradingCalendar
from ..data.ingest import calendar_path
from ..evaluation import walkforward as wf
from ..features.market import MARKET_FEATURES
from ..features.sector import SECTOR_FEATURES
from ..features.stock import STOCK_FEATURES
from ..rules.limits import limit_band
from .strategy_v2 import load_panel
from .strategy_v3 import QUOTE_URL

RULE_VERSION = "RELAY_v1"
STAGES = ["冰点", "修复", "发酵", "高潮", "分歧", "退潮", "震荡"]
ROLES = {"龙头": 3, "前排": 2, "跟风": 1}
THEME = ["ml_score", "ml_rank", "heat", "lu5", "lu10", "anti5", "height", "amt_rank_grp", "leader_score", "role_i",
         "prev_role_i", "sent_score", "stage_i", "is_mainline", "is_space_leader"]
D_FEATURES = list(dict.fromkeys(STOCK_FEATURES + SECTOR_FEATURES + MARKET_FEATURES + THEME + ["d_gap", "d_oc"]))
FEATURES = D_FEATURES + ["gap"]
# deterministic + force_row_wise + 固定 4 线程:同一份数据、同一 LightGBM 版本(4.7.0)在任何机器上训练出同一个模型
PARAMS: dict[str, Any] = dict(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=500, subsample=0.5,
                              subsample_freq=1, colsample_bytree=0.7, random_state=7, verbose=-1,
                              deterministic=True, force_row_wise=True, n_jobs=4)
TOP_N = 5


def root(cfg: Config) -> Path:
    return cfg.data_root / "system" / "relay"


def _x(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].astype(float).replace([np.inf, -np.inf], np.nan)


def _prepare(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[panel["board"] == "main"].copy()
    p["d_gap"] = p["open"] / p["prev_close"] - 1
    p["d_oc"] = p["close"] / p["open"] - 1
    p["stage_i"] = p["stage"].map({s: i for i, s in enumerate(STAGES)})
    p["role_i"] = p["role"].map(ROLES).fillna(0)
    p["prev_role_i"] = p["prev_role"].map(ROLES).fillna(0)
    keep = ["trade_date", "ts_code", "name", "board", "is_risk_warning", "eligible", "open", "high", "low", "close",
            "prev_close", "up_limit", "down_limit"] + D_FEATURES
    p = p[list(dict.fromkeys(keep))]
    floats = [c for c in p.columns if p[c].dtype == "float64"]
    p[floats] = p[floats].astype("float32")
    return p


def training_rows(p: pd.DataFrame, asof: date) -> pd.DataFrame:
    """可交易主板股票的 T 开盘买 / T+1 收盘卖结果;只保留卖出日 ≤ asof(asof 收盘后已知)的行。"""
    nb = wf._next_bar(p)
    nb["t_ref"] = nb.groupby("ts_code", sort=False)["prev_close"].shift(-1)  # T 日交易所前收(除权日为除权参考价)
    e = nb[nb["eligible"].astype(bool) & nb["t_date"].notna()]
    e = e[(e["t_open"] > 0) & ((e["t_open"] - e["t_up"]).abs() > 1e-6)]
    e = e[["trade_date", "ts_code", "t_date", "t_open", "t_close", "t_ref", "close"] + D_FEATURES].reset_index(drop=True)
    e = wf._realized(e, p[["trade_date", "ts_code", "open", "high", "close", "down_limit"]])
    e = e[e["x_close1"].notna() & (e["x_close1"] <= asof)].copy()
    e["gap"] = e["t_open"] / e["t_ref"] - 1  # 相对 T 日参考价,除权日不会被当成低开
    e["y"] = (e["r_close1"] > 0).astype(int)
    return e


def train(cfg: Config, d: date, panel: pd.DataFrame | None = None) -> Path:
    import lightgbm as lgb

    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    t = cal.next(d)
    p = _prepare(load_panel(cfg) if panel is None else panel)
    p = p[p["trade_date"] <= d]
    rows = training_rows(p, d)
    model = lgb.LGBMClassifier(**PARAMS).fit(_x(rows, FEATURES), rows["y"])
    day = p[(p["trade_date"] == d) & p["eligible"].astype(bool)].copy()
    if day.empty:
        raise RuntimeError(f"没有 {d} 的可交易股票特征,先跑 build / structure")
    out = root(cfg) / f"{d:%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.txt").exists():
        raise RuntimeError(f"{out} 已有模型,信号日的模型与快照只增不改")
    model.booster_.save_model(str(out / "model.txt"))
    day[["ts_code", "name", "board", "is_risk_warning", "close"] + D_FEATURES].to_parquet(out / "snapshot.parquet", index=False)
    meta = {"rule_version": RULE_VERSION, "signal_date": str(d), "target_date": str(t), "train_rows": int(len(rows)),
            "train_sell_date_max": str(rows["x_close1"].max()), "features": FEATURES, "params": PARAMS,
            "frozen_at": datetime.now(TZ).isoformat(), "universe": int(len(day))}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def _quotes(codes: list[str], field: int) -> dict[str, float]:
    """腾讯行情批量拉取;field 5 = 今开(9:25 后即竞价价),3 = 现价,4 = 昨收(交易所参考价)。"""
    import requests

    sess = requests.Session()
    out: dict[str, float] = {}
    sym = [("sh" if c.endswith(".SH") else "sz") + c[:6] for c in codes]
    for i in range(0, len(sym), 400):
        text = sess.get(QUOTE_URL + ",".join(sym[i:i + 400]), timeout=10).text
        for line in text.strip().split(";"):
            if "~" not in line:
                continue
            f = line.split('"')[1].split("~")
            code = f[2] + (".SH" if f[2].startswith("6") else ".SZ")
            px = float(f[field] or 0)
            if px > 0:
                out[code] = px
    return out


def score_day(snapshot: pd.DataFrame, auction: dict[str, float], ref: dict[str, float], t: date, model_file: Path,
              n: int = TOP_N) -> pd.DataFrame:
    """ref = T 日交易所前收(除权日为除权参考价);缺失时退回 D 日收盘。涨停价和跳空都以它为基准。"""
    import lightgbm as lgb

    s = snapshot.copy()
    s["auction_price"] = s["ts_code"].map(auction)
    s["ref"] = s["ts_code"].map(ref).fillna(s["close"])
    s["t_up_limit"] = [limit_band(float(r), b, bool(st), t).up for r, b, st in zip(s["ref"], s["board"], s["is_risk_warning"])]
    s["skip"] = np.where(s["auction_price"].isna(), "无竞价(停牌)",
                         np.where(s["auction_price"] >= s["t_up_limit"] - 1e-6, "竞价已到涨停价", ""))
    s = s[s["skip"] == ""].copy()
    s["gap"] = s["auction_price"] / s["ref"] - 1
    s["score"] = lgb.Booster(model_file=str(model_file)).predict(_x(s, FEATURES))
    s = s.sort_values(["score", "ts_code"], ascending=[False, True]).head(n)
    s["rank"] = np.arange(1, len(s) + 1)
    return s[["rank", "ts_code", "name", "close", "ref", "auction_price", "gap", "score"]].reset_index(drop=True)


def pick(cfg: Config, t: date, replay: bool = False) -> Path:
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    d = cal.prev(t)
    src = root(cfg) / f"{d:%Y%m%d}"
    snap = pd.read_parquet(src / "snapshot.parquet")
    if replay:  # 历史复现:用 T 日日线开盘价当 9:25 竞价价
        from ..build import features_dir

        bars = pd.read_parquet(features_dir(cfg) / "features.parquet", columns=["trade_date", "ts_code", "open", "prev_close"])
        bars = bars[bars["trade_date"] == t]
        auction = {c: float(o) for c, o in zip(bars["ts_code"], bars["open"]) if o and o > 0}
        ref = {c: float(v) for c, v in zip(bars["ts_code"], bars["prev_close"]) if v and v > 0}
        observed = datetime.combine(t, dtime(9, 26), TZ)
    else:
        observed = datetime.now(TZ)
        auction = _quotes(snap["ts_code"].tolist(), 5)
        ref = _quotes(snap["ts_code"].tolist(), 4)
        for code, px in list(auction.items())[:50]:  # 抽查时点与字段白名单
            validate_auction_record({"trade_date": str(t), "ts_code": code, "price": px, "source": "tencent_qt",
                                     "observed_available_at": observed,
                                     "underlying_event_at": datetime.combine(t, dtime(9, 25), TZ)})
    auction = {c: v for c, v in auction.items() if c in set(snap["ts_code"])}
    top = score_day(snap, auction, ref, t, src / "model.txt")
    out = src / "pick.json"
    if out.exists():
        raise RuntimeError(f"{out} 已存在,选票不可覆盖")
    doc = {"rule_version": RULE_VERSION, "signal_date": str(d), "target_date": str(t), "observed_at": observed.isoformat(),
           "historical_replay": replay, "auction_quotes": len(auction), "universe": int(len(snap)),
           "exit_rule": "T+1 收盘卖;T+1 收盘在跌停价或停牌卖不出,顺延到之后第一个能卖的开盘",
           "picks": top.to_dict("records")}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    L = [f"# {RULE_VERSION} 选票:{t}(信号日 {d})", "",
         f"- 竞价行情 {len(auction)} / {len(snap)} 只;{'历史复现(开盘价当竞价价)' if replay else '实时 ' + observed.strftime('%H:%M:%S')}",
         f"- 买入:前 5 名 9:30 后等权买入(单票不超过账户的 4%);卖出:{doc['exit_rule']}", "",
         "| 名次 | 代码 | 名称 | T 日参考价 | 竞价 | 跳空 | 分数 |", "|---|---|---|---|---|---|---|"]
    for r in top.itertuples(index=False):
        L.append(f"| {r.rank} | {r.ts_code} | {r.name} | {r.ref:.2f} | {r.auction_price:.2f} | {r.gap:+.2%} | {r.score:.3f} |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return out.with_suffix(".md")


def fill(cfg: Config, t: date) -> Path:
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    src = root(cfg) / f"{cal.prev(t):%Y%m%d}"
    doc = json.loads((src / "pick.json").read_text(encoding="utf-8"))
    now = datetime.now(TZ)
    if now.date() != t or now.time() < dtime(9, 30):
        raise RuntimeError("9:30 之后、当天才能记录可成交价")
    px = _quotes([r["ts_code"] for r in doc["picks"]], 3)
    out = src / "fill.json"
    if out.exists():
        raise RuntimeError(f"{out} 已存在,可成交价记录不可覆盖")
    out.write_text(json.dumps({"observed_at": now.isoformat(), "prices": px}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def settle(cfg: Config) -> Path:
    """结算所有已到 T+1 收盘的选票,写台账(每次全量重算,不存可变状态)。"""
    from ..build import features_dir

    f = pd.read_parquet(features_dir(cfg) / "features.parquet",
                        columns=["trade_date", "ts_code", "open", "high", "close", "down_limit"])
    rows = []
    for pj in sorted(root(cfg).glob("*/pick.json")):
        doc = json.loads(pj.read_text(encoding="utf-8"))
        for top in doc["picks"][:TOP_N]:
            rows.append(_settle_one(f, pj, doc, top))
    led = pd.DataFrame(rows)
    out = root(cfg) / "ledger.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(out, index=False)
    done = led.dropna(subset=["ret_open"]) if "ret_open" in led else led.iloc[0:0]
    live = done[~done["replay"].astype(bool)] if len(done) else done
    L = [f"# {RULE_VERSION} 台账(前 {TOP_N} 名等权)", "",
         f"- 已结算 {len(done)} 笔 / {done['target_date'].nunique() if len(done) else 0} 天(其中实时 {len(live)} 笔)"]
    if len(done):
        day = done.groupby("target_date")["ret_open"].mean()
        L.append(f"- 按开盘价:每笔平均 {done['ret_open'].mean():+.2%},胜率 {(done['ret_open'] > 0).mean():.0%};"
                 f"按天(5 只等权)平均 {day.mean():+.2%},赚钱的天 {(day > 0).mean():.0%}")
    if len(live) and live["ret_fill"].notna().any():
        lf = live.dropna(subset=["ret_fill"])
        L.append(f"- 按 9:31 价:每笔平均 {lf['ret_fill'].mean():+.2%},胜率 {(lf['ret_fill'] > 0).mean():.0%};"
                 f"9:31 价比开盘价平均 {lf['fill_vs_open'].mean():+.2%}")
    (out.with_suffix(".md")).write_text("\n".join(L) + "\n", encoding="utf-8")
    return out.with_suffix(".md")


def _settle_one(f: pd.DataFrame, pj: Path, doc: dict, top: dict) -> dict[str, Any]:
    """单只选票:T 开盘(及 9:31 价)买入,T+1 收盘卖;收盘跌停或停牌顺延到之后第一个能卖的开盘。"""
    code, t = top["ts_code"], date.fromisoformat(doc["target_date"])
    bars = f[(f["ts_code"] == code) & (f["trade_date"] >= t)].sort_values("trade_date").head(6).reset_index(drop=True)
    fillp = None
    if (pj.parent / "fill.json").exists():
        fillp = json.loads((pj.parent / "fill.json").read_text(encoding="utf-8"))["prices"].get(code)
    row: dict[str, Any] = {"target_date": str(t), "ts_code": code, "name": top["name"], "replay": doc["historical_replay"],
                           "auction_price": top["auction_price"], "fill_0931": fillp}
    if len(bars) >= 2 and bars.loc[0, "trade_date"] == t:
        row["t_open"] = float(bars.loc[0, "open"])
        exit_px, exit_d = None, None
        c2, dn2 = bars.loc[1, "close"], bars.loc[1, "down_limit"]
        if c2 > 0 and abs(c2 - dn2) > 1e-6:
            exit_px, exit_d = float(c2), bars.loc[1, "trade_date"]
        else:
            for k in range(2, len(bars)):
                o, h, dn = bars.loc[k, "open"], bars.loc[k, "high"], bars.loc[k, "down_limit"]
                if o > 0 and not (abs(o - dn) < 1e-6 and abs(h - dn) < 1e-6):
                    exit_px, exit_d = float(o), bars.loc[k, "trade_date"]
                    break
        if exit_px is not None:
            cost = wf.cost_on(exit_d)
            row.update({"exit_date": str(exit_d), "exit_price": exit_px,
                        "ret_open": exit_px / row["t_open"] - 1 - cost,
                        "ret_fill": (exit_px / fillp - 1 - cost) if fillp else None,
                        "fill_vs_open": (fillp / row["t_open"] - 1) if fillp else None})
    return row
