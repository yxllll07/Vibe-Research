"""M2 单股票时序编码器(TemporalCNN)与 M3 市场—行业—个股交叉注意力 Transformer。

结构与超参数按研究方案第 4 章写死,不在验证期调参:
- 序列长度 L=60;编码维度 d=128;M3 为候选 query 的交叉注意力,2 层、4 头、dropout 0.1。
- M3 的 token:1 个市场 token(D 日市场广度)、31 个申万一级行业 token(D 日行业结构)、
  同行业 D 日成交额前 16 的同业个股 token(与候选共享 M2 编码器)、候选自身。
  同一截面的 token 都在 D 日收盘前已发生,彼此可以关注;不会看到 T 日行情。
- 消融:M3_shuffled(同业换成随机另一行业的股票,固定种子)、M3_no_market(去掉市场 token)。
训练:训练期可买入的候选样本,每个 (交易日, 模式) 最多抽 60 个(固定种子),
目标与 M1 相同(扣费后代理收益为正),Adam 1e-3,batch 256,固定 4 个 epoch。
所有通道的标准化统计量只用训练期日期拟合。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..candidates.modes import MODES
from ..features.market import MARKET_FEATURES
from .models import ROUND_TRIP_COST, TARGETS

log = logging.getLogger(__name__)
SEED = 20261017
L = 60
D_MODEL = 128
HEADS = 4
LAYERS = 2
DROPOUT = 0.1
PEERS = 16
EPOCHS = 4
BATCH = 256
LR = 1e-3
PER_DAY_MODE_CAP = 60

SEQ_CHANNELS = ["ret_1", "body", "upper_shadow", "lower_shadow", "amplitude", "log_amount_xs_rank", "amount_ratio_5",
                "close_to_ma_5", "close_to_ma_20", "drawdown_20", "limit_up_close", "broke_board", "limit_down_close",
                "limit_streak", "role_rel_ret_1_vs_sec"]
SECTOR_CHANNELS = ["sec_n", "sec_ret_median", "sec_up_ratio", "sec_limit_up", "sec_broke", "sec_max_streak",
                   "sec_limit_up_density", "sec_amount_share", "sec_rel_ret_1", "sec_ret_3", "sec_ret_5", "sec_ret_10",
                   "sec_rel_ret_3", "sec_rel_ret_5", "sec_rel_ret_10", "sec_amount_share_chg_5", "sec_limit_up_5",
                   "sec_strength_rank"]


def _std(x: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    """按通道标准化,统计量只来自训练期行;±inf / NaN → 0(标准化后即均值)。"""
    x = np.where(np.isfinite(x), x, np.nan)
    mu = np.nanmean(x[train_rows], axis=0)
    sd = np.nanstd(x[train_rows], axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (x - mu) / sd
    return np.nan_to_num(np.clip(z, -5, 5), nan=0.0).astype(np.float32)


@dataclass
class Tensors:
    dates: list
    codes: list
    seq: np.ndarray          # [n_dates, n_stocks, C],缺失为 0
    present: np.ndarray      # [n_dates, n_stocks] bool
    market: np.ndarray       # [n_dates, M]
    sector: np.ndarray       # [n_dates, 31, S]
    sector_ids: list
    stock_sector: np.ndarray  # [n_dates, n_stocks] 行业序号,-1 = 无
    peers: np.ndarray        # [n_dates, n_stocks, PEERS] 同业股票序号,-1 = 空
    peers_shuffled: np.ndarray


def build_tensors(feats: pd.DataFrame, train_end) -> Tensors:
    f = feats[feats["board"] == "main"].copy()
    dates = sorted(f["trade_date"].unique())
    codes = sorted(f["ts_code"].unique())
    di = {d: i for i, d in enumerate(dates)}
    si = {c: i for i, c in enumerate(codes)}
    f["_d"] = f["trade_date"].map(di)
    f["_s"] = f["ts_code"].map(si)
    train_rows = (f["trade_date"] <= train_end).to_numpy()

    raw = f[SEQ_CHANNELS].astype(float).to_numpy()
    raw[:, SEQ_CHANNELS.index("amount_ratio_5")] = np.log(np.clip(raw[:, SEQ_CHANNELS.index("amount_ratio_5")], 1e-3, None))
    z = _std(raw, train_rows)
    seq = np.zeros((len(dates), len(codes), len(SEQ_CHANNELS)), dtype=np.float32)
    present = np.zeros((len(dates), len(codes)), dtype=bool)
    seq[f["_d"], f["_s"]] = z
    present[f["_d"], f["_s"]] = ~f["is_suspended"].to_numpy()

    mk = f.groupby("trade_date")[MARKET_FEATURES].first().reindex(dates)
    mk_train = np.array([d <= train_end for d in dates])
    market = _std(mk.astype(float).to_numpy(), mk_train)

    sector_ids = sorted(f["sw_l1"].dropna().unique())
    sx = {s: i for i, s in enumerate(sector_ids)}
    sec = f.dropna(subset=["sw_l1"]).groupby(["trade_date", "sw_l1"])[SECTOR_CHANNELS].first()
    sec_arr = np.full((len(dates), len(sector_ids), len(SECTOR_CHANNELS)), np.nan)
    for (d, s), row in zip(sec.index, sec.to_numpy()):
        sec_arr[di[d], sx[s]] = row
    flat = sec_arr.reshape(-1, len(SECTOR_CHANNELS))
    flat_train = np.repeat(mk_train, len(sector_ids))
    sector = _std(flat, flat_train).reshape(sec_arr.shape)

    stock_sector = np.full((len(dates), len(codes)), -1, dtype=np.int64)
    has = f["sw_l1"].notna()
    stock_sector[f.loc[has, "_d"], f.loc[has, "_s"]] = f.loc[has, "sw_l1"].map(sx).to_numpy()

    # 同业:D 日同一行业成交额前 PEERS+1 名(之后去掉候选自己)
    peers = np.full((len(dates), len(codes), PEERS), -1, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    peers_sh = np.full_like(peers, -1)
    top = (f[has & ~f["is_suspended"]].sort_values(["_d", "sw_l1", "amount_cny"], ascending=[True, True, False])
           .groupby(["_d", "sw_l1"]).head(PEERS + 1))
    lists = {k: g["_s"].to_numpy() for k, g in top.groupby(["_d", "sw_l1"])}
    sectors_by_day: dict[int, list[str]] = {}
    for d, sname in lists:
        sectors_by_day.setdefault(d, []).append(sname)
    members_all = {k: g["_s"].to_numpy() for k, g in f[has].groupby(["_d", "sw_l1"])}
    for (d, sname), stocks in members_all.items():
        mine = lists.get((d, sname), np.array([], dtype=np.int64))
        others = [x for x in sectors_by_day.get(d, []) if x != sname]
        for s in stocks:
            p = mine[mine != s][:PEERS]
            peers[d, s, :len(p)] = p
            if others:
                o = lists[(d, others[rng.integers(len(others))])]
                q = o[o != s][:PEERS]
                peers_sh[d, s, :len(q)] = q
    return Tensors(dates, codes, seq, present, market, sector, sector_ids, stock_sector, peers, peers_sh)


class TemporalCNN(nn.Module):
    def __init__(self, channels: int, d: int = D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels + 1, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(64, d, 5, padding=8, dilation=4), nn.GELU(),
        )
        self.out = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [N, L, C+1]
        h = self.net(x.transpose(1, 2))                 # [N, d, L]
        return self.out(torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1))


class M2(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.enc = TemporalCNN(channels)
        self.mode = nn.Embedding(len(MODES), D_MODEL)
        self.head = nn.Sequential(nn.Linear(D_MODEL, 64), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(64, 1))

    def forward(self, b: dict) -> torch.Tensor:
        return self.head(self.enc(b["cand_seq"]) + self.mode(b["mode"])).squeeze(-1)


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D_MODEL, HEADS, dropout=DROPOUT, batch_first=True)
        self.n1 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(4 * D_MODEL, D_MODEL))
        self.n2 = nn.LayerNorm(D_MODEL)

    def forward(self, q, kv, mask):
        a, _ = self.attn(q, kv, kv, key_padding_mask=mask)
        q = self.n1(q + a)
        return self.n2(q + self.ff(q))


class M3(nn.Module):
    def __init__(self, channels: int, n_market: int, n_sector_feats: int, n_sectors: int, use_market: bool = True):
        super().__init__()
        self.use_market = use_market
        self.enc = TemporalCNN(channels)
        self.mode = nn.Embedding(len(MODES), D_MODEL)
        self.market = nn.Linear(n_market, D_MODEL)
        self.sector = nn.Linear(n_sector_feats, D_MODEL)
        self.sector_id = nn.Embedding(n_sectors + 1, D_MODEL)
        self.kind = nn.Embedding(4, D_MODEL)  # 0 市场 1 行业 2 同业 3 候选自身
        self.own_sector = nn.Embedding(2, D_MODEL)  # 行业 token 是否为候选所属行业
        self.blocks = nn.ModuleList([CrossBlock() for _ in range(LAYERS)])
        self.head = nn.Sequential(nn.Linear(D_MODEL, 64), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(64, 1))

    def forward(self, b: dict) -> torch.Tensor:
        n = b["cand_seq"].shape[0]
        cand = self.enc(b["cand_seq"])                                    # [N, d]
        peer = self.enc(b["peer_seq"].flatten(0, 1)).view(n, PEERS, -1)    # [N, P, d]
        ns = b["sector"].shape[1]
        sid = torch.arange(ns).expand(n, ns)
        own = (sid == b["cand_sector"].unsqueeze(1)).long()
        sec = self.sector(b["sector"]) + self.sector_id(sid) + self.own_sector(own) + self.kind.weight[1]
        toks = [sec, peer + self.kind.weight[2], (cand + self.kind.weight[3]).unsqueeze(1)]
        masks = [torch.zeros(n, ns, dtype=torch.bool), ~b["peer_ok"], torch.zeros(n, 1, dtype=torch.bool)]
        if self.use_market:
            toks.insert(0, (self.market(b["market"]) + self.kind.weight[0]).unsqueeze(1))
            masks.insert(0, torch.zeros(n, 1, dtype=torch.bool))
        kv, mask = torch.cat(toks, 1), torch.cat(masks, 1)
        q = (cand + self.mode(b["mode"])).unsqueeze(1)
        for blk in self.blocks:
            q = blk(q, kv, mask)
        return self.head(q.squeeze(1)).squeeze(-1)


def _windows(T: Tensors, d_idx: np.ndarray, s_idx: np.ndarray) -> np.ndarray:
    """取 [N, L, C+1] 的历史窗口(最后一维是「当日有行情」标记),只含 ≤ D 的日期。"""
    offs = np.arange(-L + 1, 1)
    dd = d_idx[:, None] + offs[None, :]
    ok = dd >= 0
    dd = np.clip(dd, 0, None)
    ss = np.broadcast_to(s_idx[:, None], dd.shape)
    x = T.seq[dd, ss] * ok[..., None]
    p = (T.present[dd, ss] & ok)[..., None].astype(np.float32)
    return np.concatenate([x, p], axis=-1)


def make_batch(T: Tensors, rows: pd.DataFrame, shuffled: bool = False, need_context: bool = True) -> dict:
    di = {d: i for i, d in enumerate(T.dates)}
    si = {c: i for i, c in enumerate(T.codes)}
    d_idx = rows["trade_date"].map(di).to_numpy()
    s_idx = rows["ts_code"].map(si).to_numpy()
    b = {"cand_seq": torch.from_numpy(_windows(T, d_idx, s_idx)),
         "mode": torch.from_numpy(rows["mode"].map({m: i for i, m in enumerate(MODES)}).to_numpy().copy())}
    if need_context:
        pe = (T.peers_shuffled if shuffled else T.peers)[d_idx, s_idx]           # [N, P]
        ok = pe >= 0
        pw = _windows(T, np.repeat(d_idx, PEERS), np.where(ok, pe, 0).reshape(-1))
        b["peer_seq"] = torch.from_numpy(pw.reshape(len(rows), PEERS, L, -1) * ok[..., None, None])
        b["peer_ok"] = torch.from_numpy(ok)
        b["market"] = torch.from_numpy(T.market[d_idx])
        b["sector"] = torch.from_numpy(T.sector[d_idx])
        b["cand_sector"] = torch.from_numpy(T.stock_sector[d_idx, s_idx])
    return b


def training_sample(train: pd.DataFrame) -> pd.DataFrame:
    return (train.sample(frac=1.0, random_state=SEED).groupby(["trade_date", "mode"]).head(PER_DAY_MODE_CAP)
            .sort_values(["trade_date", "mode", "ts_code"]).reset_index(drop=True))


def _target(rows: pd.DataFrame) -> np.ndarray:
    r = np.select([rows["mode"] == m for m in MODES], [rows[TARGETS[m]] for m in MODES], default=np.nan)
    return (r > ROUND_TRIP_COST).astype(np.float32)


def fit(kind: str, T: Tensors, train_rows: pd.DataFrame) -> nn.Module:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    C = T.seq.shape[2]
    if kind == "m2_tcnn":
        model: nn.Module = M2(C)
    else:
        model = M3(C, T.market.shape[1], T.sector.shape[2], len(T.sector_ids), use_market=(kind != "m3_no_market"))
    shuffled = kind == "m3_shuffled"
    need = kind != "m2_tcnn"
    y = _target(train_rows)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(len(train_rows))
        tot = 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            b = make_batch(T, train_rows.iloc[idx], shuffled, need)
            opt.zero_grad()
            loss = lossf(model(b), torch.from_numpy(y[idx]))
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        log.info("%s epoch %d loss %.4f", kind, ep + 1, tot / len(order))
    return model


@torch.no_grad()
def predict(kind: str, model: nn.Module, T: Tensors, rows: pd.DataFrame, chunk: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(rows), chunk):
        b = make_batch(T, rows.iloc[i:i + chunk], kind == "m3_shuffled", kind != "m2_tcnn")
        out.append(torch.sigmoid(model(b)).numpy())
    return np.concatenate(out) if out else np.array([])


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


DEEP_CONFIGS = ["m2_tcnn", "m3_xattn", "m3_shuffled", "m3_no_market"]
