"""证券分类:交易所 + 代码段 → 证券类型 / 板块;名称 → 风险警示 / 退市整理 / 上市首日标记。

代码段只用来在同一交易所内区分品种,交易所本身来自数据源的市场字段(通达信 sh / sz / bj 文件),
不是只看代码前缀。v0.1 没有逐日的官方板块字段,这一点写在 docs/DATA_AUDIT.md 的局限里。
"""
from __future__ import annotations

EXCHANGE_BY_TDX_MARKET = {"sh": "SSE", "sz": "SZSE", "bj": "BSE"}
SUFFIX = {"SSE": "SH", "SZSE": "SZ", "BSE": "BJ"}

# (交易所, 前缀) → (证券类型, 板块)。002 中小板 2021-04-06 并入深市主板,本项目窗口内都按主板。
_PREFIX_RULES: list[tuple[str, tuple[str, ...], str, str]] = [
    ("SSE", ("600", "601", "603", "605"), "A", "main"),
    ("SSE", ("688", "689"), "A", "star"),
    ("SSE", ("900",), "B", "b_share"),
    ("SZSE", ("000", "001", "002", "003", "004"), "A", "main"),
    ("SZSE", ("300", "301", "302"), "A", "chinext"),
    ("SZSE", ("200", "201"), "B", "b_share"),
    ("SZSE", ("399",), "index", "index"),
    ("BSE", ("43", "83", "87", "88", "92"), "A", "bse"),
]
# 上交所指数在通达信里有 000xxx(与深市股票同号)和 999xxx 两种编号,只在 SSE 内识别。
_SSE_INDEX_PREFIXES = ("000", "999")


def classify(exchange: str, code: str) -> tuple[str, str]:
    if exchange == "SSE" and code.startswith(_SSE_INDEX_PREFIXES):
        return "index", "index"
    for ex, prefixes, sec_type, board in _PREFIX_RULES:
        if ex == exchange and code.startswith(prefixes):
            return sec_type, board
    return "other", "other"


def is_risk_warning(name: str) -> bool:
    """ST / *ST / SST / S*ST 等风险警示简称。名称是当日快照里的简称,按日生效。"""
    return "ST" in name.upper()


def is_delisting_period(name: str) -> bool:
    """退市整理期简称带「退」(如「XX退」「退市XX」)。"""
    return name.endswith("退") or name.startswith("退市")


def is_listing_day_marker(name: str) -> bool:
    """上市首日简称以 N 开头;注册制新股前 5 日以 C 开头(交易所行情口径)。"""
    return len(name) > 1 and name[0] in ("N", "C") and not name[1].isascii()


def ts_code(exchange: str, code: str) -> str:
    return f"{code}.{SUFFIX[exchange]}"
