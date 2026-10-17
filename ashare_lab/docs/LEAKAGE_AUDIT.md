# 泄漏审计

## 时钟(北京时间)

| 时点 | 用途 | 代码 |
|---|---|---|
| D 日 15:00 | 日线 event_time | `data/tdx.py` |
| D 日 20:00 | 日线 available_at(保守假设:通达信包当日几小时后才发布,没有官方时间戳) | 配置 `data.tdx_available_at_local_time` |
| D 日 21:00 | 盘后冻结候选的决策时刻 | `data/pit.py: d_postclose` |
| T 日 9:14:59 | 冻结截止 | `candidates/freeze.py` |
| T 日 9:25 / 9:26 | 竞价 underlying_event_at / 最早可取 | `data/auction.py` |

`AsOfView` 只返回 `available_at <= 决策时刻` 的行;点名要的数据尚不可见时抛 `LeakageError`。

## 逐类特征

| 特征 | 用到的数据 | 最晚数据日 | 可用时刻 |
|---|---|---|---|
| 个股收益、K 线形态、成交额比、均线偏离、回撤(`features/stock.py`) | 该股 ≤ D 的日线 | D | D 20:00 |
| 涨跌停事件、连板数、涨停 / 炸板计数(`features/events.py`) | ≤ D 的不复权价 + 当日前收 + 按日期规则 | D | D 20:00 |
| 截面分位 | 同一交易日 D 的全部股票 | D | D 20:00 |
| 市场广度、梯队、晋级率(`features/market.py`) | D 日全部主板非 ST 股票 | D | D 20:00 |
| 开板次数、首封时间 | — | — | UNKNOWN(日线不可得,不填 0) |
| 换手率、市值 | — | — | v0.1 关闭 |

滚动窗口全部向后看(`rolling` 只含当前与过去);收益用 `close / prev_close` 逐日连乘,不用今天下载的前复权序列。
测试 `test_no_forward_core_role` 改写 D 之后的行情,断言 D 日及以前的所有特征不变。

## 标签

标签在独立的 `labels.py`,键为 `(signal_date, ts_code)`;y_A / y_B / y_2to3 / r_h* / t_one_word_up 都只来自 T 日及以后。

## 切分

- 训练 / 验证 / 盲测按日期切;标签窗口跨入下一期的样本清除(purge),每期开头 embargo 个交易日清除。
- 标准化器只能用训练期样本拟合(`TrainOnlyScaler`),混入训练期之后的日期直接报错。
- 盲测标签只有在模型哈希与配置哈希锁定后可读(`BlindTestLock`);锁定后改动任一哈希,盲测作废。

## 执行限制(proxy_daily_only)

一字涨停 / 开盘即涨停不算买到;滑点越过涨停价不成交;T+1;一字跌停止损顺延,不按理论止损价卖出;
跳空低开按开盘价成交。所有结果标注「不代表 9:25 之后可成交」。
