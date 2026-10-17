# A-Structure Lab(A 股市场结构学习系统)

研究 / 模拟交易系统:**不自动下单,不承诺收益**。研究对象是沪深主板非风险警示股票,
目标是在严格按时点还原的逐日回测里,检验「加入板块 / 全市场关系」相对「只看个股」有没有真实增量。

完整研究设计见使用者提供的《研究设计与技术实施方案》;本目录的取舍与现状见:

| 文档 | 内容 |
|---|---|
| [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) | 可行性评估、两年窗口与免费数据源的决定 |
| [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md) | 逐数据源:覆盖、单位、已做验证、缺失 |
| [`docs/LEAKAGE_AUDIT.md`](docs/LEAKAGE_AUDIT.md) | 逐类特征的 available_at 依据、切分与执行限制 |
| [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) | 已完成 / 待办 / 下一步 |

## 和 Vibe-Research 的关系

独立的 Python 包,与 Vibe-Research 同仓。复用两样东西:
- 取数层里已实测的通达信盘后包解析器(`.agents/skills/data-access/scripts/sources/astock_ported.py`);
- 东财涨停池等当日接口,用于交叉核对和将来的前向纸面交易。

研究线程里的 Agent 大脑、编排器与本包无关;本包不改动 Vibe-Research 的任何代码。

## 环境

Python ≥ 3.11,复用仓库的 `.venv`:

```bash
scripts/setup                                   # 首次:建 .venv、装取数层依赖
.venv/bin/pip install -r ashare_lab/requirements.txt
```

只用 CPU;不需要 GPU、AVX2 或 Tushare token。

## 命令

在仓库根目录运行(`python` 指 `.venv/bin/python`):

```bash
python -m ashare_lab env-report                 # 写 ashare_lab/ENVIRONMENT_REPORT.md
python -m ashare_lab calendar                   # 交易日历(新浪)
python -m ashare_lab sectors                    # 申万 2021 行业个股归属及变迁
python -m ashare_lab ingest                     # 通达信盘后包 2024-07-01 ~ 2026-10-16,可断点续传
python -m ashare_lab quality --crosscheck-date 2026-10-16   # 覆盖率 / 一致性 / 与东财涨停池核对
python -m ashare_lab build                      # 特征表 + 标签表(分开存)
python -m ashare_lab experiment                 # M0 / M1 训练 + 验证期评估与代理回测(不碰盲测期)
python -m ashare_lab freeze --signal-date 2026-10-16        # D 日冻结票池 JSON
python -m pytest ashare_lab/tests -q            # 红线与单元测试(合成夹具)
```

数据默认在 `.local/ashare_lab/`(已被 git 忽略;环境变量 `ASHARE_LAB_DATA` 可改):

```text
raw/manifest.jsonl              每个原始文件的 url / sha256 / 抓取时间,只追加
raw/tdx_package/YYYY/*.zip      通达信原始包,不覆盖
normalized/daily_bar/YYYY/*.parquet
normalized/calendar.parquet
reports/                        拉取报告、质量报告、逐月覆盖率
```

## 目录

```text
ashare_lab/
  configs/eod_v1.yaml     实验配置(切分、股票池、执行口径)
  data/                   原始区、日历、通达信适配、按时点读取、竞价 schema、质量报告
  universe/               历史股票池(含已退市)、板块成员按时点查询
  rules/                  按日期生效的涨跌幅与费率、日频代理成交(T+1 / 一字 / 跌停)
  features/               涨跌停事件、个股特征、市场广度与连板梯队
  labels.py               标签(独立于特征)
  candidates/modes.py     五模式入池条件与 M0 打分(占位规则,待 R1 替换)
  candidates/freeze.py    D 日冻结票池、T 日 9:25 二阶段过滤
  evaluation/split.py     切分、purge / embargo、训练期标准化器、盲测锁
  evaluation/models.py    M1:逻辑回归 / LightGBM,stock vs full 消融
  evaluation/proxy_backtest.py   日频代理回测(不代表可成交)
  evaluation/experiment.py       训练、验证、报告
  tests/                  12 条红线 + 单元测试
```

## 局限(v0.1)

- 日线只能判断收盘封板与盘中触板;开板次数、首封时间为 UNKNOWN。
- 没有历史股本:换手率、流通市值类特征整体关闭。
- 板块:只用申万 2021 一级行业;概念板块没有历史快照证据前不使用。
- 五个模式的入池与退出是**占位规则**,不是使用者的 R1;拿到 R1 前的收益数字只用于验证管线。
- 9:25 竞价只有 schema 与过滤接口,没有数据源;缺数据时输出 `NO_AUCTION_DATA`。
- 代理回测(`proxy_daily_only`)只检验信号方向,**不代表 9:25 之后真的能成交**。
