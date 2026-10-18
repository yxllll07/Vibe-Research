# A 股主板「D 日预期 → T 日开盘买入」交易系统:完整复现说明

> 给另一个 Agent(如 Codex)看的规格书。按本文从零实现,应能逐日得到与参考实现相同的情绪阶段、主线、龙头、候选、开关表和选票。
> 参考实现在本仓库 `ashare_lab/`(Python 3.12,pandas / numpy / duckdb / lightgbm 不需要)。文中每节都标了对应源文件。
> 研究 / 纸面用途,不是投资建议。

---

## 0. 一句话

**D 日收盘后**,用只截至 D 日的数据判断情绪阶段、主线和龙头,按五个模式生成候选;再用**只来自过去、已经收盘的交易结果**学出「什么阶段做什么模式」的开关表,冻结一个最多 10 只的预选池。
**T 日 9:25** 只看池内股票的集合竞价,跳过竞价已到涨停价或停牌的,**取第一只在开盘买入**。
**成功 = T 日收盘价 > T 日开盘价 × 1.0015**(当天收涨,0.15% 约等于双边费用)。不持仓过夜的部分不在本系统评价范围内。

---

## 1. 数据(`data/tdx.py`、`data/sws.py`、`data/calendar.py`)

| 数据 | 来源 | 用法 |
|---|---|---|
| 逐日全市场不复权日线 + 当日简称 | 通达信官网盘后包 `https://www.tdx.com.cn/products/data/data/g4day/{YYYYMMDD}.zip` | 每个交易日一包,含沪深北全部证券;字段:market(sh/sz/bj)、code、name、prev_close、open、high、low、close、volume(股)、amount(元) |
| 交易日历 | akshare `tool_trade_date_hist_sina` | 判断 D → T(下一交易日) |
| 申万 2021 行业归属及变迁 | `https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls` | 每行(股票、计入日期、6 位行业码);站点缺中间证书,随包附 `data/certs/` 的 DigiCert 中间证书,**不关闭 TLS 校验** |
| T 日 9:25 竞价(实盘) | 腾讯行情 `https://qt.gtimg.cn/q=sh600000,sz000001,…` | 只请求预选池里的代码 |

窗口:2024-07-01 起(参考实现到 2026-10-16)。所有数据追加写入原始区并记 sha256,不覆盖。

### 1.1 证券分类(`data/securities.py`)
- 交易所来自盘后包的 market 字段:sh → SSE,sz → SZSE,bj → BSE。
- **主板 A 股**:SSE 的 600/601/603/605 开头;SZSE 的 000/001/002/003/004 开头。SSE 的 000/999 开头是指数。
- 风险警示:简称(大写)含 `ST`。退市整理:简称以「退」结尾或以「退市」开头。上市首日标记:简称首字为 N 或 C 且第二个字符非 ASCII。
- 停牌:`volume == 0`;停牌行的 open / high / low 若 ≤ 0 置为缺失(盘后包里记 0)。
- 代码格式 `600000.SH` / `000001.SZ`。

### 1.2 申万行业按时点(`data/sws.py: daily_l1`)
- `sw_l1 = "SW1_" + 行业码前 2 位`,`sw_l2 = "SW2_" + 行业码前 4 位`。
- effective_from = 计入日期;effective_to = 同一股票下一条记录的计入日期。
- 对面板每行 (D, 股票) 用 `merge_asof(backward)` 找 effective_from ≤ D 的最近一条,且 D < effective_to(或无 effective_to);查不到为缺失。

---

## 2. 涨跌停与事件(`rules/limits.py`、`features/events.py`)

- 涨跌幅比例(按 D 日生效):主板普通 ±10%;主板风险警示 **2026-07-06 前 ±5%,之后 ±10%**;创业板 / 科创板 ±20%;北交所 ±30%;注册制新股上市前 5 个交易日无限制;退市整理期首日无限制。
- **涨停价 = round_half_up(prev_close × (1 + 比例), 0.01)**,跌停价同理;prev_close 用盘后包里交易所当日前收(除权日为除权参考价),**不能用上一日收盘或复权价**。用 Decimal 计算。
- 价格比较一律 `abs(a - b) < 1e-6`。未停牌的行才有事件:
  - `limit_up_close`:close == 涨停价;`limit_down_close`:close == 跌停价
  - `touched_up`:high == 涨停价;`broke_board`(炸板):touched_up 且非 limit_up_close
  - `one_word_up`(一字涨停):open == 涨停价 且 low == 涨停价
  - `limit_streak`:按该股自己的交易行序列,连续 limit_up_close 的计数(非涨停日为 0)

---

## 3. 个股特征(`features/stock.py`,只用 ≤ D 的行,窗口都向后看)

按 (ts_code, trade_date) 排序,同一股票内按该股在面板里的行滚动(盘后包收录的停牌行也算一行):
- `ret_1 = close / prev_close − 1`;`adj_index = exp(cumsum(log1p(ret_1)))`;`ret_n = adj_index / adj_index.shift(n) − 1`(n = 20 等)
- `close_pos_in_range = (close − low) / (high − low)`,high == low 时取 0.5
- `amount_ratio_5 = amount / mean(amount, 最近 5 行含当日)`(满 5 行才有值)
- `close_to_ma_20 = adj_index / mean(adj_index, 20 行含当日) − 1`;`close_to_ma_60` 同理(满窗口才有值)
- `prev_limit_up = limit_up_close.shift(1)`(缺失记 False);`prev_streak = limit_streak.shift(1)`(缺失记 0)

### 3.1 研究股票池 `eligible`(`universe/eligibility.py`)
沪深主板 A 股 且 非风险警示 且 非退市整理 且 未停牌 且 不在上市前 5 个交易日(上市日 = 窗口内首次出现且带上市首日标记的那天;窗口开始时已存在的视为老股)。

---

## 4. 情绪周期(`structure/sentiment.py`)

**统计口径**:当日 board == main、非风险警示、非退市整理、未停牌的股票(用全部这些行,不只 eligible)。

每日指标:
| 指标 | 定义 |
|---|---|
| n_limit_up | limit_up_close 家数 |
| n_limit_down | limit_down_close 家数 |
| broke_rate | 炸板家数 /(涨停家数 + 炸板家数);分母 0 为缺失 |
| yday_lu_ret | prev_streak ≥ 1 的股票当日 ret_1 均值(昨日涨停今日表现) |
| promo_1to2 | prev_streak == 1 的股票中当日收盘涨停的比例 |
| max_height | 当日 limit_streak 最大值(无涨停为 0) |
| high_loss | prev_streak ≥ 2 且 ret_1 ≤ −5% 的家数 |

**情绪分**:对 7 个部件 `[(yday_lu_ret,+1), (n_limit_up,+1), (promo_1to2,+1), (broke_rate,−1), (n_limit_down,−1), (high_loss,−1), (max_height,+1)]`,
先乘方向符号,再算**滚动分位**:窗口 = 过去 60 个交易日(含当日),去掉缺失后不足 20 个为缺失,否则 = 窗口内 ≤ 当日值的比例。
`score = 7 个分位的均值(忽略缺失)`。另算 `pct_n_limit_up`(n_limit_up 的分位)、`pct_n_limit_down_raw`、`pct_high_loss_raw`(原值、不乘符号的分位)。

**阶段**:按日期顺序逐日判定,先命中先用。prev = 前一日已判定的阶段(第一天视为「震荡」),ps / pps = 前一日 / 前两日 score:
1. score 缺失 → 震荡
2. score ≥ 0.80 且 pct_n_limit_up ≥ 0.80 → **高潮**
3. prev ∈ {高潮, 发酵} 且(ps − score ≥ 0.20 或 pct_n_limit_down_raw ≥ 0.85 或 pct_high_loss_raw ≥ 0.85)→ **分歧**
4. score ≤ 0.20 → **冰点**
5. score < 0.45 且(score < ps < pps,或 prev ∈ {分歧, 退潮})→ **退潮**
6. prev ∈ {冰点, 退潮} 且 score > ps → **修复**
7. score ≥ 0.55 → **发酵**
8. 其余 → **震荡**

---

## 5. 主线与龙头(`structure/theme.py`)

口径:board == main、非风险警示、非退市整理的行(含停牌行,但组级统计只用未停牌且 sw_l2 非缺失的行)。

### 5.1 主线分(每日在各二级行业之间比较)
组级表(sw_l2 × 日,只用未停牌行):`lu` = 组内收盘涨停数,`ret` = 组内 ret_1 等权均值,`max_streak` = 组内最大 limit_streak。按组内日期排序(shift / rolling 按该组的行):
- `heat = lu + 0.7 × lu.shift(1) + 0.5 × lu.shift(2)`(缺失补 0)
- `persist` = 最近 5 行中 `lu ≥ 2` 的天数(min_periods = 1)
- `ret3 = expm1(最近 3 行 log1p(ret) 之和)`(min_periods = 1)
- `ml_score = 0.35·pr(heat) + 0.20·pr(persist) + 0.20·pr(max_streak) + 0.25·pr(ret3)`,pr = 当日各组之间的百分位排名(pandas `rank(pct=True)`,并列取平均)
- `ml_rank` = 当日 ml_score 降序名次(并列按出现顺序,`method="first"`)
- **主线 `is_mainline` = ml_rank ≤ 3 且 heat ≥ 4**;个股继承所在 sw_l2 当日的 ml_score / ml_rank / is_mainline / grp_ret(= 组 ret)

### 5.2 龙头分(主线内外都算,在同一 sw_l2 当日之间比较)
个股按自己的行序列:`lu5` / `lu10` = 最近 5 / 10 行涨停次数(min_periods = 1);`anti` = 当日涨停且 grp_ret < 0;`anti5` = 最近 5 行 anti 次数;
`days_since_first_lu10` = 最近 10 行(含当日)内**第一次**涨停距今的行数;`height` = limit_streak > 0 时取 limit_streak,否则取 prev_streak;
`amt_rank_grp` = 当日同 sw_l2 内成交额百分位(rank pct=True)。
- 候选 = lu5 > 0 的股票;在 (日, sw_l2) 的候选之间:
  `leader_score = 0.35·pr(height) + 0.25·pr(lu10) + 0.20·pr(anti5) + 0.10·pr(days_since_first_lu10) + 0.10·amt_rank_grp`
- `leader_rank` = 降序名次(method="first");`role`:第 1 名「龙头」,2 ~ 3 名「前排」,其余候选「跟风」,非候选为空串
- `is_space_leader`(空间龙)= 未停牌 且 limit_streak == 当日全部未停牌行的最大连板 且 该最大值 ≥ 2

### 5.3 合并成面板(`system/strategy_v2.py: load_panel`)
特征表 ⟕ 主线龙头表 ⟕ 情绪表(stage、score 记为 sent_score)。is_mainline / is_space_leader 缺失记 False,role 缺失记空串。
按股票行序列 shift(1) 得 `prev_role`、`prev_leader_score`、`prev_is_space_leader`、`prev_is_mainline`。

---

## 6. 五个模式的候选(`candidates/modes_v2.py: candidates`,全部只用 D 日数据)

记 e = eligible,ml = is_mainline,`pct_d(x)` = 该模式入池股票之间当日的百分位排名。

| 模式 | 入池条件 | 模式分 score |
|---|---|---|
| **A 首板预判** | e 且 ml 且 非 touched_up 且 lu5 == 0 且 ret_1 > 0 且 close_pos_in_range ≥ 0.6 且 amount_ratio_5 ≥ 1.2 且 ret_20 < 0.30 | 0.4·ml_score + 0.3·amt_rank_grp + 0.3·pct_d(ret_1) |
| **B 首板接力** | e 且 ml 且 limit_streak == 1 | 0.45·ml_score + 0.35·amt_rank_grp + 0.2·(非一字涨停 ? 1 : 0) |
| **C 龙头首次分歧** | e 且 [(prev_role == 龙头 且 prev_is_mainline 且 prev_streak ≥ 2) 或 (prev_is_space_leader 且 prev_streak ≥ 3)] 且 非 limit_up_close 且 非 limit_down_close 且 ret_1 > −7% | prev_leader_score(缺失记 0) |
| **D 分歧转强** | e 且 ml 且 role ∈ {龙头, 前排} 且 [broke_board 或 (prev_limit_up 且 ret_1 < 0 且 非 limit_down_close)] | 0.5·leader_score(缺失 0) + 0.5·ml_score |
| **E 主线中军** | e 且 ml_rank ≤ 5 且 close_to_ma_20 > 0 且 close_to_ma_60 > 0 且 0.05 ≤ ret_20 ≤ 0.50 且 amt_rank_grp ≥ 0.8 且 非 limit_up_close | 0.4·ml_score + 0.4·amt_rank_grp + 0.2·pct_d(ret_20) |

一只票可同时属于多个模式,每个 (股票, 模式) 一行,带上 D 日的 stage。

---

## 7. 开关表:只用过去的结果学(`evaluation/walkforward.py`)——当前规则的核心

### 7.1 每个候选的 T 日结果
T = D 的下一交易日(按该股行序列的下一行)。`buyable` = T 日 open > 0 且 open ≠ T 日涨停价;
`win` = T_close / T_open − 1 > 0.0015;`t_oc` = T_close / T_open − 1。
**同日全市场基准**:D 日 eligible 且 T 日 buyable 的全部股票,当日 win 的比例 `mkt_win`。

### 7.2 学习(截止日 cutoff)
1. 只用 `T 日 < cutoff` 的行(T 日收盘已经发生);**出现任何 T 日 ≥ cutoff 的行必须报错**,不能静默丢弃。窗口 = 从数据起点到截止日之前的全部历史(扩张窗口)。
2. 取 buyable 的候选,按 score 降序,每个 (D 日, 模式) 取第 1 名。
3. 按 (stage, 模式) 汇总:`days` = 天数,`win` = 收涨率。`base` = 历史上各日 mkt_win 的平均。
4. **放行条件:days ≥ 20 且 win ≥ base + 0.03**(这两个门槛是预先定的,不要按结果调)。
5. 每个阶段放行的模式按 win 从高到低排成优先级。

### 7.3 出预选池并选票(单个信号日 D)
1. 取 D 日全部候选中 (stage, 模式) 被放行的行;按 (模式优先级, score 降序, ts_code 升序) 排序;同一只股票只保留第一次出现;取前 10 只,依次编号 order = 1 … 10。
2. T 日 9:25:按 order 遍历,竞价缺失(停牌)或竞价价 ≥ 涨停价 − 1e-6 的跳过,**第一只可买的开盘买入**(每天最多 1 只)。阶段没有放行模式 → 空仓。

### 7.4 两种运行方式
- **逐月前推评估**(`walk_forward`):每个月的 cutoff = 当月第一个交易日,开关表每月重学一次,用于该月所有信号日。
- **实盘**(`system/strategy_v3.py: write_plan`,`--rules wf` 默认):每晚用 ≤ D 的面板重学,cutoff = D 的下一个自然日(即包含 T 日 ≤ D 的全部结果);把开关表、预选池写成 `plan.vN.json`,只增不改。

---

## 8. 每天怎么跑(北京时间)

| 时间 | 动作 | 命令(参考实现) |
|---|---|---|
| D 日 17:00 ~ 20:00 盘后包发布后 | 补日历和盘后包 → 重建特征 → 重算情绪 / 主线 / 龙头 | `python -m ashare_lab ingest --start D --end D`、`build`、`structure` |
| D 日 21:00 前 | 学开关表、冻结预选池 | `python -m ashare_lab plan-v3 --date D` |
| T 日 9:26 ~ 9:29 | 只拉池内腾讯行情:字段按 `~` 分隔,`f[5]` 今开(9:25 后即竞价价),为 0 时用 `f[3]`;涨停价用 D 日 close 按第 2 节算;取第一只可买的 | `python -m ashare_lab auction-v3 --date T` |
| T 日 9:30 起 | 买入;收盘记录是否收涨 | — |

竞价记录必须是 9:25 最终撮合(`underlying_event_at` = 9:25,`observed_available_at` ≥ 9:26 且同日),不接受 9:20 ~ 9:25 的过程数据(`data/auction.py` 白名单)。

---

## 9. 复现自检(参考实现的输出,逐项对上才算复现成功)

**情绪阶段**(D 日):2026-06-22 高潮,06-23 分歧,06-24 震荡,06-25 发酵,06-26 分歧,06-29 退潮,06-30 修复,07-01 发酵,07-02 分歧,07-22 震荡,07-23 高潮,07-24 分歧,10-15 退潮,10-16 退潮。
- 07-23 的指标:涨停 104、跌停 2、炸板率 0.181、昨日涨停今日 +4.3%、最高板 6(立新能源)、情绪分 0.933。

**主线**(D = 2026-07-23):SW2_6308(当日 23 只涨停,龙头长缆科技 3 板)、SW2_2403(龙头盛达资源)、SW2_2202(龙头金牛化工)。

**逐月前推**(`walkforward --start 2025-04-01 --end 2026-10-15`):信号日 373 天,出手 **259** 天,当天收涨率 **56.37%**,平均开→收扣 0.15% 后 **+0.70%**,同期全市场 47.07%。
分段:2025-04 ~ 09 共 84 天 53.6%;2025-10 ~ 2026-03 共 72 天 56.9%;2026-04 ~ 10 共 103 天 58.3%。
按 (阶段-模式) 的笔数 / 收涨率:震荡-B 58 / 65.5%,分歧-B 65 / 63.1%,退潮-E 38 / 63.2%,修复-B 20 / 55.0%,发酵-E 40 / 47.5%,发酵-B 21 / 38.1%。

**2026-10 月份学到的开关表**(base 0.475):修复 → D、B、E;分歧 → B;发酵 → E、C;退潮 → E、C;震荡 → B、D;高潮 → C;冰点 → 无。

**实盘计划**(D = 2026-10-16,退潮,放行 E、C):1 新经典 603096.SH(E),2 上海雅仕 603329.SH(E),3 海螺水泥 600585.SH(E),4 塔牌集团 002233.SZ(E),5 慈文传媒 002343.SZ(C)。

---

## 10. 必须遵守的纪律

1. **未来数据一律无效**:特征只用 ≤ D;开关表只用 T 日已收盘的结果;发现越界直接报错。
2. 冻结的计划只增不改;T 日只能在冻结名单内取舍,不能加票。
3. 竞价到涨停价的票买不到,必须跳过;一字板不能当作买到。
4. 涨跌停价必须基于交易所前收、按日期生效的比例、Decimal 四舍五入到分。
5. 不要按回测结果调整门槛(20 天、+3 个百分点、0.15%)、模式条件或权重;要改就从改动日起重新前推评估,并说明改动前看过哪些数据。

## 11. 已知局限

- **成交价理想化**:日线开盘价 = 9:25 竞价价;实盘看到竞价后只能在 9:30 之后按连续竞价价格买入,可能更差。
- 模式条件、情绪阈值是作者在 2026-10 复盘过 2026-06 ~ 07 几天后写的,定义层面没有完全隔离;真正的检验是 2026-10-19 之后的前向记录。
- 每段只有 70 ~ 100 笔,收涨率 95% 区间约 ±6 个百分点。
- 行业用申万二级近似题材(无历史概念板块快照)。
- 历史版本:`--rules v3` / `--rules v4` 为手写开关表(看过全样本,回测约 60%,含事后偏差),仅作对照,说明见 `docs/SYSTEM_v3.md`。
