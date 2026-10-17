# 实现状态

## 已完成

| 步骤 | 内容 | 验证 |
|---|---|---|
| S1 | 包结构、配置、CLI(`python -m ashare_lab`)、日志、环境报告、数据根目录 `.local/ashare_lab` | `env-report` 实跑 |
| S2 | 按日期生效的涨跌幅(含 2026-07-06 主板 ST 5%→10%)与费率;日频代理成交(T+1、一字 / 开盘即涨停不成交、滑点越界、一字跌停止损顺延、跳空按开盘);按时点视图;竞价 schema(白名单);冻结票池(只增不改 + 二阶段过滤);切分 purge / embargo、训练期标准化器、盲测锁;板块成员按时点查询 | 12 条红线测试 |
| S3 | 新浪交易日历;通达信盘后包逐日拉取(原始 zip + sha256 清单,可续传)、归一化;质量报告(覆盖率、价格带越界、前收连续性、股票池结构、与东财涨停池交叉核对) | 见 `DATA_AUDIT.md` |
| S4 | 申万 2021 行业个股归属及变迁(随包附中间证书,证书校验不关);按时点映射一级行业;行业强度 / 涨停密度 / 扩散 / 相对强度 / 成交额占比;个股行业内角色;`build` 一键生成特征表与标签表 | 单元测试 + 实跑 |
| S5 | 五模式基础池(方案的结构定义);M0 规则由训练期逐日 RankIC 推导(DATA_R0_v1,等权分位、无拟合系数);每模式最多 2 只;主动槽 A/B/D 取一、核心槽 C/E;`freeze` 写冻结票池 JSON | 单元测试 + 实跑 |
| S6 | M1:逻辑回归 / LightGBM,stock vs full 特征消融;RankIC(t)、五分组价差、AUC、Precision@1;代理回测(费用、槽位、未成交原因);分块自助法置信区间;预注册 → 锁定 → 盲测只评估一次 | 已完成,见 `RESULTS_eod_v1.md` |
| P4 | 前向纸面交易起点:2026-10-16 信号 → 2026-10-19 的冻结票池已用 DATA_R0_v1 生成 | `.local/ashare_lab/artifacts/pools/20261019/frozen.v1.json` |

## 命令顺序

```bash
python -m ashare_lab calendar
python -m ashare_lab sectors
python -m ashare_lab ingest
python -m ashare_lab quality --crosscheck-date 2026-10-16
python -m ashare_lab build
python -m ashare_lab experiment          # 训练期推导规则与拟合,验证期评估,不碰盲测期
python -m ashare_lab lock && python -m ashare_lab blind   # 预注册后锁定,盲测只评估一次
python -m ashare_lab freeze --signal-date 2026-10-16 --score m0   # 盲测里表现最好的是 M0
```

## 待办

1. **P4 前向纸面交易**:每天收盘后 ingest 当日包 → build → freeze → 次日复盘;同时拉东财涨停 / 炸板池交叉核对。
   需要一台常驻的机器(云容器会被回收)。
2. **v2(事后提出,只用 2026-10-19 以后的前向数据评估)**:E、B 池内前 N 只等权;暂停 C;重新设计退出。
3. 可选外部数据:历史股本(换手率 / 市值)、Tushare `stk_auction`(9:25 竞价)、分钟数据(`tradable_minute`)。
