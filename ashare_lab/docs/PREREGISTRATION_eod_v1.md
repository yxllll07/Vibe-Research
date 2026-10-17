# 盲测预注册:eod_v1 / run 20261017T182447

写于揭盲之前(2026-10-17)。盲测期 2026-04-01 ~ 2026-10-16 在此之前**没有被任何规则推导、模型训练或配置挑选读取过**。

## 锁定对象

- 配置哈希 `82fee8f8cd3b80eb`,模型哈希 `3aa8eddc6ccd95f1`(对 models.pkl 文件字节取哈希)
- M0 规则 DATA_R0_v1(训练期 2024-10-08 ~ 2025-09-30 推导):
  - A:body(−)、close_to_ma_5(−)、ret_1(−)、ret_3(−)
  - B:role_amount_rank_in_sec(+)、amount_ratio_5(+)、close_pos_in_range(+)、role_rel_ret_5_vs_sec(−)
  - C:limit_up_count_20(−)、ret_20(−)、prev_streak(−)、ret_10(−)
  - D:limit_up_count_20(−)、close_to_ma_60(−)、ret_20(−)、ret_10(−)
  - E:close_to_ma_20(−)、role_amount_rank_in_sec(−)、amount_ratio_20(−)、prev_streak(−)
- M1:逻辑回归 / LightGBM × stock / full,固定参数与种子
- 评估全部 5 个配置,不在揭盲后挑最好的那个当结论

## 揭盲前的判断依据(验证期 2025-10-09 ~ 2026-03-31)

| 问题 | 验证期结果 | 揭盲时检验什么 |
|---|---|---|
| 信号:M0 池内排序有没有样本外预测力 | RankIC 全部为正;A t=6.7、E t=5.6、B t=3.1、D t=2.0、C t=1.6 | M0 在 A、E 的 RankIC 是否仍为正且 t ≥ 2 |
| H1:加市场 / 行业特征是否带来增量 | full 相对 stock 没有一致提升 | full 的 RankIC / 回测是否一致优于 stock |
| H3:信号能否在方案的交易约束下(每模式每天 1 只)变成净收益 | 5 个配置总收益全为负,日均收益 CI 全部跨 0 | 回测日均收益 CI 是否高于 0 |

## 预期

信号层面预期延续(A、E 的 RankIC 为正);交易层面预期仍不显著盈利;H1 预期不成立。
若盲测与预期相反,如实报告,不回头调参。揭盲后本盲测期不再作为合法的样本外检验,后续验证靠前向纸面交易(P4)。
