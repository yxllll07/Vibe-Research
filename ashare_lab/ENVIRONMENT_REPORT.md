# ENVIRONMENT_REPORT

生成时间:2026-10-17T16:37:53+08:00

## 机器

- Python 3.13.16 · Linux x86_64
- CPU 核数 4 · AVX2 有
- 内存 15.7 GB · 数据目录可用磁盘 31.0 GB
- GPU:不检测、不依赖(M0/M1 只用 CPU)

## 依赖

| 包 | 版本 |
|---|---|
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| pyarrow | 26.0.0 |
| duckdb | 1.5.6 |
| pyyaml | 6.0.3 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| torch | 未安装 |
| akshare | 1.18.94 |
| requests | 2.34.2 |

## 网络

| 数据源 | 结果 |
|---|---|
| 通达信盘后包 | 可达(HTTP 403) |
| 新浪交易日历 | 可达(HTTP 200) |
| 东财涨停池 | 可达(HTTP 404) |
| Tushare | 可达(HTTP 200) |

## 凭据

- TUSHARE_TOKEN:未设置(v0.1 不需要)
- baostock:走自有 TCP 协议,云端代理环境下不可用,v0.1 不依赖
