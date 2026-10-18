"""资金持续接力探针:个股层面用 D 日数据(以及 9:25 竞价跳空)预测「T 开盘买、T+1 收盘卖仍赚钱」。

逐季前推:每季初用卖出日 < 季初的全部可交易主板样本训练 LightGBM,给本季打分;每日取分数最高的票。
结论见 docs/REALIZABLE.md 第 8 节:只用 D 日数据 AUC ≈ 0.51(找不到接力票);加竞价跳空后每日第一名
5 年每年为正(平均 +0.50%,开盘价成交、未计滑点)。运行:python -m ashare_lab.evaluation.relay_probe <输出目录>
"""
# mypy: ignore-errors
import sys
from pathlib import Path

import pandas as pd, numpy as np, lightgbm as lgb, json, time
from datetime import date, timedelta
from sklearn.metrics import roc_auc_score
from ashare_lab.config import load_config
from ashare_lab.system.strategy_v2 import load_panel
from ashare_lab.features.stock import STOCK_FEATURES
from ashare_lab.features.sector import SECTOR_FEATURES
from ashare_lab.features.market import MARKET_FEATURES
from ashare_lab.evaluation import walkforward as wf


def main(out_dir: Path) -> None:
    S = str(out_dir) + '/'
    t0=time.time()
    p=load_panel(load_config())
    p['d_gap']=p['open']/p['prev_close']-1; p['d_oc']=p['close']/p['open']-1
    ST=['冰点','修复','发酵','高潮','分歧','退潮','震荡']
    p['stage_i']=p['stage'].map({s:i for i,s in enumerate(ST)}); p['role_i']=p['role'].map({'龙头':3,'前排':2,'跟风':1}).fillna(0)
    p['prev_role_i']=p['prev_role'].map({'龙头':3,'前排':2,'跟风':1}).fillna(0)
    TH=['ml_score','ml_rank','heat','lu5','lu10','anti5','height','amt_rank_grp','leader_score','role_i','prev_role_i','sent_score','stage_i','is_mainline','is_space_leader']
    F=list(dict.fromkeys(STOCK_FEATURES+SECTOR_FEATURES+MARKET_FEATURES+TH+['d_gap','d_oc']))
    p=p[p.board=='main']
    keep=list(dict.fromkeys(['trade_date','ts_code','eligible','open','high','low','close','up_limit','down_limit']+F))
    p=p[keep].copy()
    for col in p.columns:
        if p[col].dtype=='float64': p[col]=p[col].astype('float32')
    import gc; gc.collect()
    print('slim panel',p.shape,round(p.memory_usage(deep=True).sum()/1e9,2),'GB',flush=True)
    nb=wf._next_bar(p)
    e=nb[nb.eligible.astype(bool)&nb.t_date.notna()].copy()
    e['buyable']=(e.t_open>0)&((e.t_open-e.t_up).abs()>1e-6)
    e=e[e.buyable].reset_index(drop=True)
    e=e[['trade_date','ts_code','t_date','t_open','t_close','close','buyable']+F].reset_index(drop=True)
    del nb; gc.collect()
    e=wf._realized(e,p[['trade_date','ts_code','open','high','close','down_limit']])
    del p; gc.collect()
    e=e[e.x_date.notna()].reset_index(drop=True)
    e['gap']=e.t_open/e.close-1
    e['y']=(e.r_close1>0).astype(int)
    e['t_oc']=e.t_close/e.t_open-1
    print('rows',len(e),'build',round(time.time()-t0),'s',flush=True)
    X=lambda df,cols: df[cols].astype(float).replace([np.inf,-np.inf],np.nan)
    P=dict(n_estimators=250,learning_rate=0.05,num_leaves=31,min_child_samples=500,subsample=0.5,subsample_freq=1,colsample_bytree=0.7,random_state=7,verbose=-1)
    qs=[date(y,m,1) for y in range(2022,2027) for m in (1,4,7,10) if date(2022,4,1)<=date(y,m,1)<=date(2026,10,1)]
    out=[]
    for i,q in enumerate(qs):
        q_end=qs[i+1] if i+1<len(qs) else date(2026,10,17)
        tr=e[(e.x_date<q)]; te=e[(e.trade_date>=q)&(e.trade_date<q_end)]
        if te.empty: continue
        m1=lgb.LGBMClassifier(**P).fit(X(tr,F),tr.y)
        m2=lgb.LGBMClassifier(**P).fit(X(tr,F+['gap']),tr.y)
        te=te.assign(s_d=m1.predict_proba(X(te,F))[:,1],s_g=m2.predict_proba(X(te,F+['gap']))[:,1])
        out.append(te[['trade_date','ts_code','y','r_close1','t_oc','gap','s_d','s_g','is_mainline','limit_streak']])
        print(q,len(tr),len(te),round(time.time()-t0),'s',flush=True)
    o=pd.concat(out); o.to_parquet(S+'relay_pred.parquet')
    o['yr']=o.trade_date.astype(str).str[:4]
    print('\nAUC D日',round(roc_auc_score(o.y,o.s_d),3),'D日+竞价',round(roc_auc_score(o.y,o.s_g),3), {y:(round(roc_auc_score(g.y,g.s_d),3),round(roc_auc_score(g.y,g.s_g),3)) for y,g in o.groupby('yr')})
    base=o.groupby('yr').r_close1.mean()
    def topk(col,k,pool=None):
        x=o if pool is None else o[o.groupby('trade_date')['s_d'].rank(ascending=False,method='first')<=pool]
        t=x.sort_values(col,ascending=False).groupby('trade_date').head(k)
        g=t.groupby('yr').r_close1.agg(['mean',lambda s:(s>0).mean()]); g.columns=['均','胜']
        return f"全期 {t.r_close1.mean():+.2%} 胜{(t.r_close1>0).mean():.0%} | "+' '.join(f"{y}:{r['均']:+.2%}" for y,r in g.iterrows())
    print('全市场平均(T开买T+1收卖):',{y:f'{v:+.2%}' for y,v in base.items()})
    for k in (1,5,10):
        print(f'D日模型 前{k}:',topk('s_d',k))
    for k in (1,5):
        print(f'D日前30池 → 竞价模型 前{k}:',topk('s_g',k,pool=30))
        print(f'全市场竞价模型 前{k}:',topk('s_g',k))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
