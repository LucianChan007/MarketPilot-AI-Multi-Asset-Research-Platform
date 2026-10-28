from __future__ import annotations
"""Leakage-aware exploratory M0/M1 walk-forward benchmark; M2/M3 gated on earnings data.

News classifications generated after their publication dates are explicitly treated as
retrospective text encodings, not point-in-time live features. Results are exploratory
and cannot promote a production model. No network calls are made.
"""
import json, math, sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data' / 'market_data.db'
OUT = ROOT / 'reports' / 'research'
MARKET_FEATURES = ['ret_1','ret_5','ret_20','vol_20','vs_sma20']
NEWS_FEATURES = ['news_net_1d','news_net_5d','news_net_20d','news_weight_1d','news_weight_5d','news_weight_20d','news_count_1d','news_count_5d','news_count_20d','news_earnings_20d','news_macro_20d','news_guidance_20d']


def load_data():
    conn=sqlite3.connect(DB)
    prices=pd.read_sql_query("SELECT trade_date, close, adj_close, volume FROM market_daily WHERE UPPER(ticker)='QQQM' ORDER BY trade_date",conn)
    news=pd.read_sql_query("""SELECT n.news_id,n.published_at,n.source_name,n.title,n.category,
        f.event_type,f.nasdaq_bias,f.impact_score,f.relevance_score,f.confidence,f.classified_at_utc
        FROM news_items n LEFT JOIN news_event_features f ON f.news_id=n.news_id
        AND f.feature_version=(SELECT MAX(f2.feature_version) FROM news_event_features f2 WHERE f2.news_id=n.news_id)
        ORDER BY n.published_at""",conn)
    tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    earnings_candidates=[t for t in tables if any(k in t.lower() for k in ('earn','financial','fundamental','estimate','guidance'))]
    conn.close()
    return prices,news,earnings_candidates


def build_frame(prices, news):
    p=prices.copy(); p['date']=pd.to_datetime(p.trade_date,errors='coerce').dt.date.astype(str)
    p['price']=pd.to_numeric(p.adj_close,errors='coerce').fillna(pd.to_numeric(p.close,errors='coerce'))
    p['ret_1']=p.price.pct_change()*100; p['ret_5']=p.price.pct_change(5)*100; p['ret_20']=p.price.pct_change(20)*100
    daily=p.price.pct_change()*100
    p['vol_20']=daily.rolling(20).std()*math.sqrt(252)
    p['vs_sma20']=(p.price/p.price.rolling(20).mean()-1)*100
    p['target_return']=p.price.shift(-1).div(p.price).sub(1).mul(100)
    p['target_direction']=(p.target_return>0).astype(int)
    for col in NEWS_FEATURES: p[col]=0.0
    n=news.copy(); n['published']=pd.to_datetime(n.published_at,utc=True,errors='coerce')
    n=n[n.published.notna()].copy()
    # Article publication day is converted to New York trading date. Events published after
    # 16:00 ET are first eligible for the next trading session; weekends/holidays roll forward.
    n['published_ny']=n.published.dt.tz_convert('America/New_York')
    market_dates=list(p.date); date_idx={d:i for i,d in enumerate(market_dates)}
    def eligible_index(ts):
        d=ts.date().isoformat(); tm=ts.time()
        if d in date_idx and (tm.hour<16 or (tm.hour==16 and tm.minute==0)):
            return date_idx[d]
        later=[x for x in market_dates if x>d]
        return date_idx[later[0]] if later else None
    n['eligible_date']=n.published_ny.apply(eligible_index)
    for _,r in n.iterrows():
        idx=r.eligible_date
        if pd.isna(idx) or idx is None: continue
        idx=int(idx)
        bias=str(r.nasdaq_bias or 'unknown').lower()
        direction=1.0 if bias=='bullish' else (-1.0 if bias=='bearish' else 0.0)
        impact=float(r.impact_score) if pd.notna(r.impact_score) else 0.0
        relevance=float(r.relevance_score) if pd.notna(r.relevance_score) else 0.0
        confidence=float(r.confidence) if pd.notna(r.confidence) else 0.0
        weighted=direction*max(0,min(2,abs(impact)))*max(0,min(1,relevance))*max(0,min(1,confidence))
        et=str(r.event_type or 'unknown').lower()
        for window,days in [('1d',1),('5d',5),('20d',20)]:
            # Feature is assigned to eligible date and then accumulated only forward from that date.
            p.loc[idx:,f'news_net_{window}'] += weighted
            p.loc[idx:,f'news_weight_{window}'] += abs(impact)*relevance*confidence
            p.loc[idx:,f'news_count_{window}'] += 1
        if et=='earnings': p.loc[idx:,'news_earnings_20d'] += 1
        if et in {'monetary_policy','inflation','employment','economic_growth'}: p.loc[idx:,'news_macro_20d'] += 1
        if et=='company_guidance': p.loc[idx:,'news_guidance_20d'] += 1
    # Rolling-window counts/scores are formed from publication-day event impulses, not cumulative lifetime counts.
    # Recompute using per-session impulses so old news decays out of 1/5/20-session windows.
    impulse=pd.DataFrame(0.0,index=p.index,columns=['net','weight','count','earnings','macro','guidance'])
    for _,r in n.iterrows():
        idx=r.eligible_date
        if pd.isna(idx) or idx is None: continue
        idx=int(idx); bias=str(r.nasdaq_bias or 'unknown').lower()
        direction=1.0 if bias=='bullish' else (-1.0 if bias=='bearish' else 0.0)
        impact=float(r.impact_score) if pd.notna(r.impact_score) else 0.0
        relevance=float(r.relevance_score) if pd.notna(r.relevance_score) else 0.0
        confidence=float(r.confidence) if pd.notna(r.confidence) else 0.0
        impulse.loc[idx,'net'] += direction*max(0,min(2,abs(impact)))*max(0,min(1,relevance))*max(0,min(1,confidence))
        impulse.loc[idx,'weight'] += abs(impact)*relevance*confidence; impulse.loc[idx,'count'] += 1
        et=str(r.event_type or 'unknown').lower()
        impulse.loc[idx,'earnings'] += int(et=='earnings')
        impulse.loc[idx,'macro'] += int(et in {'monetary_policy','inflation','employment','economic_growth'})
        impulse.loc[idx,'guidance'] += int(et=='company_guidance')
    for window,days in [('1d',1),('5d',5),('20d',20)]:
        p[f'news_net_{window}']=impulse.net.rolling(days,min_periods=1).sum().values
        p[f'news_weight_{window}']=impulse.weight.rolling(days,min_periods=1).sum().values
        p[f'news_count_{window}']=impulse['count'].rolling(days,min_periods=1).sum().values
    p['news_earnings_20d']=impulse.earnings.rolling(20,min_periods=1).sum().values
    p['news_macro_20d']=impulse['macro'].rolling(20,min_periods=1).sum().values
    p['news_guidance_20d']=impulse.guidance.rolling(20,min_periods=1).sum().values
    return p,n


def ridge_fit_predict(x_train,y_train,x_test,alpha=10.0):
    mean=x_train.mean(axis=0); std=x_train.std(axis=0); std[std<1e-9]=1.0
    xt=(x_train-mean)/std; xv=(x_test-mean)/std
    ym=float(y_train.mean()); yc=y_train-ym
    beta=np.linalg.solve(xt.T@xt+alpha*np.eye(xt.shape[1]),xt.T@yc)
    return float(ym+(xv@beta).ravel()[0])


def logistic_fit_predict(x_train,y_train,x_test,alpha=0.5,steps=500,lr=.05):
    mean=x_train.mean(axis=0); std=x_train.std(axis=0); std[std<1e-9]=1.0
    x=(x_train-mean)/std; xv=(x_test-mean)/std
    x=np.c_[np.ones(len(x)),x]; xv=np.c_[np.ones(len(xv)),xv]
    beta=np.zeros(x.shape[1]);
    for _ in range(steps):
        z=np.clip(x@beta,-20,20); prob=1/(1+np.exp(-z)); grad=x.T@(prob-y_train)/len(y_train)+alpha*np.r_[0,beta[1:]]/len(y_train)
        beta-=lr*grad
    z=float(np.clip((xv@beta)[0],-20,20)); return 1/(1+math.exp(-z))


def metrics(rows):
    if not rows: return {'n':0}
    actual=np.array([r['actual_return_pct'] for r in rows]); pred=np.array([r['pred_return_pct'] for r in rows]); y=np.array([r['actual_direction'] for r in rows]); prob=np.clip(np.array([r['prob_up'] for r in rows]),1e-6,1-1e-6); guess=(prob>=.5).astype(int)
    recalls=[]
    for c in (0,1):
        mask=y==c
        if mask.any(): recalls.append(float((guess[mask]==c).mean()))
    return {'n':len(rows),'mae_pct':float(np.mean(np.abs(actual-pred))),'rmse_pct':float(np.sqrt(np.mean((actual-pred)**2))),
      'direction_accuracy':float((guess==y).mean()),'balanced_accuracy':float(np.mean(recalls)) if recalls else None,
      'brier_score':float(np.mean((prob-y)**2)),'log_loss':float(-np.mean(y*np.log(prob)+(1-y)*np.log(1-prob)))}


def run_model(frame, features, min_train=80):
    valid=frame.dropna(subset=features+['target_return']).copy()
    results=[]
    for i in range(min_train,len(valid)):
        train=valid.iloc[:i]; test=valid.iloc[[i]]
        xtr=train[features].to_numpy(float); ytr=train.target_return.to_numpy(float); xte=test[features].to_numpy(float)
        # directional classifier target strictly derived from next-session return; trained only on earlier rows
        direction=(train.target_return.to_numpy(float)>0).astype(float)
        try: pred=ridge_fit_predict(xtr,ytr,xte)
        except np.linalg.LinAlgError: pred=float(train.target_return.mean())
        try: prob=logistic_fit_predict(xtr,direction,xte)
        except (ValueError,FloatingPointError): prob=.5
        actual=float(test.target_return.iloc[0])
        results.append({'date':str(test.date.iloc[0]),'actual_return_pct':actual,'pred_return_pct':pred,'actual_direction':int(actual>0),'prob_up':prob})
    return results


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prices,news,earnings_candidates=load_data()
    if prices.empty: raise SystemExit('market_daily 中没有 QQQM 数据')
    frame,valid_news=build_frame(prices,news)
    # Historical classifications were produced after some publication dates. This is an offline
    # retrospective encoding experiment only; never use it to claim live point-in-time performance.
    m0=run_model(frame,MARKET_FEATURES)
    m1=run_model(frame,MARKET_FEATURES+NEWS_FEATURES)
    # Align on exact test dates to avoid an unfair sample comparison.
    m0_by={r['date']:r for r in m0}; m1_by={r['date']:r for r in m1}; common=sorted(set(m0_by)&set(m1_by))
    a0=[m0_by[d] for d in common]; a1=[m1_by[d] for d in common]
    coverage={'price_rows':int(len(frame)),'news_rows':int(len(news)),'classified_news_rows':int(news.event_type.notna().sum()),'unclassified_news_rows':int(news.event_type.isna().sum()),'unique_news_ids':int(news.news_id.nunique()),'published_timestamp_missing':int(news.published_at.isna().sum()),'common_oos_dates':len(common),'oos_first_date':common[0] if common else None,'oos_last_date':common[-1] if common else None,'earnings_candidate_tables':earnings_candidates}
    # Classification timestamps that postdate article publication mean this is retrospective NLP encoding, not live feature replay.
    if 'classified_at_utc' in news:
        pub=pd.to_datetime(news.published_at,utc=True,errors='coerce'); cls=pd.to_datetime(news.classified_at_utc,utc=True,errors='coerce')
        coverage['classified_after_publication_rows']=int(((cls>pub)&cls.notna()&pub.notna()).sum())
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'experiment':'QQQM M0 vs M1 expanding-window walk-forward, same out-of-sample dates','data_coverage':coverage,
      'features':{'M0':MARKET_FEATURES,'M1':MARKET_FEATURES+NEWS_FEATURES,'M2':None,'M3':None},
      'metrics':{'M0':metrics(a0),'M1':metrics(a1)},'paired_delta_M1_minus_M0':{},
      'methodology':{'minimum_initial_training_rows':80,'walk_forward':'expanding training window; one-step-ahead; no shuffled split','same_test_dates':True,'news_alignment':'publication timestamp converted to America/New_York; after-close articles are eligible from next market session','retrospective_classification_warning':True,'production_promotion_allowed':False},
      'gates':{'M0_M1':'exploratory_only','M2':'blocked_no_structured_earnings_consensus_table','M3':'blocked_no_structured_earnings_consensus_table','production_model_changed':False}}
    for metric in ('mae_pct','rmse_pct','direction_accuracy','balanced_accuracy','brier_score','log_loss'):
        x=report['metrics']['M0'].get(metric); y=report['metrics']['M1'].get(metric)
        report['paired_delta_M1_minus_M0'][metric]=None if x is None or y is None else round(y-x,8)
    # Paired bootstrap confidence interval for M1-M0 MAE difference, deterministic seed.
    if common:
        d=np.array([abs(m1_by[k]['actual_return_pct']-m1_by[k]['pred_return_pct'])-abs(m0_by[k]['actual_return_pct']-m0_by[k]['pred_return_pct']) for k in common])
        rng=np.random.default_rng(20261022); means=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(2000)])
        report['paired_mae_bootstrap_95pct_pct']={'mean_delta_m1_minus_m0':float(d.mean()),'low':float(np.quantile(means,.025)),'high':float(np.quantile(means,.975)),'replicates':2000}
    outcsv=OUT/'M0_M1_walkforward_paired_predictions.csv'
    pd.DataFrame([{'date':d,'m0_actual_return_pct':m0_by[d]['actual_return_pct'],'m0_pred_return_pct':m0_by[d]['pred_return_pct'],'m0_prob_up':m0_by[d]['prob_up'],'m1_actual_return_pct':m1_by[d]['actual_return_pct'],'m1_pred_return_pct':m1_by[d]['pred_return_pct'],'m1_prob_up':m1_by[d]['prob_up']} for d in common]).to_csv(outcsv,index=False,encoding='utf-8-sig')
    jpath=OUT/'M0_M1_walkforward_experiment.json'; jpath.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# QQQM 新闻特征滚动样本外实验报告','',f"生成时间 UTC：{report['generated_at_utc']}",'','## 数据覆盖','',f"- QQQM 日线：{coverage['price_rows']} 条；{frame.date.min()} 至 {frame.date.max()}",f"- 新闻：{coverage['news_rows']} 条，已分类 {coverage['classified_news_rows']} 条，未分类 {coverage['unclassified_news_rows']} 条",f"- 同一批样本外日期：{coverage['common_oos_dates']} 个，{coverage['oos_first_date']} 至 {coverage['oos_last_date']}",f"- 分类晚于发布时间的新闻：{coverage.get('classified_after_publication_rows','未知')} 条",'', '## 同日期样本外结果','', '| 指标 | M0 价格特征 | M1 加新闻 | M1-M0（负值通常更好，方向指标除外） |','|---|---:|---:|---:|']
    for metric,label in [('mae_pct','MAE（收益率百分点）'),('rmse_pct','RMSE（收益率百分点）'),('direction_accuracy','方向准确率'),('balanced_accuracy','平衡准确率'),('brier_score','Brier Score'),('log_loss','Log Loss')]:
        x=report['metrics']['M0'].get(metric); y=report['metrics']['M1'].get(metric); d=report['paired_delta_M1_minus_M0'].get(metric)
        fmt=lambda z: 'N/A' if z is None else f'{z:.5f}'
        lines.append(f'| {label} | {fmt(x)} | {fmt(y)} | {fmt(d)} |')
    ci=report.get('paired_mae_bootstrap_95pct_pct')
    if ci: lines += ['',f"配对 MAE 差值自助法 95% 区间：[{ci['low']:.5f}, {ci['high']:.5f}]；M1-M0 均值 {ci['mean_delta_m1_minus_m0']:.5f}。"]
    lines += ['', '## 解释边界','', '- 这是探索性回测：新闻历史只有几十条，分类覆盖有限，统计功效不足。不能据此声称新闻因子稳定有效。','- 新闻事件分类由当前模型事后生成，尽管特征只按文章发布时间对齐，但并非当时实时生成的分类；结果不能等同于实时可复现策略。','- 未发现结构化财报/一致预期表，M2/M3 本轮明确阻断，未用零值伪造财报特征。','- 生产模型未改变；只有在扩充历史新闻、补齐可用时点可靠的财报数据，并在独立样本外测试中稳定改善后，才考虑晋级。','', '## 文件','',f'- 配对预测明细：`{outcsv.name}`',f'- 机器可读结果：`{jpath.name}`']
    md=OUT/'QQQM_新闻特征滚动样本外实验报告.md'; md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(md); print(jpath); print(json.dumps(report['metrics'],ensure_ascii=False,indent=2)); print('GATES',report['gates'])

if __name__=='__main__': main()
