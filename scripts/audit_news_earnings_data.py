from __future__ import annotations
"""Audit point-in-time news and earnings data availability for QQQM research."""
import json, sqlite3, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data' / 'market_data.db'
OUT = ROOT / 'reports' / 'research'


def table_exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

def cols(conn, name):
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')] if table_exists(conn,name) else []

def count(conn, table):
    return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] if table_exists(conn,table) else 0

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    conn=sqlite3.connect(DB)
    conn.row_factory=sqlite3.Row
    report={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"database":str(DB),"tables":{},"findings":[],"gate":"research_only_not_promoted"}
    for t in ['news_items','news_event_features','market_daily','macro_observations','bls_macro_observations','forecast_feedback','forecast_context_snapshots']:
        report['tables'][t]={"exists":table_exists(conn,t),"row_count":count(conn,t),"columns":cols(conn,t)}
    if table_exists(conn,'news_items'):
        row=conn.execute("SELECT MIN(published_at),MAX(published_at),COUNT(*),SUM(CASE WHEN published_at IS NULL OR TRIM(published_at)='' THEN 1 ELSE 0 END) FROM news_items").fetchone()
        report['news']={"count":row[2],"published_min":row[0],"published_max":row[1],"missing_published_at":row[3]}
        report['news']['sources']=[dict(r) for r in conn.execute("SELECT source_id,source_name,COUNT(*) AS n,MIN(published_at) AS first_published,MAX(published_at) AS last_published FROM news_items GROUP BY source_id,source_name ORDER BY n DESC")]
        report['news']['categories']=[dict(r) for r in conn.execute("SELECT category,COUNT(*) AS n FROM news_items GROUP BY category ORDER BY n DESC")]
        report['news']['duplicate_urls']=conn.execute("SELECT COUNT(*) FROM (SELECT url FROM news_items GROUP BY url HAVING COUNT(*)>1)").fetchone()[0]
    if table_exists(conn,'news_event_features'):
        row=conn.execute("SELECT COUNT(*),COUNT(DISTINCT news_id),MIN(classified_at_utc),MAX(classified_at_utc),MIN(published_at),MAX(published_at) FROM news_event_features").fetchone()
        report['news_classification']={"rows":row[0],"unique_news":row[1],"classified_min":row[2],"classified_max":row[3],"published_min":row[4],"published_max":row[5],"versions":[dict(r) for r in conn.execute("SELECT feature_version,classifier_model,COUNT(*) AS n FROM news_event_features GROUP BY feature_version,classifier_model")],"event_types":[dict(r) for r in conn.execute("SELECT event_type,COUNT(*) AS n FROM news_event_features GROUP BY event_type ORDER BY n DESC")],"biases":[dict(r) for r in conn.execute("SELECT nasdaq_bias,COUNT(*) AS n FROM news_event_features GROUP BY nasdaq_bias ORDER BY n DESC")]}
    if table_exists(conn,'market_daily'):
        report['daily_market']=[dict(r) for r in conn.execute("SELECT ticker,COUNT(*) AS n,MIN(trade_date) AS first_date,MAX(trade_date) AS last_date,COUNT(close) AS close_count FROM market_daily GROUP BY ticker ORDER BY ticker")]
    if table_exists(conn,'bls_macro_observations'):
        report['macro_release_fields']={"columns":cols(conn,'bls_macro_observations'),"note":"observation period/value is not equivalent to an exact public release timestamp or vintage"}
    earnings_tables=[]
    for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        n=r[0].lower()
        if any(k in n for k in ['earn','financial','fundamental','estimate','guidance']): earnings_tables.append(r[0])
    report['earnings']={"candidate_tables":earnings_tables,"available_structured_earnings_table":bool(earnings_tables),"required_fields":["ticker","fiscal_period","actual_release_timestamp","available_at","eps_actual","eps_estimate","revenue_actual","revenue_estimate","guidance_change","source_url","retrieved_at"]}
    conn.close()
    news_n=report.get('news',{}).get('count',0); classified=report.get('news_classification',{}).get('unique_news',0)
    if news_n < 200: report['findings'].append(f"新闻历史覆盖偏少：当前 {news_n} 条，难以支撑有统计功效的长期新闻因子回测。")
    if classified < news_n: report['findings'].append(f"新闻分类覆盖 {classified}/{news_n}；未分类新闻不能自动视为中性新闻，应区分缺失与中性。")
    if not earnings_tables: report['findings'].append("数据库未发现明确的财报/一致预期结构化表，M2/M3 暂不能声称已完成财报特征回测。")
    report['findings'].append("新闻分类记录的 classified_at_utc 晚于部分文章发布时间；做历史回测时须说明这是事后文本编码，并保证输入仅使用当时已发布的文章内容，不能使用未来新闻。")
    report['findings'].append("宏观观察值缺少完整发布时点与历史数据版本时，不可直接按 observation_date 当作当时可用信息。")
    json_path=OUT/'news_earnings_data_audit.json'; md_path=OUT/'新闻与财报历史数据审计.md'
    json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# QQQM 新闻与财报历史数据审计','',f"审计时间（UTC）：{report['generated_at_utc']}",'', '## 核心盘点','', '| 数据表 | 是否存在 | 行数 |','|---|---:|---:|']
    for t,v in report['tables'].items(): lines.append(f"| `{t}` | {'是' if v['exists'] else '否'} | {v['row_count']} |")
    if 'news' in report:
        n=report['news']; lines += ['', '## 新闻覆盖', '',f"- 新闻条数：{n['count']}",f"- 发布时间范围：{n['published_min']} 至 {n['published_max']}",f"- 缺失发布时间：{n['missing_published_at']}",f"- URL 重复组数：{n['duplicate_urls']}",'','### 来源分布','', '| 来源 | 条数 | 最早 | 最新 |','|---|---:|---|---|']
        for s in n['sources']: lines.append(f"| {s['source_name']} (`{s['source_id']}`) | {s['n']} | {s['first_published']} | {s['last_published']} |")
    if 'news_classification' in report:
        n=report['news_classification']; lines += ['', '## 新闻事件特征缓存','',f"- 已分类文章：{n['unique_news']} 条",f"- 分类时间范围：{n['classified_min']} 至 {n['classified_max']}",'','### 事件类型','', '| 类型 | 条数 |','|---|---:|']
        for x in n['event_types']: lines.append(f"| {x['event_type']} | {x['n']} |")
    lines += ['', '## 财报数据可用性','',f"- 发现的候选财报表：{', '.join(report['earnings']['candidate_tables']) if earnings_tables else '无'}",'- 必须补齐实际发布时间、可用时点、EPS/营收实际值及一致预期、来源链接和抓取时间。','', '## 风险与结论','']
    lines += [f'- {x}' for x in report['findings']]
    lines += ['', '## 研究结论','', '当前项目具备新闻条目与部分 DeepSeek 分类缓存，但新闻覆盖很小，且未发现结构化财报/预期数据库。可以先完成 M0/M1 的工程验证与覆盖率报告；M2/M3 必须等有可靠财报数据后才能进行。不得将缺失特征填成中性值来夸大覆盖率。']
    md_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(md_path); print(json_path); print('\n'.join(report['findings']))
if __name__=='__main__': main()
