from __future__ import annotations

"""Export the CURRENT canonical intraday forecast to a WeChat-ready HTML draft.

This script does not create or change predictions. It reads the existing forecast
JSON, plots that forecast, and writes Markdown + self-contained HTML + PNG.
"""
import argparse
import base64
import html
import json
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"预测文件不是 JSON 对象：{path}")
    return data


def latest_market_date(db_path: Path, ticker: str) -> str | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        try:
            row = conn.execute(
                "SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)=UPPER(?)", (ticker,)
            ).fetchone()
        except sqlite3.Error:
            return None
    return str(row[0])[:10] if row and row[0] else None


def plot_forecast(forecast: dict[str, Any], image_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from matplotlib import font_manager
        installed = {font.name for font in font_manager.fontManager.ttflist}
        for name in ("Microsoft YaHei", "Microsoft JhengHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
            if name in installed:
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                break
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

    curve = forecast.get("curve") or []
    if not curve:
        raise ValueError("预测 JSON 没有 curve 数据，不能生成公众号预测图。")
    times = [str(p.get("time_ny", "")) for p in curve]
    p10 = [float(p.get("low_price")) for p in curve]
    p50 = [float(p.get("median_price")) for p in curve]
    p90 = [float(p.get("high_price")) for p in curve]
    main = forecast.get("main_path") or {}
    raw_path = main.get("path") or []
    main_prices = {str(p.get("time_ny")): float(p["predicted_price"]) for p in raw_path
                   if p.get("time_ny") and p.get("predicted_price") is not None}
    main_y = [main_prices.get(t) for t in times]

    fig, ax = plt.subplots(figsize=(13.2, 6.6), dpi=170)
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("white")
    x = list(range(len(times)))
    ax.fill_between(x, p10, p90, alpha=0.13, label="历史参考范围（P10–P90）")
    ax.plot(x, p10, linestyle="--", linewidth=1.2, label="偏弱参考")
    ax.plot(x, p50, linewidth=2.0, label="历史中位参考路径")
    ax.plot(x, p90, linestyle="--", linewidth=1.2, label="偏强参考")
    if any(v is not None for v in main_y):
        ax.plot(x, main_y, linewidth=2.7, linestyle="-.", marker="o", markersize=2.7,
                label=f"主预测路径：{main.get('model_name', '实验模型')}")
    previous_close = forecast.get("previous_session_close")
    if previous_close is not None:
        ax.axhline(float(previous_close), color="#6B7280", linewidth=1, linestyle=":",
                   label=f"前收 ${float(previous_close):.2f}")
    estimated_open = forecast.get("estimated_open") or {}
    if estimated_open.get("p50_price") is not None:
        ax.scatter([0], [float(estimated_open["p50_price"])], marker="D", s=52, zorder=5,
                   label=f"预计开盘参考 ${float(estimated_open['p50_price']):.2f}")
    close_obj = forecast.get("estimated_close_from_path") or {}
    if close_obj.get("p50_price") is not None:
        ax.scatter([len(x)-1], [float(close_obj["p50_price"])], marker="s", s=48, zorder=5,
                   label=f"预计收盘参考 ${float(close_obj['p50_price']):.2f}")
    tick_indices = [i for i, t in enumerate(times) if t in {"09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "15:45"}]
    if not tick_indices:
        tick_indices = list(range(0, len(times), max(1, len(times)//8)))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([times[i] for i in tick_indices], rotation=0)
    ax.set_title(f"QQQM 下一交易日走势参考｜目标日期 {forecast.get('forecast_date_weekday_estimate', '待确认')}",
                 loc="left", fontsize=15, fontweight="bold", color="#26364B", pad=15)
    ax.set_xlabel("美东时间（ET）")
    ax.set_ylabel("价格（美元）")
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", ncols=2, fontsize=8.2, framealpha=0.95)
    news = forecast.get("news_context") or {}
    plt.figtext(0.01, 0.01,
                f"行情基准：{forecast.get('as_of_date','未知')} 收盘；新闻快照 {news.get('included_article_count', 0)} 条。"
                "预测路径是历史数据训练得到的实验估算；P10/P50/P90 为历史参考情景，不是概率保证。",
                fontsize=8, color="#536173")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_article(forecast: dict[str, Any], chart_path: Path) -> tuple[str, str]:
    ticker = str(forecast.get("ticker") or "QQQM")
    as_of = str(forecast.get("as_of_date") or "未知")
    target = str(forecast.get("forecast_date_weekday_estimate") or "待确认")
    previous_close = forecast.get("previous_session_close")
    op = (forecast.get("estimated_open") or {}).get("p50_price")
    close = (forecast.get("estimated_close_from_path") or {}).get("p50_price")
    main = forecast.get("main_path") or {}
    method = str(main.get("model_name") or forecast.get("curve_method") or "历史情景参考")
    training_rows = main.get("training_rows", main.get("candidate_count", "数据集决定"))
    news_ctx = forecast.get("news_context") or {}
    news_items = news_ctx.get("items") or []
    valid_items = [x for x in news_items if isinstance(x, dict) and x.get("title")]

    open_text = f"${float(op):.2f}" if op is not None else "暂缺"
    close_text = f"${float(close):.2f}" if close is not None else "暂缺"
    prev_text = f"${float(previous_close):.2f}" if previous_close is not None else "暂缺"
    news_lines: list[str] = []
    for item in valid_items[:5]:
        title = str(item.get("title") or "").replace("\n", " ").strip()
        source = str(item.get("source_name") or "新闻来源待核对").strip()
        published = str(item.get("published_at") or "发布时间待核对")
        news_lines.append(f"- {title}（{source}，发布时间：{published}）")
    if not news_lines:
        news_lines = ["- 当前预测快照没有可展示的合格新闻条目。发布前请人工核实最新新闻，不要把缺少新闻当作没有风险。"]

    caveats = forecast.get("caveats") or []
    caveat_lines = [f"- {str(item)}" for item in caveats[:6]] if caveats else ["- 历史回测不能保证未来表现，预测路径可能与实际行情明显不同。"]
    md = f"""# QQQM 明日观察：先看路线图，别把预测当水晶球

> 行情基准：{as_of} 美股收盘｜目标交易日：{target}｜模型：{method}

## 先说人话：明天看什么？

QQQM 是一只跟纳斯达克 100 指数相关的大型科技股 ETF。模型基于截至 **{as_of}** 已知的日线、盘中历史数据和可用的新闻快照，给出一个下一交易日情景参考。它不是内幕消息，更不是保证中奖的刮刮乐。

- 前一交易日收盘：**{prev_text}**
- 预计开盘参考：**{open_text}**
- 预计收盘参考：**{close_text}**
- 主路径模型：**{method}**
- 路径模型训练/候选记录数：**{training_rows}**

![QQQM 下一交易日预测曲线](data:image/png;base64,{base64.b64encode(chart_path.read_bytes()).decode('ascii')})

## 这条曲线该怎么看？

粗线是主预测路径；其他线是过去相似或历史分位情景的参考。它们用于表达“过去数据支持哪些可能的走法”，并非精确到每一分钟的承诺，也不是经过充分校准的上涨概率。开盘跳空、突发新闻和大型科技公司的业绩都可能让真实走势偏离模型。

## 新闻线索：市场在聊什么？

{chr(10).join(news_lines)}

当前快照纳入新闻 **{news_ctx.get('included_article_count', 0)} 条**。新闻需要看发布时间和相关性，旧消息不能当成今天的新消息；新闻情绪也不等于价格一定朝同方向变化。

## 明天收盘后，我们会拿真实数据验收

下一交易日结束后，系统会将实际开盘、最高、最低、收盘和盘中走势与今天保存的预测快照对照，记录误差，再与简单基准和其他候选模型比较。**只有在多次样本外验证中持续更好，候选模型才值得升级。** 一天猜对了，不封神；一天猜错了，也不急着乱改参数。

## 风险提示

{chr(10).join(caveat_lines)}

本文是个人编程与市场研究实验，不构成投资建议，也不是买入或卖出指令。市场价格可能大幅波动，请勿仅凭此文进行交易。

*QQQM Investment Assistant｜让模型接受真实行情的考试，而不是自己给自己颁奖。*
"""

    img64 = base64.b64encode(chart_path.read_bytes()).decode("ascii")
    news_html_items = "".join(
        "<li>{}（{}，发布时间：{}）</li>".format(
            html.escape(str(item.get("title") or "")),
            html.escape(str(item.get("source_name") or "来源待核实")),
            html.escape(str(item.get("published_at") or "发布时间待核实")),
        )
        for item in valid_items[:5]
    ) or "<li>当前预测快照没有可展示的合格新闻条目。发布前请人工核实最新新闻。</li>"
    caveat_html_items = "".join(
        "<li>{}</li>".format(html.escape(str(item))) for item in caveats[:6]
    ) or "<li>历史回测不能保证未来表现，预测路径可能与实际行情明显不同。</li>"
    article_html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QQQM 明日观察：先看路线图，别把预测当水晶球</title>
<style>
body{{margin:0;background:#f4f6fa;color:#253247;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif;line-height:1.8}}
.article{{max-width:760px;margin:0 auto;background:#fff;padding:24px 22px 36px}}
h1{{font-size:26px;line-height:1.35;color:#183153;margin:8px 0 12px}} h2{{font-size:20px;margin-top:30px;color:#183153;border-left:4px solid #3b82f6;padding-left:10px}}
p,li{{font-size:16px}} .meta{{font-size:13px;color:#66758b;background:#f4f7fb;padding:10px 12px;border-radius:8px}}
.kpis{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:14px 0}}
.kpi{{background:#f3f7fc;border-radius:10px;padding:12px 8px;text-align:center}} .kpi span{{display:block;font-size:12px;color:#65758b}} .kpi strong{{font-size:20px;color:#173d6c}}
img{{max-width:100%;height:auto;border-radius:10px}} .tip{{background:#fff8e7;border-left:4px solid #e8a23a;padding:12px 14px}}
footer{{font-size:13px;color:#718096;border-top:1px solid #e4eaf2;margin-top:30px;padding-top:14px}}
@media(max-width:560px){{.article{{padding:18px 14px}}h1{{font-size:22px}}.kpis{{grid-template-columns:1fr}}.kpi{{display:flex;align-items:center;justify-content:space-between}}}}
</style></head><body><main class="article">
<h1>QQQM 明日观察：先看路线图，别把预测当水晶球</h1>
<div class="meta">行情基准：{html.escape(as_of)} 收盘｜目标交易日：{html.escape(target)}｜模型：{html.escape(method)}</div>
<p>QQQM 是一只与纳斯达克 100 指数相关的大型科技股 ETF。模型基于截至 <b>{html.escape(as_of)}</b> 已知的历史行情、盘中数据和可用新闻，整理出下一交易日的情景参考。它不是内幕消息，更不是保证中奖的刮刮乐。</p>
<div class="kpis"><div class="kpi"><span>前收</span><strong>{html.escape(prev_text)}</strong></div><div class="kpi"><span>预计开盘参考</span><strong>{html.escape(open_text)}</strong></div><div class="kpi"><span>预计收盘参考</span><strong>{html.escape(close_text)}</strong></div></div>
<img alt="QQQM 下一交易日预测曲线" src="data:image/png;base64,{img64}">
<h2>这条曲线怎么看？</h2><p>粗线是主预测路径；其他线是历史参考情景。它们不是每一分钟的承诺，也不是经过充分校准的上涨概率。开盘跳空、突发新闻和大型科技公司的业绩，都可能让真实走势偏离模型。</p>
<h2>新闻线索：市场在聊什么？</h2><ul>{news_html_items}</ul>
<p>当前快照纳入新闻 <b>{int(news_ctx.get('included_article_count') or 0)}</b> 条。看新闻要看发布时间和相关性，旧消息不能当成今天的新消息；新闻情绪也不代表价格一定朝同方向变化。</p>
<h2>明天收盘后，我们会拿真实数据验收</h2><p>系统会将实际开盘、最高、最低、收盘和盘中走势与今天保存的预测快照对照，记录误差，再和简单基准及其他候选模型比较。多次样本外验证持续更好的模型，才值得升级。一天猜对了，不封神；一天猜错了，也不急着乱改参数。</p>
<h2>风险提示</h2><ul>{caveat_html_items}</ul>
<p class="tip">本文是个人编程与市场研究实验，不构成投资建议，也不是买入或卖出指令。市场价格可能大幅波动，请勿仅凭此文进行交易。</p>
<footer>QQQM Investment Assistant｜让模型接受真实行情的考试，而不是自己给自己颁奖。</footer>
</main></body></html>"""
    return md, article_html


def main() -> int:
    parser = argparse.ArgumentParser(description="把已有的盘中预测快照导出为公众号 HTML，不生成新预测。")
    parser.add_argument("--forecast", default="reports/intraday/QQQM_intraday_forecast_latest.json")
    parser.add_argument("--db", default="data/market_data.db")
    parser.add_argument("--output-dir", default="reports/articles")
    parser.add_argument("--allow-past", action="store_true", help="允许将过期预测重新导出为历史文章；默认拒绝过期预测")
    args = parser.parse_args()
    forecast_path = (ROOT / args.forecast).resolve() if not Path(args.forecast).is_absolute() else Path(args.forecast)
    db_path = (ROOT / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)
    out_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    try:
        forecast = read_json(forecast_path)
        target = str(forecast.get("forecast_date_weekday_estimate") or "")
        as_of = str(forecast.get("as_of_date") or "")
        if not target or not as_of:
            raise ValueError("预测文件缺少 forecast_date_weekday_estimate 或 as_of_date")
        date.fromisoformat(target)
        latest = latest_market_date(db_path, str(forecast.get("ticker") or "QQQM"))
        if latest and target <= latest and not args.allow_past:
            raise ValueError(f"预测目标日 {target} 已不晚于本地真实行情最新日期 {latest}，拒绝把过期预测当作明日文章。若只是重导历史文章，可加 --allow-past。")
        stamp = target.replace("-", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / f"QQQM_forecast_{stamp}_curve.png"
        md_path = out_dir / f"QQQM_forecast_{stamp}_wechat.md"
        html_path = out_dir / f"QQQM_forecast_{stamp}_wechat.html"
        plot_forecast(forecast, image_path)
        markdown, article_html = build_article(forecast, image_path)
        md_path.write_text(markdown, encoding="utf-8")
        html_path.write_text(article_html, encoding="utf-8")
        print("公众号草稿已导出：")
        print(f"  HTML：{html_path}")
        print(f"  Markdown：{md_path}")
        print(f"  曲线图：{image_path}")
        print("注意：文件只生成草稿，不会自动发布到微信公众号。")
        return 0
    except Exception as exc:
        print(f"导出失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
