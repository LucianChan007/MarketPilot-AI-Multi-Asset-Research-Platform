import importlib.util
import unittest
from pathlib import Path
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_news_earnings_walkforward_experiment.py'
spec = importlib.util.spec_from_file_location('news_earnings_walkforward_experiment', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class NewsEarningsWalkforwardTests(unittest.TestCase):
    def test_after_close_news_is_not_available_same_session(self):
        prices = pd.DataFrame({
            'trade_date':['2026-10-19','2026-10-20','2026-10-21'],
            'close':[100,101,102], 'adj_close':[100,101,102], 'volume':[1,1,1]
        })
        news = pd.DataFrame([{
            'news_id':'n1','published_at':'2026-10-19T21:30:00+00:00', # 17:30 ET
            'source_name':'test','title':'test','category':'general','event_type':'other',
            'nasdaq_bias':'bullish','impact_score':1.0,'relevance_score':1.0,'confidence':1.0,
            'classified_at_utc':'2026-10-20T00:00:00+00:00'
        }])
        frame, _ = mod.build_frame(prices, news)
        self.assertEqual(frame.loc[0,'news_count_1d'], 0)
        self.assertEqual(frame.loc[1,'news_count_1d'], 1)

    def test_metrics_returns_required_fields(self):
        rows=[
            {'actual_return_pct':1.0,'pred_return_pct':.5,'actual_direction':1,'prob_up':.7},
            {'actual_return_pct':-1.0,'pred_return_pct':-.2,'actual_direction':0,'prob_up':.3},
        ]
        result=mod.metrics(rows)
        for key in ('mae_pct','rmse_pct','direction_accuracy','balanced_accuracy','brier_score','log_loss'):
            self.assertIn(key,result)
        self.assertEqual(result['n'],2)
        self.assertAlmostEqual(result['direction_accuracy'],1.0)

if __name__ == '__main__': unittest.main()
