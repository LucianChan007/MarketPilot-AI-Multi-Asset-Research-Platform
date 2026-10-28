import importlib.util
from pathlib import Path
import unittest
import pandas as pd
SCRIPT=Path(__file__).resolve().parents[1]/'app'/'earnings_features.py'
spec=importlib.util.spec_from_file_location('earnings_features',SCRIPT); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class EarningsFeatureTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame([{
            'ticker':'AAPL','fiscal_period':'2026Q3','actual_release_timestamp':'2026-10-20T20:00:00Z',
            'available_at':'2026-10-20T20:01:00Z','eps_actual':1.2,'eps_estimate':1.0,
            'revenue_actual':100,'revenue_estimate':95,'guidance_change_pct':2.5,
            'source_name':'test','source_url':'https://example.com/release','retrieved_at':'2026-10-21T00:00:00Z'}])
    def test_no_future_earnings_leakage(self):
        result=mod.features_asof(self.sample(),'2026-10-20T19:59:00Z')
        self.assertTrue(pd.isna(result['earnings_eps_surprise_pct']))
        result=mod.features_asof(self.sample(),'2026-10-20T20:02:00Z')
        self.assertAlmostEqual(result['earnings_eps_surprise_pct'],20.0)
    def test_reject_availability_before_release(self):
        x=self.sample(); x.loc[0,'available_at']='2026-10-20T19:00:00Z'
        clean,issues=mod.validate_earnings_frame(x)
        self.assertEqual(len(clean),0); self.assertTrue(any('available_before_release' in x for x in issues))
    def test_missing_consensus_is_not_zero(self):
        x=self.sample(); x.loc[0,'eps_estimate']=None
        result=mod.features_asof(x,'2026-10-21T00:00:00Z')
        self.assertTrue(pd.isna(result['earnings_eps_surprise_pct']))

    def test_holdings_aggregate_uses_only_known_snapshot(self):
        earnings = pd.concat([self.sample(), self.sample().assign(ticker='MSFT', eps_actual=2.2, eps_estimate=2.0)], ignore_index=True)
        holdings = pd.DataFrame([
            {'ticker':'AAPL','weight':0.6,'available_at':'2026-10-20T19:00:00Z','source_url':'https://example.com/holdings'},
            {'ticker':'MSFT','weight':0.4,'available_at':'2026-10-20T19:00:00Z','source_url':'https://example.com/holdings'},
            {'ticker':'NVDA','weight':0.9,'available_at':'2026-10-21T00:00:00Z','source_url':'https://example.com/future-holdings'},
        ])
        result=mod.aggregate_constituent_features_asof(earnings, holdings, '2026-10-20T21:00:00Z', min_weight_coverage=.7)
        self.assertAlmostEqual(result['earnings_eps_surprise_pct'], 16.0)
        self.assertAlmostEqual(result['earnings_weight_coverage'], 1.0)
        self.assertEqual(result['earnings_constituent_count'], 2.0)

    def test_holdings_rows_with_different_timestamps_keep_same_day_snapshot(self):
        earnings = pd.concat([self.sample(), self.sample().assign(ticker='MSFT', eps_actual=2.2, eps_estimate=2.0)], ignore_index=True)
        holdings = pd.DataFrame([
            {'ticker':'AAPL','weight':0.6,'available_at':'2026-10-20T19:00:01Z','source_url':'https://example.com/holdings'},
            {'ticker':'MSFT','weight':0.4,'available_at':'2026-10-20T19:00:07Z','source_url':'https://example.com/holdings'},
        ])
        result=mod.aggregate_constituent_features_asof(earnings, holdings, '2026-10-20T21:00:00Z', min_weight_coverage=.7)
        self.assertAlmostEqual(result['earnings_eps_surprise_pct'], 16.0)
        self.assertEqual(result['earnings_constituent_count'], 2.0)

    def test_holdings_snapshot_id_prevents_mixing_snapshots(self):
        earnings = pd.concat([self.sample(), self.sample().assign(ticker='MSFT', eps_actual=2.2, eps_estimate=2.0)], ignore_index=True)
        holdings = pd.DataFrame([
            {'ticker':'AAPL','weight':0.6,'available_at':'2026-10-20T19:00:01Z','snapshot_id':'old','source_url':'https://example.com/holdings'},
            {'ticker':'MSFT','weight':0.4,'available_at':'2026-10-20T19:00:07Z','snapshot_id':'old','source_url':'https://example.com/holdings'},
            {'ticker':'AAPL','weight':0.7,'available_at':'2026-10-20T20:00:01Z','snapshot_id':'new','source_url':'https://example.com/holdings'},
        ])
        result=mod.aggregate_constituent_features_asof(earnings, holdings, '2026-10-20T21:00:00Z', min_weight_coverage=.7)
        self.assertAlmostEqual(result['earnings_eps_surprise_pct'], 20.0)
        self.assertEqual(result['earnings_constituent_count'], 1.0)

    def test_same_day_distinct_exports_are_not_mixed_without_snapshot_id(self):
        earnings = pd.concat([
            self.sample(),
            self.sample().assign(ticker='MSFT', eps_actual=2.2, eps_estimate=2.0),
            self.sample().assign(ticker='NVDA', eps_actual=3.0, eps_estimate=2.0),
        ], ignore_index=True)
        holdings = pd.DataFrame([
            {'ticker':'AAPL','weight':0.6,'available_at':'2026-10-20T09:00:00Z','source_url':'https://example.com/snapshot'},
            {'ticker':'MSFT','weight':0.4,'available_at':'2026-10-20T09:00:07Z','source_url':'https://example.com/snapshot'},
            # Separate export later the same UTC day, should be its own snapshot.
            {'ticker':'NVDA','weight':1.0,'available_at':'2026-10-20T15:00:00Z','source_url':'https://example.com/snapshot'},
        ])
        result=mod.aggregate_constituent_features_asof(earnings, holdings, '2026-10-20T21:00:00Z', min_weight_coverage=.1)
        self.assertEqual(result['earnings_constituent_count'], 1.0)
        self.assertAlmostEqual(result['earnings_eps_surprise_pct'], 50.0)

    def test_holdings_rejects_negative_weight_and_missing_source(self):
        holdings = pd.DataFrame([
            {'ticker':'AAPL','weight':-0.2,'available_at':'2026-10-20T19:00:00Z','source_url':'https://example.com/holdings'},
            {'ticker':'MSFT','weight':0.8,'available_at':'2026-10-20T19:00:00Z','source_url':''},
        ])
        result=mod.aggregate_constituent_features_asof(self.sample(), holdings, '2026-10-21T00:00:00Z')
        self.assertEqual(result['earnings_weight_coverage'], 0.0)
        self.assertTrue(pd.isna(result['earnings_eps_surprise_pct']))

    def test_holdings_aggregate_blocks_low_coverage(self):
        earnings=self.sample()
        holdings=pd.DataFrame([{'ticker':'AAPL','weight':0.2,'available_at':'2026-10-20T19:00:00Z','source_url':'https://example.com/holdings'}, {'ticker':'MSFT','weight':0.8,'available_at':'2026-10-20T19:00:00Z','source_url':'https://example.com/holdings'}])
        result=mod.aggregate_constituent_features_asof(earnings, holdings, '2026-10-21T00:00:00Z', min_weight_coverage=.7)
        self.assertTrue(pd.isna(result['earnings_eps_surprise_pct']))
        self.assertLess(result['earnings_weight_coverage'], .7)

if __name__=='__main__': unittest.main()
