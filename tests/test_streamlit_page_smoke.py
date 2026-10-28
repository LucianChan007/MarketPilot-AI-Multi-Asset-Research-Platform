"""Exercise the real logged-out and default-home Streamlit entrypoint.

Run after installing requirements.txt:
  .venv\\Scripts\\python.exe -m unittest tests.test_streamlit_page_smoke -v
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # Core-only CI environments may omit GUI dependencies.
    AppTest = None  # type: ignore[assignment,misc]


@unittest.skipIf(AppTest is None, "Streamlit is not installed; run after pip install -r requirements.txt")
class StreamlitPageSmokeTests(unittest.TestCase):
    def test_real_dashboard_renders_login_then_authenticated_four_asset_home(self):
        # The first run checks the actual login page. We then set the session principal
        # and stub only the database user lookup so the test reaches the real st.navigation
        # entrypoint without writing a test account into the user's local users.db.
        entrypoint = ROOT / "dashboard.py"
        app = AppTest.from_file(str(entrypoint), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0, "Dashboard login page raised a Streamlit exception")
        logged_out_text = "\n".join(
            str(getattr(element, "value", ""))
            for group in (app.title, app.caption, app.info, app.warning)
            for element in group
        )
        self.assertIn("MarketPilot AI", logged_out_text)

        principal = {"user_id": "marketpilot-ui-smoke-user", "username": "ui_smoke"}
        app.session_state["qqqm_authenticated_user"] = principal
        with patch("app.services.auth.AuthService.get_user_by_id", return_value=principal):
            app.run()
        self.assertEqual(len(app.exception), 0, "Authenticated default dashboard raised a Streamlit exception")
        rendered = "\n".join(
            str(getattr(element, "value", ""))
            for group in (app.title, app.header, app.subheader, app.markdown, app.caption, app.info, app.warning, app.error)
            for element in group
        )
        self.assertIn("四资产行情总览", rendered)

        # Streamlit 的按钮不属于 Markdown 文本，
        # 因此必须从 app.button 单独检查。
        button_labels = [
            str(getattr(element, "label", ""))
            for element in app.button
        ]

        self.assertTrue(
            any("仅更新行情" in label for label in button_labels),
            f"未找到“仅更新行情”按钮。当前按钮：{button_labels}",
        )

        self.assertIn("QQQM", rendered)
        self.assertIn("AAPL", rendered)
        self.assertIn("MSFT", rendered)
        self.assertIn("NVDA", rendered)


if __name__ == "__main__":
    unittest.main()
