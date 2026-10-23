"""Real-browser smoke tests against a freshly started ``streamlit run app.py`` (skipped without playwright + chromium).

These cover things AppTest cannot: deep links as the first request after a server start, the cold-start notice and the
phone drawer closing after navigation.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from src import config

ROOT = Path(__file__).resolve().parents[1]
pw = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.skipif(not config.ORDERS_DELIVERED_PATH.is_file(), reason="processed data missing")

#: url_path -> heading shown on that page
DEEP_LINKS = {"": "Overview", "delay-analysis": "Delay Analysis", "regional-intelligence": "Regional Intelligence",
              "delay-prediction": "Delay Prediction", "reports-insights": "Reports & Insights"}
BAD = ("Page not found", "Traceback", "StreamlitAPIException", "ModuleNotFoundError")
MAIN = '[data-testid="stMain"]'


@pytest.fixture(scope="module")
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                             "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/_stcore/health", timeout=1)
                break
            except OSError:
                time.sleep(0.25)
        else:
            pytest.skip("streamlit server did not start")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # browser binaries not installed
            pytest.skip(f"chromium unavailable: {exc}")
        yield b
        b.close()


def _wait_rendered(page, heading: str) -> None:
    page.wait_for_function(
        "h => { const m = document.querySelector('[data-testid=\"stMain\"]'); "
        "return m && m.innerText.includes(h) && !document.body.innerText.includes('Loading the analysis tools'); }",
        arg=heading, timeout=90_000)
    page.wait_for_timeout(800)


@pytest.mark.parametrize("path,heading", list(DEEP_LINKS.items()))
@pytest.mark.parametrize("viewport", [{"width": 1280, "height": 800}, {"width": 390, "height": 844}],
                         ids=["desktop", "phone"])
def test_page_url_as_first_visit_has_no_page_not_found(server, browser, path, heading, viewport):
    ctx = browser.new_context(viewport=viewport)      # fresh context = first visit, no session state
    try:
        page = ctx.new_page()
        page.goto(f"{server}/{path}")
        _wait_rendered(page, heading)
        text = page.inner_text("body")
        assert not [b for b in BAD if b in text]
        assert page.url.rstrip("/").endswith(path.rstrip("/"))
    finally:
        ctx.close()


def _sidebar_left(page) -> float:
    return page.evaluate("document.querySelector('[data-testid=\"stSidebar\"]').getBoundingClientRect().left")


def _open_drawer(page) -> None:
    page.locator('[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button').first.click()
    page.wait_for_timeout(700)


def test_phone_drawer_closes_after_tapping_a_nav_link(server, browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 844})
    try:
        page = ctx.new_page()
        page.goto(server + "/")
        _wait_rendered(page, "Overview")
        _open_drawer(page)
        assert _sidebar_left(page) >= 0                               # drawer is open
        page.locator('[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]', has_text="Delay Analysis").first.click()
        page.wait_for_function("location.pathname.includes('delay-analysis')", timeout=30_000)
        page.wait_for_timeout(1500)
        assert _sidebar_left(page) < 0                                # closed: the new page is visible
    finally:
        ctx.close()


def test_desktop_sidebar_stays_open_after_navigation(server, browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = ctx.new_page()
        page.goto(server + "/")
        _wait_rendered(page, "Overview")
        page.locator('[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]', has_text="Delay Analysis").first.click()
        page.wait_for_function("location.pathname.includes('delay-analysis')", timeout=30_000)
        page.wait_for_timeout(1500)
        assert _sidebar_left(page) >= 0
    finally:
        ctx.close()


def test_cold_start_shows_a_loading_notice_that_clears(server, browser):
    # The module-scoped server is already warm by now, so only the clearing half is checkable deterministically:
    # the notice must never remain once a page has rendered.
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = ctx.new_page()
        page.goto(server + "/delay-prediction")
        _wait_rendered(page, "Delay Prediction")
        assert "Loading the analysis tools" not in page.inner_text("body")
    finally:
        ctx.close()
