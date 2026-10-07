"""
Unit tests for Streamlit helper functions and dashboard pages.
"""

import pytest
from app.utils import (
    get_page_config,
    PAGES,
    PAGE_ICONS,
    PAGE_TITLES,
    PAGE_SHORT,
    render_hero,
    render_disclaimer,
    inject_global_css,
)
from streamlit.testing.v1 import AppTest


def test_get_page_config():
    """Verify standard page configuration dictionary."""
    config = get_page_config()
    assert isinstance(config, dict)
    assert "AquaSense" in config["page_title"]
    assert config["page_icon"] == "🌊"
    assert config["layout"] == "wide"
    assert config["initial_sidebar_state"] == "expanded"


def test_pages_metadata_consistency():
    """Verify pages metadata structure and consistency."""
    assert len(PAGES) == 6
    for icon, title, short, path in PAGES:
        assert isinstance(icon, str)
        assert isinstance(title, str)
        assert isinstance(short, str)
        assert isinstance(path, str)
        assert path in PAGE_ICONS
        assert PAGE_ICONS[path] == icon
        assert PAGE_TITLES[path] == title
        assert PAGE_SHORT[path] == short


def test_app_dashboard_apptest():
    """Verify dashboard.py loads cleanly via AppTest."""
    at = AppTest.from_file("app/dashboard.py", default_timeout=20)
    at.run()
    assert not at.exception, f"dashboard.py raised exceptions: {at.exception}"


def test_pages_apptest():
    """Verify all Streamlit subpages load cleanly without exceptions."""
    pages = [
        "app/pages/1_data_story.py",
        "app/pages/2_statistical_eda.py",
        "app/pages/3_feature_importance.py",
        "app/pages/4_what_if_analysis.py",
        "app/pages/5_business_impact.py",
    ]
    for p in pages:
        at = AppTest.from_file(p, default_timeout=20)
        at.run()
        assert not at.exception, f"Page {p} raised exceptions: {at.exception}"
