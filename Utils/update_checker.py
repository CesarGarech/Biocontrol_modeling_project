"""
GitHub Releases version checker for the Biocontrol Dashboard.

Checks the GitHub Releases API for newer versions and displays a
notification banner in the Streamlit sidebar.
"""

import streamlit as st

GITHUB_API = "https://api.github.com/repos/CesarGarech/Biocontrol_modeling_project/releases/latest"

try:
    from version import __version__ as CURRENT_VERSION
except Exception:
    CURRENT_VERSION = "1.0.0"


def check_for_updates():
    """Check the GitHub Releases API for a newer version.

    Returns a dict:
        {
            "available": bool,
            "current": str,
            "latest": str,
            "url": str,
        }
    All exceptions are silently caught so the app never crashes.
    """
    result = {
        "available": False,
        "current": CURRENT_VERSION,
        "latest": CURRENT_VERSION,
        "url": "",
    }
    try:
        import requests

        response = requests.get(GITHUB_API, timeout=5)
        response.raise_for_status()
        data = response.json()
        latest_tag = data.get("tag_name", "").lstrip("v")
        release_url = data.get("html_url", "")

        result["latest"] = latest_tag or CURRENT_VERSION
        result["url"] = release_url

        if latest_tag and latest_tag != CURRENT_VERSION:
            result["available"] = True
    except Exception:
        pass

    return result


def show_update_banner():
    """Display a sidebar warning if a newer version is available.

    If no update is available, this function does nothing.
    """
    info = check_for_updates()
    if info["available"]:
        st.sidebar.warning(
            f"🔔 **Update available!**\n\n"
            f"Current version: `{info['current']}`\n\n"
            f"Latest version: `{info['latest']}`\n\n"
            "**How to update:**\n\n"
            "- **Option A (recommended):** Run `update_dashboard.bat` "
            "(Windows) or `update_dashboard.sh` (Linux/Mac) in the "
            "installation folder.\n\n"
            f"- **Option B:** [Download the new installer]({info['url']}) "
            "from GitHub Releases."
        )
