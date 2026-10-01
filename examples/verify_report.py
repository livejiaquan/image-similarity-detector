"""Browser acceptance checks for the included synthetic demo report.

Requires playwright and its Chromium headless shell. Does not request remote resources.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright


def verify() -> None:
    report = Path("docs/demo/report.html").resolve()
    assets = Path("docs/assets")
    assets.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    requests: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: errors.append(message.text) if message.type == "error" else None,
        )
        page.on("request", lambda request: requests.append(request.url))
        page.goto(report.as_uri())
        assert page.locator(".match-card:visible").count() == 6
        for label, count in [("Exact", 2), ("Near", 4), ("Cross-root", 4)]:
            page.get_by_role("button", name=label, exact=True).click()
            assert page.locator(".match-card:visible").count() == count
        page.get_by_role("button", name="All matches", exact=True).click()
        page.locator("#search").fill("copy-01")
        assert page.locator(".match-card:visible").count() == 2
        page.locator("#search").fill("no-such-file")
        assert page.locator("#filter-empty").is_visible()
        page.locator("#search").fill("")
        page.evaluate("window.scrollTo(0,0)")
        page.screenshot(path=str(assets / "report.png"))
        for width in [390, 720, 1000]:
            page.set_viewport_size({"width": width, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            if width == 390:
                page.screenshot(path=str(assets / "report-mobile.png"), full_page=True)
        assert all(not url.startswith(("http:", "https:")) for url in requests)
        assert not errors, errors
        browser.close()
    print("Browser checks passed: filters, search, responsive layout, CSP and offline resources.")


if __name__ == "__main__":
    verify()
