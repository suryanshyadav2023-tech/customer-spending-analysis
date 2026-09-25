"""Smoke test for the Streamlit demo: every page renders without an exception, and the Customer page
reproduces the hurdle predictions of outputs/report/worked_example.md for Users A/B/C.

Run from the project root:  python tests/test_app.py
"""
import re
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def run(x):
    """x is an AppTest or a widget after set_value(); both .run() and return the AppTest."""
    at = x.run()
    if at.exception:
        raise AssertionError("\n".join(str(e.value) for e in at.exception))
    return at


def main():
    at = run(AppTest.from_file(str(ROOT / "app.py"), default_timeout=300))
    print("Overview page: ok,", len(at.markdown), "markdown blocks")

    at = run(at.radio(key="page").set_value("Customer"))
    expected = dict(re.findall(r"## (User [ABC]):.*?Hurdle prediction: p x E = .*? = ([\d,.]+)",
                               (ROOT / "outputs" / "report" / "worked_example.md").read_text(encoding="utf-8"),
                               flags=re.S))
    for user in ["User A", "User B", "User C"]:
        at = run(at.selectbox(key="user").set_value(user))
        shown = {m.label: m.value for m in at.metric}
        want = f"{float(expected[user].replace(',', '')):,.2f}"
        assert shown["Hurdle prediction"] == want, (user, shown["Hurdle prediction"], want)
        print(f"Customer page, {user}: ok (segment {shown['Segment (K-means)']}, hurdle {shown['Hurdle prediction']} "
              f"= worked_example.md)")
    at = run(at.radio(key="mode").set_value("Test-set index"))
    for i in [0, 12345, 51115]:
        at = run(at.number_input(key="test_index").set_value(i))
        print(f"Customer page, test-set index {i}: ok")

    at = run(at.radio(key="page").set_value("Segments"))
    for s in at.selectbox(key="segment").options:
        at = run(at.selectbox(key="segment").set_value(s))
        print(f"Segments page, {s}: ok")
    print("all pages loaded without errors")


if __name__ == "__main__":
    sys.exit(main())
