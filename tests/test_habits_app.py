from streamlit.testing.v1 import AppTest


def test_habits_app_smoke_test():
    """A simple smoke test that runs the habits app to ensure it doesn't crash."""
    at = AppTest.from_file("src/pages/habits.py")
    at.run(timeout=30)
    assert not at.exception, "".join(map(str, at.exception))
