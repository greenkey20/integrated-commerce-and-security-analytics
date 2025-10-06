"""Pytest configuration hooks for the project.

This file sets up a tests-friendly `st.session_state` so tests that import
streamlit and access session_state at import time do not fail when running
`pytest` (Streamlit's session state normally requires `streamlit run`).
"""
from types import SimpleNamespace


class TestSessionState:
    """A tiny object that behaves both like a mapping and an attribute container.

    Supports:
    - 'key' in st.session_state
    - st.session_state.key = value
    - st.session_state['key'] = value
    - st.session_state.get(key, default)
    """

    def __init__(self, initial=None):
        object.__setattr__(self, "_data", dict(initial or {}))

    def __contains__(self, key):
        return key in self._data

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    # mapping APIs
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()


def _init_streamlit_state_once():
    try:
        import streamlit as st
        try:
            # If accessing or checking raises, replace session_state
            _ = 'current_focus_tab' in st.session_state
        except Exception:
            st.session_state = TestSessionState({
                'current_focus_tab': 'retail',
                'current_focus_header': 'retail',
            })
    except Exception:
        # streamlit not installed in environment; nothing to do
        pass


# initialize immediately when conftest is imported
_init_streamlit_state_once()


def pytest_configure(config):
    # Initialize Streamlit session state early for tests that import streamlit
    try:
        import streamlit as st

        # Only overwrite if session_state doesn't behave as expected
        try:
            has = 'current_focus_tab' in st.session_state
            # If this did not raise, assume session_state is usable
        except Exception:
            # Replace with our tests-friendly object
            st.session_state = TestSessionState({
                'current_focus_tab': 'retail',
                'current_focus_header': 'retail',
            })
    except Exception:
        # If streamlit can't be imported in this environment, skip silently.
        pass
