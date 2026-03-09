"""Minimal streamlit stub for running tests that import Streamlit files.

This is a tests-only lightweight replacement that implements
- st.session_state as a simple attribute/mapping object
- title, markdown, info: no-ops
- sidebar context manager providing markdown, columns, button, selectbox
- columns returning simple context managers (usable with `with col:`)
- button/selectbox returning defaults so import-time UI code doesn't fail

This is intentionally small and not a real Streamlit implementation.
"""
from types import SimpleNamespace


class SessionState:
    def __init__(self, initial=None):
        self._data = dict(initial or {})

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

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)


# global session_state
session_state = SessionState({
    'current_focus_tab': 'retail',
    'current_focus_header': 'retail',
})


# top-level UI functions (no-ops)
def title(text):
    return None


def markdown(text):
    return None


def info(text):
    return None


class _Column:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def columns(n):
    # return n simple column context managers
    return tuple(_Column() for _ in range(n))


class _Sidebar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def markdown(self, text):
        return None

    def columns(self, n):
        return columns(n)


# button returns False by default (no clicks during import)
def button(label, key=None, type=None, use_container_width=False):
    return False


# selectbox returns first option
def selectbox(label, options, key=None):
    if options:
        return options[0]
    return None


# provide a context manager `sidebar` to be used as `with st.sidebar:`
class _SidebarContext:
    def __enter__(self):
        return _Sidebar()

    def __exit__(self, exc_type, exc, tb):
        return False


sidebar = _SidebarContext()

# expose commonly used names
st = SimpleNamespace(
    session_state=session_state,
    title=title,
    markdown=markdown,
    info=info,
    sidebar=sidebar,
    columns=columns,
    button=button,
    selectbox=selectbox,
)

