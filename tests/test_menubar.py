"""
The menu bar, and the confirmation on the one action that cannot be undone.

Import and delete used to be nine 26px buttons wedged around three dropdowns
in a window meant to be glanceable during a meeting, for actions used a
handful of times ever. Clear Transcript was a button between two harmless
ones and asked nothing before discarding the meeting.
"""
import os
import queue
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import app  # noqa: E402


# --------------------------------------------------------------------------
# Clearing the transcript
# --------------------------------------------------------------------------

class _FakeTranscriber:
    def __init__(self):
        self.structured_transcript = {"speaker": [1, 2, 3], "you": [4]}
        self.cleared = False

    def clear_transcript_data(self):
        self.cleared = True


class _FakeUI:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


def _clear(monkeypatch, answer, **kw):
    asked = {}

    def askyesno(title, message):
        asked["title"], asked["message"] = title, message
        return answer

    monkeypatch.setattr(app, "messagebox",
                        types.SimpleNamespace(askyesno=askyesno))
    t, ui = _FakeTranscriber(), _FakeUI()
    app.clear_context(t, queue.Queue(), queue.Queue(), ui, **kw)
    return t, ui, asked


def test_declining_the_confirmation_keeps_the_transcript(monkeypatch):
    t, ui, asked = _clear(monkeypatch, answer=False)
    assert asked, "it must ask before discarding a meeting"
    assert not t.cleared and not ui.cleared


def test_accepting_the_confirmation_clears_it(monkeypatch):
    t, ui, _ = _clear(monkeypatch, answer=True)
    assert t.cleared and ui.cleared


def test_the_question_says_how_much_is_being_discarded(monkeypatch):
    """"Are you sure?" is not a question anyone can answer."""
    _, _, asked = _clear(monkeypatch, answer=False)
    assert "4 turns" in asked["message"], asked["message"]
    assert "cannot be undone" in asked["message"].lower()


def test_the_answers_are_cleared_with_the_transcript(monkeypatch):
    responder = types.SimpleNamespace(answers=["a", "b"], response="b")
    _clear(monkeypatch, answer=True, responder=responder)
    assert responder.answers == [] and responder.response == ""


# --------------------------------------------------------------------------
# The menu
# --------------------------------------------------------------------------

@pytest.fixture
def menubar():
    ctk = pytest.importorskip("customtkinter")
    try:
        root = ctk.CTk()
    except Exception as e:                      # no display
        pytest.skip(f"no Tk display: {e}")
    calls = []
    stock = {"system_role": ["inbound_cs", "default"],
             "case_detail": ["inbound_cs"],
             "knowledge": []}
    mb = app._build_menubar(
        root,
        on_export=lambda: calls.append("export"),
        on_clear=lambda: calls.append("clear"),
        importers={k: (lambda k=k: calls.append(f"import:{k}")) for k in stock},
        removers={k: (lambda n, k=k: calls.append(f"delete:{k}:{n}"))
                  for k in stock},
        editors={k: (lambda n, k=k: calls.append(f"edit:{k}:{n}"))
                 for k in stock},
        list_templates=lambda k: stock[k])
    root.update()
    yield root, mb, calls, stock
    root.destroy()


def _labels(m):
    return [m.entrycget(i, "label") if m.type(i) != "separator" else "---"
            for i in range(m.index("end") + 1)]


def _menu(mb, name):
    return mb.nametowidget(mb.entrycget(name, "menu"))


def test_the_file_menu_carries_import_manage_and_export(menubar):
    _, mb, _, _ = menubar
    labels = " | ".join(_labels(_menu(mb, "File")))
    for expected in ("Import", "Manage Templates", "Export Conversation"):
        assert expected in labels, f"{expected!r} missing from {labels}"


def test_import_picks_the_category_from_a_submenu(menubar):
    _, mb, calls, _ = menubar
    f = _menu(mb, "File")
    assert f.type(f.index("Import")) == "cascade"
    imp = _menu(f, "Import")
    assert _labels(imp) == ["System Role…", "Case Detail…", "Knowledge Base…"]
    imp.invoke(imp.index("Case Detail…"))
    assert calls == ["import:case_detail"]


def test_manage_goes_category_then_template_then_action(menubar):
    _, mb, calls, _ = menubar
    manage = _menu(_menu(mb, "File"), "Manage Templates")
    assert _labels(manage) == ["System Role", "Case Detail", "Knowledge Base"]

    category = _menu(manage, "System Role")
    assert _labels(category) == ["inbound_cs", "default"]

    actions = _menu(category, "default")
    assert _labels(actions) == ["Edit…", "Delete…"]
    actions.invoke(actions.index("Delete…"))
    actions.invoke(actions.index("Edit…"))
    assert calls == ["delete:system_role:default", "edit:system_role:default"]


def test_a_category_with_nothing_in_it_says_so(menubar):
    """An empty submenu reads as broken; a disabled line reads as empty."""
    _, mb, _, _ = menubar
    empty = _menu(_menu(_menu(mb, "File"), "Manage Templates"),
                  "Knowledge Base")
    assert _labels(empty) == ["(none imported)"]
    assert empty.entrycget(0, "state") == "disabled"


def test_the_template_list_is_rebuilt_when_the_menu_opens(menubar):
    """
    Built once, it would list what existed at startup: a template imported
    afterwards would not be there, and a deleted one would still be offered.
    """
    _, mb, _, stock = menubar
    manage = _menu(_menu(mb, "File"), "Manage Templates")
    stock["knowledge"].append("pricing_faq")
    stock["system_role"].remove("default")

    manage.tk.call(manage.cget("postcommand"))       # what opening it does

    assert _labels(_menu(manage, "Knowledge Base")) == ["pricing_faq"]
    assert _labels(_menu(manage, "System Role")) == ["inbound_cs"]


def test_clearing_is_in_edit_not_file(menubar):
    """
    It is not a file operation, and it sat among three that are -- each of
    which opens a file dialog. macOS keeps Clear next to Select All.
    """
    _, mb, _, _ = menubar
    assert "Clear Transcript" not in " ".join(_labels(_menu(mb, "File")))
    assert "Clear Transcript" in " ".join(_labels(_menu(mb, "Edit")))


def test_copy_and_select_all_are_listed(menubar):
    """
    Tk has always bound these at the Text class level, and nothing on screen
    said so. An undiscoverable feature is close to an absent one.
    """
    _, mb, _, _ = menubar
    labels = _labels(_menu(mb, "Edit"))
    assert "Copy" in labels and "Select All" in labels


def test_copy_is_not_bound_a_second_time(menubar):
    """
    Tk resolves <<Copy>> to <Mod1-Key-c> itself. Adding bind_all for the same
    key would run the copy twice on one keypress.
    """
    import inspect
    from src import app
    body = inspect.getsource(app._build_menubar)
    assert "<Command-c>" not in body and "<Command-a>" not in body


def test_a_menu_command_acts_on_the_focused_pane(menubar):
    """There are two text panes; the menu has to act on the one in use."""
    root, _, _, _ = menubar
    import customtkinter as ctk
    box = ctk.CTkTextbox(root)
    box.pack()
    box.insert("1.0", "pick me")
    box._textbox.focus_force()
    root.update()
    from src import app
    app._send_to_focus(root, "<<SelectAll>>")()
    root.update()
    assert box._textbox.tag_ranges("sel"), "nothing was selected"





def test_the_export_shortcut_is_bound_and_not_only_drawn(menubar):
    """
    `accelerator=` only draws the key beside the label. Without the matching
    bind_all the menu advertises a shortcut that does nothing, which is worse
    than showing none.
    """
    root, mb, calls, _ = menubar
    f = _menu(mb, "File")
    assert f.entrycget("Export Conversation…", "accelerator") == "Command-E"
    root.event_generate("<Command-e>")
    root.update()
    assert "export" in calls, "the accelerator label is not a binding"


def test_clearing_has_no_shortcut(menubar):
    """It discards a meeting. A shortcut is a thing you hit by accident."""
    _, mb, _, _ = menubar
    e = _menu(mb, "Edit")
    assert e.entrycget("Clear Transcript…", "accelerator") == ""


def test_the_import_and_delete_buttons_are_gone_from_the_panel():
    import inspect
    body = inspect.getsource(app.create_ui_components)
    assert 'text="+"' not in body
    assert 'text="\\u2212"' not in body
    assert "Export Conversation" not in body.split("buttons_data")[1][:400]
