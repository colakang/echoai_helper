"""
Prompt templates: importing them, finding them, and when they matter.

Three things that were wrong together, and are easier to understand as one
story than as three:

- there was no way to add a template except by putting a file in the right
  directory, and once installed from a wheel that directory is inside
  site-packages;
- the controls for choosing one were live in meeting mode, where nothing reads
  what they select;
- and the machinery they feed was gated on a checkbox rather than on the mode,
  so meeting mode could quietly spend money on replies nobody would see.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.TemplateManager import TemplateManager
from src.config import AudioConfig, PathConfig, SystemConfig


@pytest.fixture
def user_templates(tmp_path, monkeypatch):
    """Point the user template directory somewhere disposable."""
    monkeypatch.setattr(PathConfig, "get_user_config_path",
                        staticmethod(lambda: str(tmp_path)))
    return tmp_path / "prompt"


# --------------------------------------------------------------------------
# Importing
# --------------------------------------------------------------------------

def test_an_imported_template_lands_outside_the_package(user_templates, tmp_path):
    """
    Not beside the shipped ones. Installed from a wheel those live in
    site-packages: writing there is wrong, and a reinstall would delete
    whatever the user had added without mentioning it.
    """
    source = tmp_path / "my_persona.txt"
    source.write_text("You are a helpful interviewer.", encoding="utf-8")

    name = TemplateManager.import_template("case_detail", str(source))

    assert name == "my_persona"
    landed = user_templates / "case_detail" / "my_persona.txt"
    assert landed.exists()
    assert "site-packages" not in str(landed)


def test_importing_does_not_silently_overwrite(user_templates, tmp_path):
    """
    Someone re-importing a familiar filename should not lose the template they
    spent an afternoon writing. A suffix costs a moment of confusion; the
    alternative costs the file.
    """
    source = tmp_path / "notes.txt"
    source.write_text("first", encoding="utf-8")
    first = TemplateManager.import_template("knowledge", str(source))

    source.write_text("second", encoding="utf-8")
    second = TemplateManager.import_template("knowledge", str(source))

    assert first == "notes"
    assert second == "notes_1"
    assert (user_templates / "knowledge" / "notes.txt").read_text() == "first"


def test_an_imported_template_appears_in_the_list(user_templates, tmp_path):
    source = tmp_path / "extra.txt"
    source.write_text("content", encoding="utf-8")
    TemplateManager.import_template("knowledge", str(source))

    assert "extra" in TemplateManager.get_template_files("knowledge")


def test_shipped_templates_are_still_listed(user_templates):
    """Importing must add to what is available, not replace it."""
    assert "none" in TemplateManager.get_template_files("knowledge")
    assert "inbound_cs" in TemplateManager.get_template_files("system_role")


def test_a_user_template_shadows_a_shipped_one_of_the_same_name(user_templates, tmp_path):
    """Someone who names theirs after a shipped template means theirs."""
    source = tmp_path / "none.txt"
    source.write_text("mine", encoding="utf-8")
    TemplateManager.import_template("knowledge", str(source))

    resolved = TemplateManager.resolve_template("knowledge", "none")
    assert str(user_templates) in resolved
    assert open(resolved, encoding="utf-8").read() == "mine"
    # And it is offered once, not twice.
    assert TemplateManager.get_template_files("knowledge").count("none") == 1


def test_importing_a_missing_file_fails_quietly(user_templates):
    assert TemplateManager.import_template("knowledge", "/no/such/file") is None


def test_importing_into_an_unknown_category_fails(user_templates, tmp_path):
    source = tmp_path / "x.txt"
    source.write_text("x", encoding="utf-8")
    assert TemplateManager.import_template("not_a_category", str(source)) is None


# --------------------------------------------------------------------------
# When the templates actually matter
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_mode():
    profile, record_only = AudioConfig.get_profile(), SystemConfig.get_record_only_mode()
    yield
    AudioConfig.set_profile(profile)
    SystemConfig.set_record_only_mode(record_only)


def test_meeting_mode_never_asks_for_replies():
    """
    The bug this closes. Reply suggestions were gated only on the Record Only
    checkbox, so a user in meeting mode who unticked it began paying for a
    model call on every sentence the far end spoke -- with nothing on screen
    to say it was happening, and the answers arriving in a pane they were not
    looking at.
    """
    AudioConfig.set_profile("meeting")
    SystemConfig.set_record_only_mode(False)
    assert not AudioConfig.replies_enabled()


def test_interview_mode_asks_for_replies():
    AudioConfig.set_profile("interview")
    SystemConfig.set_record_only_mode(False)
    assert AudioConfig.replies_enabled()


def test_record_only_still_overrides_inside_interview_mode():
    """Mode is the outer gate; the checkbox remains meaningful within it."""
    AudioConfig.set_profile("interview")
    SystemConfig.set_record_only_mode(True)
    assert not AudioConfig.replies_enabled()


def test_the_transcriber_uses_that_gate_and_not_the_checkbox_alone():
    """
    Pinned at the call site too. The greyed-out controls are only honest if
    the machinery behind them is genuinely off; a version that disabled the
    dropdowns and left the responder running would be worse than one that did
    neither, because it would look fixed.
    """
    import inspect
    from src import AudioTranscriber as module
    body = inspect.getsource(module.AudioTranscriber._finalize_segment)
    assert "replies_enabled()" in body
    assert "get_record_only_mode()" not in body


def test_the_ui_disables_the_templates_outside_interview_mode():
    import inspect
    from src import app
    body = inspect.getsource(app.create_ui_components)
    assert "_apply_template_availability" in body
    handler = body.split("def _apply_template_availability", 1)[1][:800]
    assert 'profile.key == "interview"' in handler
    assert '"disabled"' in handler
    # Both the dropdowns and the import buttons, or the user can still import
    # into a mode that will never read it.
    assert "template_menus" in handler and "template_imports" in handler


# --------------------------------------------------------------------------
# Braces in an imported template
# --------------------------------------------------------------------------
#
# Only two placeholders are supported, and they are substituted by name. The
# code used to call str.format(), which interprets every brace in the file.
# That was survivable while the only templates were the two shipped ones, and
# stopped being survivable the moment templates could be imported -- a prompt
# is exactly the kind of text that contains braces on purpose.
#
# Each of these used to leave the role unchanged while the dropdown showed the
# new selection, so the app went on using the previous persona while appearing
# to have switched.

@pytest.mark.parametrize("hazard,description", [
    ('{"role": "user"}', "a JSON example in the prompt"),
    ("{customer_name}", "a placeholder meant literally"),
    ("use { for grouping", "an unmatched brace"),
    ("}{", "backwards braces"),
    ("{{escaped}}", "doubled braces meant as literal text"),
])
def test_braces_in_a_template_do_not_break_it(user_templates, tmp_path,
                                              hazard, description):
    source = tmp_path / "persona.py"
    source.write_text(f'ROLE = """{hazard}\n{{case_detail}}\n{{knowledge}}"""',
                      encoding="utf-8")
    name = TemplateManager.import_template("system_role", str(source))

    role = TemplateManager.update_system_role(name, "none", "none")

    assert role is not None, f"{description} broke the template"
    assert hazard in role, "the prompt must reach the model exactly as written"


def test_the_two_supported_placeholders_are_still_substituted(user_templates, tmp_path):
    source = tmp_path / "persona.py"
    source.write_text(
        'ROLE = """Case: {case_detail}\nKnows: {knowledge}"""', encoding="utf-8")
    name = TemplateManager.import_template("system_role", str(source))

    detail = tmp_path / "d.txt"; detail.write_text("DETAIL", encoding="utf-8")
    know = tmp_path / "k.txt"; know.write_text("KNOWLEDGE", encoding="utf-8")
    TemplateManager.import_template("case_detail", str(detail))
    TemplateManager.import_template("knowledge", str(know))

    role = TemplateManager.update_system_role(name, "d", "k")

    assert "Case: DETAIL" in role
    assert "Knows: KNOWLEDGE" in role
    assert "{case_detail}" not in role and "{knowledge}" not in role


def test_a_failed_switch_is_shown_on_screen():
    """
    Not only logged. A combination that cannot be loaded leaves the previous
    persona in force while the dropdown shows the new one -- the app looks
    changed and is not.
    """
    import inspect
    from src import app
    body = inspect.getsource(app.create_ui_components)
    handler = body.split("def on_selection_change", 1)[1][:1200]
    assert "showwarning" in handler or "showerror" in handler
