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


@pytest.mark.parametrize("profile", ["meeting", "interview"])
def test_replies_follow_the_switch_in_every_mode(profile):
    """
    Answering during a meeting is a supported use, not an accident.

    This briefly also required interview mode, on the reasoning that a longer
    pause makes an answer arrive too late to matter. That removed the ability
    to answer semi-automatically during a meeting -- one of the three things
    this app is for. A later answer is later, not useless, and whether it is
    worth having is the user's call.
    """
    AudioConfig.set_profile(profile)
    SystemConfig.set_record_only_mode(False)
    assert AudioConfig.replies_enabled()

    SystemConfig.set_record_only_mode(True)
    assert not AudioConfig.replies_enabled()


def test_replies_are_off_unless_asked_for():
    """
    Nothing is spent by anyone who has not turned them on. The shipped default
    suppresses replies; enabling them is a deliberate act in either mode.
    """
    import json
    from pathlib import Path
    from src.config import PathConfig
    shipped = json.loads(
        (Path(PathConfig.get_config_path()) / "settings.json").read_text())
    assert shipped["record_only_mode"] is True


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


def test_the_template_controls_are_never_disabled():
    """
    They were greyed out twice, on two different conditions, and both were
    wrong in practice.

    Keying them to the mode disabled the persona during a meeting, where
    answering is a supported use. Keying them to the replies switch disabled
    them out of the box, because replies ship off -- so the controls for
    configuring replies were unusable until you found the switch that turned
    on the thing they configure, with nothing on screen saying so.

    Configuring a persona is harmless whether or not replies are on. A hint
    says what they affect; nothing needs hiding.
    """
    import inspect
    from src import app
    body = inspect.getsource(app.create_ui_components)
    assert "_apply_template_availability" not in body
    assert "used for the reply suggestions" in body


# --------------------------------------------------------------------------
# Braces in an imported template
# --------------------------------------------------------------------------
#
# Only two placeholders are supported, substituted by name. The code used to
# call str.format(), which interprets every brace in the file -- survivable
# while the only templates were the shipped two, and not once templates could
# be imported: a prompt is exactly the kind of text that contains braces on
# purpose. Each case below used to leave the role unchanged while the dropdown
# showed the new selection.

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


# --------------------------------------------------------------------------
# Choosing the model
# --------------------------------------------------------------------------

from src.config import LLMConfig                                   # noqa: E402


@pytest.fixture(autouse=True)
def restore_model():
    previous = LLMConfig._model
    yield
    LLMConfig._model = previous


def test_the_shipped_default_is_offered():
    assert LLMConfig.configured_model()
    assert LLMConfig.configured_model() in LLMConfig.available_models()


def test_choosing_a_model_overrides_the_file():
    LLMConfig.set_model("some-other-model")
    assert LLMConfig.get_model() == "some-other-model"


def test_clearing_the_choice_falls_back_to_the_file():
    LLMConfig.set_model("temporary")
    LLMConfig.set_model(None)
    assert LLMConfig.get_model() == LLMConfig.configured_model()


def test_the_current_model_is_always_in_the_menu(monkeypatch):
    """
    Even when the list omits it. A menu that cannot show what is actually in
    use is worse than no menu: the user reads the wrong answer off it.
    """
    monkeypatch.setattr(LLMConfig, "_llm_section",
                        classmethod(lambda cls: {"models": ["a", "b"]}))
    LLMConfig.set_model("something-not-listed")
    assert LLMConfig.available_models()[0] == "something-not-listed"


def test_no_model_name_is_invented_in_code(monkeypatch):
    """
    If conf.yaml lists nothing and configures nothing, the menu is empty rather
    than populated with names the code made up. Model names change faster than
    releases do; one baked into the source is stale the week after it ships.
    """
    monkeypatch.setattr(LLMConfig, "_llm_section", classmethod(lambda cls: {}))
    LLMConfig.set_model(None)
    assert LLMConfig.available_models() == []


def test_the_list_comes_from_configuration_not_source():
    """The shipped list lives in conf.yaml, where it can be edited without a release."""
    import yaml
    from pathlib import Path
    from src.config import PathConfig
    config = yaml.safe_load(Path(PathConfig.get_conf_file()).read_text(encoding="utf-8"))
    assert config["LLM"]["models"], "conf.yaml must carry the menu's options"


def test_both_consumers_use_the_same_choice():
    """
    The live replies and the cleanup pass at export must not end up on
    different models. They are built in different places and at different
    times, which is exactly how that would happen unnoticed.
    """
    import inspect
    from src import app, GPTResponder as responder_module
    assert "LLMConfig.get_model()" in inspect.getsource(app._build_polish_provider)
    assert "LLMConfig.get_model()" in inspect.getsource(
        responder_module.GPTResponder._initialize_llm_provider)


def test_the_live_provider_can_be_rebuilt():
    """
    Built once at construction, so without this a model chosen from the menu
    would apply to export and silently not to the replies.
    """
    from src.GPTResponder import GPTResponder
    assert hasattr(GPTResponder, "reload_provider")


def test_the_chosen_model_survives_a_restart(tmp_path, monkeypatch):
    """
    The bug this closes. The saved choice was applied while building the UI,
    and the responder builds its provider *before* that -- so the model was
    persisted correctly and then ignored on every restart, with the menu
    showing one model and the provider using another.

    Loading on first use removes the ordering question rather than answering
    it, which matters because the answer was not obvious and not stable.
    """
    from src.SettingsManager import SettingsManager

    monkeypatch.setattr(SettingsManager, "get_setting",
                        lambda self, key, default=None:
                        "saved-model" if key == "llm_model" else default)
    LLMConfig._model = None
    LLMConfig._loaded = False

    # Nothing has touched the UI. A fresh process asking the question directly
    # -- which is what GPTResponder does at construction -- must see the
    # saved choice.
    assert LLMConfig.get_model() == "saved-model"


def test_loading_the_saved_model_happens_once(monkeypatch):
    """Reading settings on every call would put file IO on the response path."""
    from src.SettingsManager import SettingsManager
    calls = []
    monkeypatch.setattr(SettingsManager, "get_setting",
                        lambda self, key, default=None: calls.append(key) or None)
    LLMConfig._model = None
    LLMConfig._loaded = False

    LLMConfig.get_model(); LLMConfig.get_model(); LLMConfig.get_model()
    assert calls.count("llm_model") == 1


def test_the_ui_does_not_re_apply_the_saved_model():
    """It is loaded on demand; doing it in the UI is what made it too late."""
    import inspect
    from src import app
    body = inspect.getsource(app.create_ui_components)
    assert 'get_setting("llm_model")' not in body


# --------------------------------------------------------------------------
# Reaching backends that are not OpenAI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model,expected", [
    ("gpt-4o-mini", "openai"),
    ("gpt-5.6-luna", "openai"),
    ("gemini/gemini-2.0-flash", "litellm"),
    ("ollama/llama3", "litellm"),
    ("anthropic/claude-sonnet-4-5", "litellm"),
])
def test_the_model_name_picks_the_backend(model, expected):
    """
    One menu, not a provider dropdown beside a model dropdown.

    The two would only ever be set in valid combinations anyway, so making the
    user keep them in sync is asking them to maintain an invariant the app can
    see for itself. A vendor prefix -- which is how LiteLLM already names its
    models -- says everything needed.
    """
    assert LLMConfig.provider_for(model) == expected


def test_adding_a_backend_needs_no_code_change():
    """
    Reaching Gemini or a local Ollama is a line in conf.yaml. Pinned because
    the obvious alternative -- a provider enum in the source -- would need a
    release every time a backend appeared.
    """
    import inspect
    body = inspect.getsource(LLMConfig.provider_for)
    for vendor in ("gemini", "ollama", "anthropic", "azure"):
        assert vendor not in body.lower().split('"""')[2], \
            f"{vendor} must not be named in the routing logic"


def test_both_consumers_route_the_same_way():
    import inspect
    from src import app, GPTResponder as responder_module
    assert "provider_for()" in inspect.getsource(app._build_polish_provider)
    assert "provider_for()" in inspect.getsource(
        responder_module.GPTResponder._initialize_llm_provider)


# --------------------------------------------------------------------------
# Models that reject a temperature
# --------------------------------------------------------------------------

def test_a_rejected_temperature_is_recognised():
    """
    Measured against the live API: gpt-5.6-luna answers a 400 with
    "Unsupported value: 'temperature' does not support 0.6 with this model.
    Only the default (1) value is supported." while gpt-4o-mini and
    gpt-5.4-mini accept it.
    """
    from src.llm.openai_provider import OpenAIProvider as P
    rejection = ("Error code: 400 - {'error': {'message': \"Unsupported value: "
                 "'temperature' does not support 0.6 with this model. Only the "
                 "default (1) value is supported.\"}}")
    assert P._is_temperature_rejection(Exception(rejection))


def test_other_failures_are_not_mistaken_for_it():
    """Retrying without temperature would not help, and would double the bill."""
    from src.llm.openai_provider import OpenAIProvider as P
    for other in ("Error code: 404 - model_not_found",
                  "Rate limit reached",
                  "Connection error",
                  "Error code: 401 - invalid api key"):
        assert not P._is_temperature_rejection(Exception(other))


def test_no_model_is_named_in_the_temperature_handling():
    """
    Which models reject a temperature is learned, not listed. A list here
    would go stale exactly as fast as a list of model names, and discovering
    it costs one rejected request per model per process.
    """
    import inspect
    from src.llm import openai_provider
    body = inspect.getsource(openai_provider.OpenAIProvider.generate_response)
    assert "gpt-5" not in body and "gpt-4" not in body


def test_a_failure_reaching_the_user_names_the_model():
    """
    This lands in the response pane during a live interview. The raw SDK repr
    is several hundred characters of JSON; the usual cause is a model that does
    not exist or is not enabled for the account, so say which model.
    """
    from src.llm.openai_provider import OpenAIProvider
    p = OpenAIProvider(api_key="x", model="gpt-9-imaginary")
    shown = p._describe(Exception("The model does not exist"))
    assert "gpt-9-imaginary" in shown
    assert len(shown) < 260


def test_the_two_defaults_agree():
    """
    The code default and the shipped settings.json disagreed: the file said
    "no replies", the code said "replies". A user whose settings predated the
    key fell through to the code default and got the opposite of what shipping
    intended -- a model call on every sentence, unasked.
    """
    import json
    from pathlib import Path
    from src.SettingsManager import SettingsManager
    from src.config import PathConfig

    shipped = json.loads(
        (Path(PathConfig.get_config_path()) / "settings.json").read_text())
    for key in ("record_only_mode", "diarization", "speaker_count", "profile"):
        if key in shipped and key in SettingsManager.DEFAULT_SETTINGS:
            assert shipped[key] == SettingsManager.DEFAULT_SETTINGS[key], \
                f"{key}: shipped {shipped[key]!r} vs code " \
                f"{SettingsManager.DEFAULT_SETTINGS[key]!r}"


# --------------------------------------------------------------------------
# Answering on demand
# --------------------------------------------------------------------------
#
# The automatic path answers every finished utterance from the far end, which
# is what an interview wants: a prompt while the other person is still
# talking. A meeting is the other shape. Most of what is said needs no answer
# from you, and only you know which part does -- so the trigger has to be
# yours, and the input is the passage you point at rather than whatever
# arrived last.

def test_the_responder_can_be_asked_directly():
    from src.GPTResponder import GPTResponder
    assert hasattr(GPTResponder, "answer_passage")


def test_an_explicit_request_is_not_length_filtered():
    """
    The filter stops the automatic path answering "嗯" and "好的" -- a guess
    about whether an utterance was meant as a question. There is nothing to
    guess about when someone has highlighted the text and pressed a button.
    """
    import inspect
    from src.GPTResponder import GPTResponder
    manual = inspect.getsource(GPTResponder.answer_passage)
    assert "require_length=False" in manual

    auto = inspect.getsource(GPTResponder._answer)
    assert "if require_length and not _worth_answering" in auto


def test_it_answers_the_whole_selection():
    """
    A question in a meeting is usually spread over several turns. Answering
    only its last sentence answers the wrong question, so the button takes
    whatever is highlighted rather than a single line.
    """
    import inspect
    from src import app
    body = inspect.getsource(app.create_ui_components)
    handler = body.split("def answer_selection", 1)[1][:900]
    assert "selected_passage()" in handler


def test_the_selection_survives_pressing_the_button():
    """
    Tk hands the highlight to the system selection unless told not to, and
    clears its own sel tag the moment another widget claims it. Selecting a
    passage and pressing a button therefore destroyed the selection on the way
    to the handler, which reads as the button not working.
    """
    import inspect
    from src.TranscriptUI import TranscriptUI
    body = inspect.getsource(TranscriptUI._configure_textbox)
    assert "exportselection=False" in body


def test_a_remembered_selection_is_used_when_the_live_one_is_gone():
    """
    Belt and braces. "I highlighted that, then pressed the button" should not
    depend on focus behaviour at all -- a button that works or does not
    depending on where focus happens to be is indistinguishable from broken.
    """
    import inspect
    from src.TranscriptUI import TranscriptUI
    body = inspect.getsource(TranscriptUI.selected_passage)
    assert "_last_selection" in body


def test_an_empty_selection_says_so():
    """Silence would read as a broken button."""
    import inspect
    from src import app
    body = inspect.getsource(app.create_ui_components)
    handler = body.split("def answer_selection", 1)[1][:900]
    assert "showinfo" in handler


def test_asking_does_not_depend_on_the_automatic_switch():
    """
    The two are independent: the switch governs whether replies happen by
    themselves, and the button is an explicit request. Requiring the switch
    would mean turning on the thing you are trying to avoid in order to ask
    once.
    """
    import inspect
    from src.GPTResponder import GPTResponder
    manual = inspect.getsource(GPTResponder.answer_passage)
    assert "replies_enabled" not in manual
    assert "record_only" not in manual


def test_the_import_buttons_do_not_sit_on_the_action_buttons():
    """
    They were briefly gridded into column 1, where the action buttons live --
    small enough to still be clickable, which is how it survived being
    noticed. Each belongs beside its own dropdown.
    """
    import inspect
    from src import app
    body = inspect.getsource(app.create_ui_components)
    assert 'import_button.grid(row=row, column=0' in body
    assert 'btn.grid(row=i, column=1' in body


# --------------------------------------------------------------------------
# Reading and selecting while the transcript keeps arriving
# --------------------------------------------------------------------------
#
# A meeting appends continuously. Two things made that hostile to the one
# interaction that matters here -- highlighting a passage in order to answer
# it -- and both were in the update path rather than in the toolkit.

def test_the_view_does_not_jump_while_a_passage_is_being_selected():
    import inspect
    from src.TranscriptUI import TranscriptUI
    body = inspect.getsource(TranscriptUI)
    assert "if following_newest and not has_selection:" in body


def test_following_the_newest_means_being_near_the_top():
    """
    The list is newest first, so new lines arrive at the top. The old test
    asked whether the view was near the *end* of the document -- where the
    oldest lines are -- and so followed new arrivals precisely when the reader
    had scrolled back to read something older.
    """
    import inspect
    from src.TranscriptUI import TranscriptUI
    body = inspect.getsource(TranscriptUI)
    assert "yview()[0] <= 0.05" in body
    assert "was_at_bottom" not in body


def test_the_selection_is_not_restored_by_index():
    """
    Tk moves tags itself when text is inserted above them. Capturing
    sel.first/sel.last, inserting at "1.0", then re-adding sel at the captured
    indices pointed the highlight at whatever had shifted into those positions.
    """
    import inspect
    from src.TranscriptUI import TranscriptUI
    body = inspect.getsource(TranscriptUI)
    assert 'tag_add("sel", selection_start' not in body


def test_the_remembered_selection_is_text_not_a_position():
    """
    Which is what makes it survive the list moving underneath it: by the time
    the button is pressed, the indices mean something else.
    """
    import inspect
    from src.TranscriptUI import TranscriptUI
    body = inspect.getsource(TranscriptUI._remember_selection)
    assert 'get("sel.first", "sel.last")' in body
    assert "_last_selection = text" in body


def test_the_transcript_looks_selectable():
    """
    It carried a pointing-hand cursor, which says "click me". That is half of
    what the pane does and hides the half that matters here: dragging across
    turns is the only way to reach the on-demand answer, and nobody drags
    across something that presents itself as a list of buttons.
    """
    import inspect
    from src.TranscriptUI import TranscriptUI
    body = inspect.getsource(TranscriptUI._configure_textbox)
    assert 'cursor="xterm"' in body
    assert 'cursor="hand2"' not in body


def test_both_ways_of_choosing_are_written_down():
    """
    Neither is guessable, and there is nothing else that reveals them: the
    pane spent its life looking like a list of clickable rows.
    """
    import inspect
    from src import app
    body = inspect.getsource(app.create_ui_components)
    assert "drag" in body and "click turns" in body


def test_turns_can_be_picked_one_by_one():
    """
    Dragging cannot express the case that matters in a meeting: the question
    is spread over several of your counterpart's turns with somebody else's in
    between, and a contiguous selection has to take that somebody else along.
    """
    from src.TranscriptUI import TranscriptUI
    assert hasattr(TranscriptUI, "picked_turns")
    import inspect
    body = inspect.getsource(TranscriptUI._configure_textbox)
    assert "<Command-Button-1>" in body and "<Control-Button-1>" in body


def test_picking_is_manual_rather_than_by_label():
    """
    Not "select every turn labelled S1". The labels are not reliable enough to
    filter on -- measured: the same speaker reading out a number and talking
    scores as two different people -- so filtering by them would quietly drop
    half of what was wanted.
    """
    import inspect
    from src.TranscriptUI import TranscriptUI
    body = inspect.getsource(TranscriptUI._toggle_turn)
    assert "speaker" not in body.lower().split('"""')[-1]


def test_picked_turns_survive_the_meeting_carrying_on():
    """
    Read from the tag, not from remembered indices. Tk moves tags when text is
    inserted above them, so a pick made two minutes ago still points at the
    same words.
    """
    import inspect
    from src.TranscriptUI import TranscriptUI
    body = inspect.getsource(TranscriptUI.picked_turns)
    assert 'tag_ranges("picked")' in body


def test_a_passage_is_sent_in_the_order_it_was_spoken():
    """
    The pane is newest first, so anything lifted out of it reads backwards.
    A reversed multi-turn question is not obviously wrong to look at and
    quietly makes the question incoherent.
    """
    from src.TranscriptUI import TranscriptUI
    passage = "third\nsecond\nfirst"
    assert TranscriptUI._chronological(passage) == "first\nsecond\nthird"


def test_both_ways_of_choosing_are_reordered():
    """
    This was handled for picked turns and not for a dragged selection, so a
    dragged question arrived at the model in reverse. Getting it right in one
    of two places is how it came to be wrong; it happens in one place now.
    """
    import inspect
    from src.TranscriptUI import TranscriptUI
    body = inspect.getsource(TranscriptUI.selected_passage)
    assert body.count("_chronological(") == 3, \
        "picked turns, the live drag, and the remembered drag"
    assert "reversed" not in inspect.getsource(TranscriptUI.picked_turns)


def test_picks_win_over_a_drag():
    """Picking is the more deliberate act, and the only one that can leave a
    turn out of the middle."""
    import inspect
    from src.TranscriptUI import TranscriptUI
    body = inspect.getsource(TranscriptUI.selected_passage)
    assert body.index("picked_turns()") < body.index('get("sel.first"')


def test_picks_are_cleared_once_answered():
    """Or the next question silently inherits the last one's turns."""
    import inspect
    from src import app
    body = inspect.getsource(app.create_ui_components)
    assert "on_done=transcript_ui.clear_picks" in body
