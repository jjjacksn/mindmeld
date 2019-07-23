from unittest.mock import MagicMock

import pytest
from mindmeld.components import Conversation
from mindmeld.components.dialogue import DialogueFlow

from .test_dialogue import create_request, create_responder


def assert_reply(directives, templates, *, start_index=0, slots=None):
    """Asserts that the provided directives contain the specified reply

    Args:
        directives (list[dict[str, dict]]): list of directives returned by application
        templates (Union[str, Set[str]]): The reply must be a member of this set.
        start_index (int, optional): The index of the first client action associated
            with this reply.
        slots (dict, optional): The slots to fill the templates
    """
    slots = slots or {}
    if isinstance(templates, str):
        templates = [templates]

    texts = set(map(lambda x: x.format(**slots), templates))

    assert len(directives) >= start_index + 1
    assert directives[start_index]['name'] == 'reply'
    assert directives[start_index]['payload']['text'] in texts


def assert_target_dialogue_state(convo, target_dialogue_state):
    assert convo.params.target_dialogue_state == target_dialogue_state


@pytest.mark.conversation
def test_default_handler(async_kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests that the params are cleared in one trip from app to mm."""
    convo = Conversation(app=async_kwik_e_mart_app, app_path=kwik_e_mart_app_path, force_sync=True)
    convo.process('When does that open?')
    assert_target_dialogue_state(convo, 'send_store_hours_flow')
    directives = convo.process('are there any stores near me?').directives
    assert_target_dialogue_state(convo, 'send_store_hours_flow')
    assert_reply(directives,
                 templates='Sorry, I did not get you. Which store would you like to know about?')


@pytest.mark.conversation
def test_repeated_flow(async_kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests that the params are cleared in one trip from app to mm."""
    convo = Conversation(app=async_kwik_e_mart_app, app_path=kwik_e_mart_app_path, force_sync=True)
    convo.process('When does that open?')
    assert_target_dialogue_state(convo, 'send_store_hours_flow')
    for i in range(2):
        directives = convo.process('When does that open?').directives
        assert_reply(directives, 'Which store would you like to know about?')
        assert_target_dialogue_state(convo, 'send_store_hours_flow')
    directives = convo.process('When does that open?').directives
    assert_reply(directives, 'Sorry I cannot help you. Please try again.')
    assert_target_dialogue_state(convo, None)


@pytest.mark.conversation
def test_intent_handler_and_exit_flow(async_kwik_e_mart_app, kwik_e_mart_app_path):
    """Tests that the params are cleared in one trip from app to mm."""
    convo = Conversation(app=async_kwik_e_mart_app, app_path=kwik_e_mart_app_path, force_sync=True)
    convo.process('When does that open?')
    assert_target_dialogue_state(convo, 'send_store_hours_flow')
    directives = convo.process('exit').directives
    assert_target_dialogue_state(convo, None)
    assert_reply(directives, templates=['Bye', 'Goodbye', 'Have a nice day.'])


def assert_dialogue_state(dm, dialogue_state):
    for rule in dm.rules:
        if rule.dialogue_state == dialogue_state:
            return True
    return False


def test_dialogue_flow_async(async_kwik_e_mart_app):
    @async_kwik_e_mart_app.dialogue_flow(domain='some_domain', intent='some_intent')
    async def some_handler(request, responder):
        pass

    assert some_handler.flow_state == 'some_handler_flow'
    assert 'some_handler' in some_handler.all_flows

    dm = some_handler.dialogue_manager
    assert_dialogue_state(dm, 'some_handler')
    assert_dialogue_state(dm, 'some_handler_flow')

    assert len(some_handler.rules) == 0

    @some_handler.handle(intent='some_intent')
    async def some_flow_handler(request, responder):
        pass

    assert len(some_handler.rules) == 1

    @some_handler.handle(intent='some_intent_2', exit_flow=True)
    async def some_flow_handler_2(request, responder):
        pass

    assert len(some_handler.rules) == 2
    assert 'some_flow_handler_2' in some_handler.exit_flow_states


def test_dialogue_flow(kwik_e_mart_app):
    @kwik_e_mart_app.dialogue_flow(domain='some_domain', intent='some_intent')
    def some_handler(request, responder):
        pass

    assert some_handler.flow_state == 'some_handler_flow'
    assert 'some_handler' in some_handler.all_flows

    dm = some_handler.dialogue_manager
    assert_dialogue_state(dm, 'some_handler')
    assert_dialogue_state(dm, 'some_handler_flow')

    assert len(some_handler.rules) == 0

    @some_handler.handle(intent='some_intent')
    def some_flow_handler(request, responder):
        pass

    assert len(some_handler.rules) == 1

    @some_handler.handle(intent='some_intent_2', exit_flow=True)
    def some_flow_handler_2(request, responder):
        pass

    assert len(some_handler.rules) == 2
    assert 'some_flow_handler_2' in some_handler.exit_flow_states


def test_dialogue_flow_app_param(dm):

    def the_flow(request, responder):
        pass

    the_flow = DialogueFlow('the_flow', the_flow, dm=dm, intent='the_flow')

    flow_dm = the_flow.dialogue_manager
    assert_dialogue_state(flow_dm, 'the_flow')
    assert_dialogue_state(flow_dm, 'the_flow_flow')

    assert len(the_flow.rules) == 0

    app_received = None
    @the_flow.handle(intent='no_app')
    def _no_app(request, responder):
        responder.act('no_app')

    @the_flow.handle(intent='app_pos')
    def _app_pos(request, responder, app):
        nonlocal app_received
        app_received = app
        responder.act('app_pos')

    @the_flow.handle(intent='app_kw')
    def _app_kw(request, responder, *, app):
        nonlocal app_received
        app_received = app
        responder.act('app_kw')

    @the_flow.handle(intent='var_kw')
    def _var_kw(request, responder, **kwargs):
        nonlocal app_received
        app_received = app
        responder.act('var_kw')

    states = ['no_app', 'app_pos', 'app_kw', 'var_kw']

    for state in states:
        request = create_request('domain', state)
        responder = create_responder(request)
        app = MagicMock(name=f'{state}_mock_app')

        result = dm.apply_handler(
            request, responder, app=app, target_dialogue_state=the_flow.flow_state
        )

        if state is 'no_app':
            assert app_received is None
        else:
            assert app_received is not None

        assert len(result.directives) == 1
        assert result.directives[0]['name'] is state
        # reset app received
        app_received = None
