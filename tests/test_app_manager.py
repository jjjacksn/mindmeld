#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_dialogue
----------------------------------

Tests for app_manager module.

These tests apply only when async/await are supported.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import pytest

from mindmeld.app_manager import ApplicationManager, freeze_params
from mindmeld.components.request import Params, FrozenParams


@pytest.fixture
def app_manager(kwik_e_mart_app_path, kwik_e_mart_nlp):
    return ApplicationManager(kwik_e_mart_app_path, nlp=kwik_e_mart_nlp)


def test_freeze_params():
    params = freeze_params({'target_dialogue_state': 'some-state'})
    assert params.__class__ == FrozenParams

    input_params = Params()
    input_params.target_dialogue_state = 'some-state-2'
    params = freeze_params(input_params)
    assert params.__class__ == FrozenParams

    params = freeze_params(params)
    assert params.__class__ == FrozenParams

    with pytest.raises(TypeError):
        freeze_params([1, 2, 3])


def test_parse(app_manager):
    response = app_manager.parse('hello')

    fields = {'params', 'request', 'dialogue_state', 'directives', 'history'}
    for field in fields:
        assert field in vars(response).keys()


def test_app_storage(app_manager):
    assert len(app_manager) == 0  # should be empty to start
    assert app_manager  # truthy even when empty
    assert 'foo' not in app_manager
    app_manager['foo'] = 'bar'  # setting a value
    assert 'foo' in app_manager
    assert app_manager['foo'] is 'bar'  # value is present
    assert len(app_manager) == 1
    del app_manager['foo']
    assert 'foo' not in app_manager
    with pytest.raises(KeyError):
        app_manager['foo']
