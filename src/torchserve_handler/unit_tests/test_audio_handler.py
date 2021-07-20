# pylint: disable=W0621
# Using the same name as global function is part of pytest
"""
Basic unit test for AudioHandler class.
Ensures it can load and execute an example model
"""

import sys
import pytest
from torchserve_handler.audio_handler import AudioHandler
from .test_utils.mock_context import MockContext

sys.path.append("ts/torch_handler/unit_tests/models/tmp")


@pytest.fixture()
def model_setup():
    context = MockContext(model_name="ammod-26")
    with open(
        "./src/torchserve_handler/unit_tests/assets/test_sample.wav", "rb"
    ) as fin:
        image_bytes = fin.read()
    return (context, image_bytes)


def test_initialize(model_setup):
    model_context, _ = model_setup
    handler = AudioHandler()
    handler.initialize(model_context)
    # test if config is loaded propably ans segment_step is not default value
    assert handler.segment_duration == 2
    assert True
    return handler


def test_crash_on_batch_request(model_setup):
    context, image_bytes = model_setup
    handler = test_initialize(model_setup)
    test_data = [{"data": image_bytes}, {"data": image_bytes}]
    with pytest.raises(Exception) as e:
        assert handler.handle(test_data, context)
    assert str(e.value) == "Audiohandler is not capable of handling batch requests"


def test_handle(model_setup):
    context, image_bytes = model_setup
    handler = test_initialize(model_setup)
    test_data = [{"data": image_bytes}]
    results = handler.handle(test_data, context)
    assert 1 == len(results)
    assert "classIds" in results[0]
    assert "channels" in results[0]
    assert 4 == len(results[0]["channels"])


# def test_handle_explain(model_setup):
#     context, image_bytes = model_setup
#     context.explain = True
#     handler = test_initialize(model_setup)
#     test_data = [{"data": image_bytes, "target": 0}] * 2
#     results = handler.handle(test_data, context)
#     assert len(results) == 2
#     assert results[0]
