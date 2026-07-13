from em.backends import make_backend, MockBackend
from em.backends.base import ModelBackend, Capability
from em.config import Config


def test_mock_backend_full_protocol():
    b = MockBackend()
    assert isinstance(b, ModelBackend)
    assert b.capabilities() == {Capability.GENERATE, Capability.TOKEN_LOGPROBS,
                                Capability.ACTIVATIONS, Capability.STEERING,
                                Capability.PERPLEXITY}
    assert b.generate("hi").text
    assert b.perplexity("some text") > 0
    acts = b.activations("hi there", [4, 8])
    assert set(acts) == {4, 8}


def test_factory_returns_mock():
    cfg = Config()
    cfg.backend.kind = "mock"
    assert isinstance(make_backend(cfg), MockBackend)


def test_require_blocks_missing_capability_for_hosted_style_backend():
    # A hosted-style backend that only generates must be rejected by any
    # instrument needing activations/steering — mechanistic work needs local/GPU.
    from em.backends.base import require, CapabilityError
    import pytest

    class GenOnly:
        name = "hosted"
        def capabilities(self):
            return {Capability.GENERATE}
    with pytest.raises(CapabilityError):
        require(GenOnly(), Capability.ACTIVATIONS)
    with pytest.raises(CapabilityError):
        require(GenOnly(), Capability.STEERING)
