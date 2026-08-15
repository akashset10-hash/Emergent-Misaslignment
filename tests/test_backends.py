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


def test_hf_local_init_and_free_structure():
    """Static guard for a class of real-model bugs we can't exercise without a
    GPU: __init__ must cache decoder layers; free() must NOT rebuild them (it
    deletes self.model). Catches accidental misplacement of __init__ body into
    free()."""
    import ast, pathlib
    src = pathlib.Path("src/em/backends/hf_local.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "HFLocalBackend")
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    assert "__init__" in methods and "free" in methods

    def calls(fn, name):
        return any(isinstance(n, ast.Attribute) and n.attr == name
                   for n in ast.walk(fn))
    # __init__ builds the layer cache; free() must not (model is gone by then).
    assert calls(methods["__init__"], "_decoder_layers")
    assert not calls(methods["free"], "_decoder_layers")
