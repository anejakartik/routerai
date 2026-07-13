import app.providers as providers
from app.providers import (
    AnthropicProvider,
    Completion,
    Message,
    OpenAIProvider,
    StubProvider,
    _RealProvider,
    get_provider,
)


def test_get_provider_defaults_to_stub(monkeypatch):
    monkeypatch.delenv("ROUTERAI_PROVIDER", raising=False)
    assert get_provider().name == "stub"


def test_get_provider_selects_real_adapters(monkeypatch):
    monkeypatch.setenv("ROUTERAI_PROVIDER", "openai")
    assert isinstance(get_provider(), OpenAIProvider)
    monkeypatch.setenv("ROUTERAI_PROVIDER", "anthropic")
    assert isinstance(get_provider(), AnthropicProvider)


def test_unknown_provider_falls_back_to_stub(monkeypatch):
    monkeypatch.setenv("ROUTERAI_PROVIDER", "does-not-exist")
    assert get_provider().name == "stub"


def test_stub_completion_tagged_with_provider():
    c = StubProvider().complete("stub-small", [Message("user", "hi")])
    assert c.provider == "stub"
    assert c.total_tokens > 0


class _AlwaysFails(_RealProvider):
    name = "boom"

    def __init__(self):
        super().__init__(retries=1)
        self.calls = 0

    def _call(self, model, messages):
        self.calls += 1
        raise RuntimeError("simulated provider outage")


def test_real_provider_retries_once_then_falls_back_to_stub():
    p = _AlwaysFails()
    out = p.complete("gpt-4o-mini", [Message("user", "hello")])
    # 1 initial attempt + 1 retry = 2 calls
    assert p.calls == 2
    # graceful degrade: we still get a completion, tagged as stub
    assert isinstance(out, Completion)
    assert out.provider == "stub"
    assert out.total_tokens > 0


class _FailsThenSucceeds(_RealProvider):
    name = "flaky"

    def __init__(self):
        super().__init__(retries=1)
        self.calls = 0

    def _call(self, model, messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient")
        return Completion(text="ok", prompt_tokens=1, completion_tokens=1, provider=self.name)


def test_real_provider_retry_succeeds_without_fallback():
    p = _FailsThenSucceeds()
    out = p.complete("m", [Message("user", "x")])
    assert p.calls == 2
    assert out.provider == "flaky"
    assert out.text == "ok"


def test_openai_adapter_maps_response(monkeypatch):
    # stub out the lazily-imported `openai` module
    import sys
    import types

    class _Msg:
        content = "mapped answer"

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 11
        completion_tokens = 7

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    class _Client:
        def __init__(self, *a, **k):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kw: _Resp())
            )

    fake = types.ModuleType("openai")
    fake.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", fake)

    out = OpenAIProvider()._call("gpt-4o-mini", [Message("user", "hi")])
    assert out.text == "mapped answer"
    assert out.prompt_tokens == 11
    assert out.completion_tokens == 7
    assert out.provider == "openai"


def test_anthropic_adapter_maps_response(monkeypatch):
    import sys
    import types

    class _Block:
        text = "claude answer"

    class _Usage:
        input_tokens = 5
        output_tokens = 3

    class _Resp:
        content = [_Block()]
        usage = _Usage()

    class _Client:
        def __init__(self, *a, **k):
            self.messages = types.SimpleNamespace(create=lambda **kw: _Resp())

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    out = AnthropicProvider()._call("claude-3-5-haiku", [Message("user", "hi")])
    assert out.text == "claude answer"
    assert out.prompt_tokens == 5
    assert out.completion_tokens == 3
    assert out.provider == "anthropic"
