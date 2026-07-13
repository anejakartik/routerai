import importlib.util
import os

_GATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "eval_gate.py",
)
_spec = importlib.util.spec_from_file_location("eval_gate", _GATE_PATH)
eval_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_gate)


def test_run_eval_produces_summary():
    summary = eval_gate.run_eval()
    assert summary["total_requests"] == len(eval_gate.EVAL_PROMPTS)
    assert summary["quality_bar_pass_rate"] is not None
    assert summary["savings_pct_vs_premium"] >= 0


def test_gate_passes_with_lenient_thresholds(capsys):
    rc = eval_gate.main(["--min-pass-rate", "0.0", "--min-savings-pct", "0"])
    assert rc == 0
    assert "GATE PASSED" in capsys.readouterr().out


def test_gate_fails_on_impossible_savings_threshold(capsys):
    # savings can never exceed 100%; this must trip the gate.
    rc = eval_gate.main(["--min-pass-rate", "0.0", "--min-savings-pct", "999"])
    assert rc == 1
    assert "GATE FAILED" in capsys.readouterr().err


def test_gate_fails_on_impossible_pass_rate(capsys):
    rc = eval_gate.main(["--min-pass-rate", "1.01", "--min-savings-pct", "0"])
    assert rc == 1


def test_gate_json_output(capsys):
    rc = eval_gate.main(["--min-pass-rate", "0.0", "--min-savings-pct", "0", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"gate": "PASS"' in out
    assert '"summary"' in out
