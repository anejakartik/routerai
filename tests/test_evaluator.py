from app.evaluator import evaluate


def test_empty_response_scores_zero():
    assert evaluate("what is 2+2", "") == 0.0


def test_relevant_answer_scores_higher_than_irrelevant():
    prompt = "explain distributed consensus algorithms in detail"
    good = ("Distributed consensus algorithms let multiple nodes agree on a "
            "single value despite failures; explain examples include Paxos and "
            "Raft which handle detailed leader election and log replication.")
    bad = "purple monkey dishwasher banana"
    assert evaluate(prompt, good) > evaluate(prompt, bad)


def test_refusal_is_penalized():
    prompt = "write a summary"
    refusal = "I cannot help with that as an AI."
    assert evaluate(prompt, refusal) < 0.5


def test_score_is_bounded():
    s = evaluate("test", "a much longer response " * 50)
    assert 0.0 <= s <= 1.0
