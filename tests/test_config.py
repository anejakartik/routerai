from app.config import load_policy


def test_policy_loads_and_picks_cheapest():
    policy = load_policy()
    assert policy.baseline in policy.models
    # cheapest of the 'premium' tier must be the lowest-cost candidate
    cheapest = policy.cheapest("premium")
    candidates = policy.candidates("premium")
    assert cheapest == min(candidates, key=policy.cost_per_1k)


def test_unknown_tier_falls_back_to_default():
    policy = load_policy()
    assert policy.candidates("does-not-exist") == policy.candidates(policy.default_tier)
