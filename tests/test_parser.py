from app.agent.parser import parse_user_request


def test_parse_full_tokyo_request():
    parsed = parse_user_request(
        "Plan a 2-day trip to Tokyo. I like food, anime, and photography. My budget is moderate."
    )

    assert parsed.city == "Tokyo"
    assert parsed.duration_days == 2
    assert parsed.budget == "medium"
    assert set(parsed.interests) >= {"food", "anime", "photography"}
    assert parsed.asks_for_hotel is False
    assert parsed.asks_for_current_info is False


def test_parse_singapore_budget_synonyms():
    parsed = parse_user_request(
        "Plan a cheap 2 day trip to Singapore with nature, museums, and vegetarian food."
    )

    assert parsed.city == "Singapore"
    assert parsed.duration_days == 2
    assert parsed.budget == "low"
    assert set(parsed.interests) >= {"nature", "museums", "food"}
    assert "vegetarian" in parsed.dietary_needs


def test_missing_city_has_no_city():
    parsed = parse_user_request("Plan me a trip")

    assert parsed.city is None
    assert parsed.duration_days == 2


def test_parse_hotel_intent():
    parsed = parse_user_request("Plan a trip to Paris and recommend a hotel or place to stay.")

    assert parsed.city == "Paris"
    assert parsed.asks_for_hotel is True


def test_parse_current_info_intent():
    parsed = parse_user_request("Plan a Tokyo trip with current events and recent food recommendations.")

    assert parsed.city == "Tokyo"
    assert parsed.asks_for_current_info is True


def test_parse_follow_up_cheaper():
    parsed = parse_user_request("Make it cheaper.")

    assert parsed.city is None
    assert parsed.is_follow_up is True
    assert parsed.follow_up_intent == "cheaper"
