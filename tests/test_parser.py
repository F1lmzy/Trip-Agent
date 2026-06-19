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


def test_parse_small_budget_recognized_as_low():
    parsed = parse_user_request("Plan a 2-day trip to Tokyo on a small budget.")

    assert parsed.budget == "low"


def test_parse_tight_budget_recognized_as_low():
    parsed = parse_user_request("Plan a 2-day trip to Tokyo with a tight budget.")

    assert parsed.budget == "low"


def test_parse_shoestring_budget_recognized_as_low():
    parsed = parse_user_request("Plan a shoestring 2-day trip to Tokyo.")

    assert parsed.budget == "low"


def test_missing_city_has_no_city():
    parsed = parse_user_request("Plan me a trip")

    assert parsed.city is None
    assert parsed.duration_days == 2


def test_parse_unknown_city_dynamically_from_trip_phrase():
    parsed = parse_user_request("Plan a 2-day trip to Kyoto with temples and food.")

    assert parsed.city == "Kyoto"
    assert parsed.duration_days == 2
    assert set(parsed.interests) >= {"culture", "food"}


def test_parse_lowercase_mumbai_city():
    parsed = parse_user_request("I want to visit mumbai for 2 days with food and culture.")

    assert parsed.city == "Mumbai"
    assert parsed.duration_days == 2
    assert set(parsed.interests) >= {"food", "culture"}


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


def test_parse_flight_intent_with_origin():
    parsed = parse_user_request("Plan a 2-day trip to Tokyo flying from London. Medium budget.")

    assert parsed.city == "Tokyo"
    assert parsed.asks_for_flights is True
    assert parsed.origin_city == "London"


def test_parse_flight_intent_without_origin():
    parsed = parse_user_request("Plan a 2-day Tokyo trip with flights.")

    assert parsed.city == "Tokyo"
    assert parsed.asks_for_flights is True
    assert parsed.origin_city is None
    assert parsed.trip_type == "round_trip"


def test_parse_one_way_trip_type():
    parsed = parse_user_request("Plan a one-way flight from London to Milan.")

    assert parsed.trip_type == "one_way"
    assert parsed.asks_for_flights is True


def test_parse_one_way_hyphenated_and_compound():
    for msg in ["one-way trip to Berlin", "oneway ticket to Paris", "outbound only to Rome"]:
        parsed = parse_user_request(msg)
        assert parsed.trip_type == "one_way", msg


def test_parse_round_trip_default_when_unspecified():
    parsed = parse_user_request("Plan a 2-day Tokyo trip with flights.")
    assert parsed.trip_type == "round_trip"


def test_parse_round_trip_explicit():
    for msg in ["round trip to Paris", "round-trip from London", "return ticket to Rome"]:
        parsed = parse_user_request(msg)
        assert parsed.trip_type == "round_trip", msg


def test_parse_explicit_date_range_full_months():
    parsed = parse_user_request("Plan a trip from London to Milan from June 21 to June 25 with flights")
    assert parsed.departure_date == "2026-06-21"
    assert parsed.return_date == "2026-06-25"


def test_parse_explicit_date_range_with_ordinal_suffixes():
    # Regression: 'June 25th' / '22nd' ordinal suffixes used to break the
    # word-boundary regex, so the range was dropped and flights fell back to
    # today + duration.
    parsed = parse_user_request("Plan a trip from London to Milan from June 22 to June 25th with flights")
    assert parsed.departure_date == "2026-06-22"
    assert parsed.return_date == "2026-06-25"


def test_parse_explicit_date_range_both_ordinals():
    parsed = parse_user_request("June 22nd to June 25th")
    assert parsed.departure_date == "2026-06-22"
    assert parsed.return_date == "2026-06-25"


def test_parse_explicit_date_range_abbreviated_months():
    parsed = parse_user_request("Plan a trip Jun 22 to Jun 25")
    assert parsed.departure_date == "2026-06-22"
    assert parsed.return_date == "2026-06-25"


def test_parse_explicit_date_range_return_never_before_departure():
    # Year-boundary: Dec 28 -> Jan 5 must roll return into the next year so
    # return >= departure.
    parsed = parse_user_request("Dec 28 to Jan 5")
    assert parsed.departure_date == "2026-12-28"
    assert parsed.return_date == "2027-01-05"
    assert parsed.return_date >= parsed.departure_date


def test_parse_explicit_date_range_without_from_keyword():
    parsed = parse_user_request("Plan a Paris trip June 21 to June 25")
    assert parsed.departure_date == "2026-06-21"
    assert parsed.return_date == "2026-06-25"


def test_parse_explicit_date_range_bare_day_shares_month():
    parsed = parse_user_request("Plan a trip June 21 - 25 with hotels")
    assert parsed.departure_date == "2026-06-21"
    assert parsed.return_date == "2026-06-25"


def test_parse_explicit_date_range_hyphen_compact():
    parsed = parse_user_request("Plan a trip June 21-25")
    assert parsed.departure_date == "2026-06-21"
    assert parsed.return_date == "2026-06-25"


def test_parse_date_range_different_months():
    parsed = parse_user_request("Plan a Tokyo trip from July 3 to July 10")
    assert parsed.departure_date == "2026-07-03"
    assert parsed.return_date == "2026-07-10"


def test_parse_no_explicit_dates_when_none_mentioned():
    parsed = parse_user_request("Plan a 2-day Tokyo trip with flights")
    assert parsed.departure_date is None
    assert parsed.return_date is None


def test_parse_no_flight_intent_when_word_absent():
    parsed = parse_user_request("Plan a 2-day trip to Tokyo with food.")

    assert parsed.asks_for_flights is False
    assert parsed.origin_city is None


def test_parse_origin_city_from_departing_phrase():
    parsed = parse_user_request("Plan a Paris trip departing from Mumbai.")

    assert parsed.origin_city == "Mumbai"


def test_parse_from_origin_to_destination_picks_destination_not_origin():
    # Regression: "from london to milan" must yield city=Milan (destination),
    # not London (the origin). Previously the known-cities lookup returned
    # London first because it is iterated before Milan is even considered.
    parsed = parse_user_request(
        "plan a 2 day trip from london to milan with flights and hotels and keep the budget decent"
    )

    assert parsed.city == "Milan"
    assert parsed.origin_city == "London"
    assert parsed.asks_for_flights is True
    assert parsed.asks_for_hotel is True


def test_parse_from_origin_to_unknown_destination_still_captured():
    # Milan is not in KNOWN_CITIES, but the "from X to Y" route phrase must
    # still capture it as the destination rather than falling back to the
    # origin (London, which is known).
    parsed = parse_user_request("Plan a trip from Berlin to Plovdiv with flights.")

    assert parsed.city == "Plovdiv"
    assert parsed.origin_city == "Berlin"


def test_parse_origin_city_not_swallowed_to_keyword():
    # Regression: the origin capture used to grab "London To" (swallowing the
    # "to" of the route) producing a garbled origin city.
    parsed = parse_user_request("Plan a 2-day trip from London to Milan.")

    assert parsed.origin_city == "London"
    assert parsed.origin_city != "London To"


def test_parse_fly_to_destination_from_origin_reversed_order():
    # "fly to <dest> from <origin>" — destination still wins.
    parsed = parse_user_request("fly to london from paris")

    assert parsed.city == "London"
    assert parsed.origin_city == "Paris"


def test_parse_to_plan_not_mistaken_for_destination():
    # "to plan" must not be read as a destination city when there is no real
    # "to <city>" phrase. This guards the to-match known-city restriction.
    parsed = parse_user_request("I want to plan a trip")

    assert parsed.city is None


def test_parse_to_destination_from_origin_reversed_order():
    # Regression: "to kyoto from singapore" (destination BEFORE origin) used
    # to pick Singapore as the city because Kyoto is not in KNOWN_CITIES and
    # only the "from X to Y" order was handled.
    parsed = parse_user_request("Plan a 2-day trip to kyoto from singapore")
    assert parsed.city == "Kyoto"
    assert parsed.origin_city == "Singapore"


def test_parse_fly_to_destination_from_origin_reversed_order():
    parsed = parse_user_request("fly to milan from london")
    assert parsed.city == "Milan"
    assert parsed.origin_city == "London"


def test_parse_reversed_order_with_full_interests_message():
    parsed = parse_user_request(
        "Plan a 2-day trip to kyoto from singapore. I like anime, food, and photography. Medium budget."
    )
    assert parsed.city == "Kyoto"
    assert parsed.origin_city == "Singapore"
    assert parsed.duration_days == 2
    assert "anime" in parsed.interests
    assert "food" in parsed.interests
    assert "photography" in parsed.interests
