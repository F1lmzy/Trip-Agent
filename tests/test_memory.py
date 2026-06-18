from app.memory.short_term import ShortTermMemory


def test_short_term_memory_stores_history_per_user():
    memory = ShortTermMemory()

    memory.add_message("kavin", "user", "Plan Tokyo")
    memory.add_message("kavin", "assistant", "Here is the plan")
    memory.add_message("alex", "user", "Plan Paris")

    kavin_history = memory.get_history("kavin")
    alex_history = memory.get_history("alex")

    assert [message.content for message in kavin_history] == ["Plan Tokyo", "Here is the plan"]
    assert [message.content for message in alex_history] == ["Plan Paris"]


def test_short_term_memory_respects_max_messages():
    memory = ShortTermMemory(max_messages_per_user=2)

    memory.add_message("kavin", "user", "one")
    memory.add_message("kavin", "assistant", "two")
    memory.add_message("kavin", "user", "three")

    assert [message.content for message in memory.get_history("kavin")] == ["two", "three"]


def test_short_term_memory_can_clear_user_history():
    memory = ShortTermMemory()
    memory.add_message("kavin", "user", "Plan Tokyo")

    memory.clear("kavin")

    assert memory.get_history("kavin") == []
    assert memory.has_history("kavin") is False
