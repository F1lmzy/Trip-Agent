from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.memory.vector_store import VectorStore
from tests.fakes import FakeEmbedder


def make_long_term_memory(tmp_path) -> LongTermMemory:
    return LongTermMemory(VectorStore(path=str(tmp_path), embedder=FakeEmbedder()))


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


def test_long_term_memory_adds_and_gets_user_preferences(tmp_path):
    memory = make_long_term_memory(tmp_path)

    memory.add_preference("kavin", "I like museums")

    assert memory.get_preferences("kavin") == ["I like museums"]


def test_long_term_memory_is_scoped_by_user(tmp_path):
    memory = make_long_term_memory(tmp_path)

    memory.add_preference("kavin", "I like museums")
    memory.add_preference("alex", "I like nightlife")

    assert memory.get_preferences("kavin") == ["I like museums"]
    assert memory.get_preferences("alex") == ["I like nightlife"]


def test_long_term_memory_searches_relevant_preferences(tmp_path):
    memory = make_long_term_memory(tmp_path)
    memory.add_preference("kavin", "I prefer vegetarian food")

    results = memory.search_preferences("kavin", "Plan a food trip", limit=1)

    assert results == ["I prefer vegetarian food"]


def test_long_term_memory_clears_only_one_user(tmp_path):
    memory = make_long_term_memory(tmp_path)
    memory.add_preference("kavin", "I like museums")
    memory.add_preference("alex", "I like nightlife")

    memory.clear_preferences("kavin")

    assert memory.get_preferences("kavin") == []
    assert memory.get_preferences("alex") == ["I like nightlife"]
