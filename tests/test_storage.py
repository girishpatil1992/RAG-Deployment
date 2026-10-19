import tempfile
import unittest
from pathlib import Path

from storage import SQLiteStorage


class SQLiteStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "nested" / "app.sqlite3"
        self.storage = SQLiteStorage(self.database_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_messages_and_sources_persist_with_thread(self) -> None:
        self.storage.create_thread("thread-1")
        self.storage.add_message("thread-1", "user", "  What is this?  ")
        self.storage.add_message(
            "thread-1",
            "assistant",
            "A test document.",
            sources=[{"source": "guide.txt", "score": 0.9, "text": "A test document."}],
        )

        reopened_storage = SQLiteStorage(self.database_path)
        messages = reopened_storage.get_messages("thread-1")
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(
            reopened_storage.list_threads()[0]["title"], "What is this?"
        )
        self.assertEqual(messages[1]["sources"][0]["source"], "guide.txt")

    def test_deleting_thread_removes_its_messages(self) -> None:
        self.storage.create_thread("thread-1")
        self.storage.add_message("thread-1", "user", "Hello")
        self.storage.delete_thread("thread-1")

        self.assertEqual(self.storage.list_threads(), [])
        self.assertEqual(self.storage.get_messages("thread-1"), [])

    def test_chunks_and_embeddings_are_cached(self) -> None:
        self.storage.cache_chunks("document-1", ["chunk one", "chunk two"])
        self.storage.cache_embeddings("model-a", {"chunk one": [0.1, 0.2]})

        self.assertEqual(
            self.storage.get_cached_chunks("document-1"), ["chunk one", "chunk two"]
        )
        self.assertEqual(
            self.storage.get_cached_embeddings("model-a", ["chunk one", "chunk two"]),
            {"chunk one": [0.1, 0.2]},
        )
        self.assertEqual(self.storage.get_cached_embeddings("model-b", ["chunk one"]), {})


if __name__ == "__main__":
    unittest.main()
