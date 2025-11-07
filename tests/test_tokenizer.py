from pathlib import Path

from src.data import TextTokenizer


def test_tokenizer_round_trip(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        '{"token_to_id": {"<blank>": 0, "<unk>": 1, "a": 2, "b": 3, " ": 4}}',
        encoding="utf-8",
    )
    tokenizer = TextTokenizer.from_file(vocab_path)

    encoded = tokenizer.encode("ab a")
    assert encoded == [2, 3, 4, 2]
    decoded = tokenizer.decode(encoded)
    assert decoded == "ab a"

