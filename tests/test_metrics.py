from src.evaluation import CharacterErrorRate, WordErrorRate


def test_word_error_rate():
    wer = WordErrorRate()
    result = wer(["the quick brown fox"], ["the kwik brown fawx"])
    assert round(result.score, 2) == 0.5


def test_character_error_rate():
    cer = CharacterErrorRate()
    result = cer(["abc"], ["axc"])
    assert result.distance == 1
    assert round(result.score, 2) == 0.33
