from __future__ import annotations

import pytest

from agents.voice import get_sentence_based_splitter


def _stream(text: str, chunk_size: int, min_sentence_length: int = 20) -> list[str]:
    """Feed text through the splitter the way VoiceStreamedResult._add_text does."""
    split = get_sentence_based_splitter(min_sentence_length)
    buffer = ""
    spoken: list[str] = []
    for index in range(0, len(text), chunk_size):
        buffer += text[index : index + chunk_size]
        chunk, buffer = split(buffer)
        if chunk:
            spoken.append(chunk)
    if buffer.strip():
        spoken.append(buffer.strip())
    return spoken


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 11, 15])
@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith went to Washington. He arrived at 3 p.m. sharp.",
        "Hello there friend. How are you doing today? I am fine. Goodbye now.",
        "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten.",
        "A short one. Then a much longer sentence that exceeds the minimum easily.",
    ],
    ids=["abbreviations", "questions", "many_short", "mixed_lengths"],
)
def test_streamed_text_is_spoken_without_losing_or_gluing_words(text: str, chunk_size: int) -> None:
    """Splitting must not depend on where the model's deltas happen to break.

    The buffer was stripped before splitting, which also removed the trailing space that
    separates the held-back text from the next delta. When a delta boundary landed just
    after a space, the following word was concatenated onto the previous one, so "He "
    plus "arrived" was spoken as "Hearrived".
    """
    assert " ".join(_stream(text, chunk_size)).split() == text.split()


@pytest.mark.parametrize("chunk_size", [1, 3, 7])
@pytest.mark.parametrize(
    "text",
    [
        "这是一个足够长的中文句子，可以切分开了。第二句话也足够长了，应该单独朗读。第三句",
        "今日はとてもいい天気ですね、散歩しましょう。もう一つの文もここに",
        "This is an English sentence。中文句子在这里，也应该被切分开来。And more English",
    ],
    ids=["chinese", "japanese", "mixed"],
)
def test_streamed_cjk_text_is_split_at_sentence_boundaries(text: str, chunk_size: int) -> None:
    """CJK sentences end with `。！？` and are not followed by whitespace.

    The whitespace requirement in the split pattern therefore never matched, so a CJK reply
    was never split and the TTS call waited for the whole reply.
    """
    spoken = _stream(text, chunk_size)

    assert len(spoken) > 1
    assert [char for char in "".join(spoken) if not char.isspace()] == [
        char for char in text if not char.isspace()
    ]


def test_cjk_split_respects_min_sentence_length() -> None:
    """The minimum length applies to CJK text too, which has no word separators to rely on."""
    split = get_sentence_based_splitter(20)

    assert split("短句。下一句。") == ("", "短句。下一句。")

    spoken, remaining = split("这是一个足够长的中文句子，可以切分开了。\n下一句还没有结束")

    assert spoken == "这是一个足够长的中文句子，可以切分开了。"
    assert remaining == "下一句还没有结束"


def test_cjk_split_leaves_no_whitespace_only_remainder() -> None:
    """A terminated CJK sentence flushes everything, so the buffer must end up empty.

    A whitespace-only buffer is still truthy, so the turn-end path would synthesize it as an
    extra TTS call.
    """
    split = get_sentence_based_splitter(20)

    assert split("这是一个足够长的中文句子，可以切分开了。 ") == (
        "这是一个足够长的中文句子，可以切分开了。",
        "",
    )


def test_split_preserves_the_separator_before_the_next_delta() -> None:
    """The remainder must still end with the whitespace it was given."""
    split = get_sentence_based_splitter(20)

    spoken, remaining = split("This sentence is long enough to flush. He ")

    assert spoken == "This sentence is long enough to flush."
    # Without the trailing space, appending the next delta glues the words together.
    assert remaining == "He "
    assert (remaining + "arrived").split() == ["He", "arrived"]


def test_split_leaves_the_buffer_untouched_when_nothing_is_flushed() -> None:
    split = get_sentence_based_splitter(20)

    assert split("Too short. ") == ("", "Too short. ")
    assert split("   ") == ("", "   ")
    assert split("") == ("", "")


def test_split_without_trailing_whitespace_is_unchanged() -> None:
    split = get_sentence_based_splitter(20)

    assert split("This sentence is long enough to flush. He") == (
        "This sentence is long enough to flush.",
        "He",
    )
