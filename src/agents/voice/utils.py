import re
from collections.abc import Callable


def get_sentence_based_splitter(
    min_sentence_length: int = 20,
) -> Callable[[str], tuple[str, str]]:
    """Returns a function that splits text into chunks based on sentence boundaries.

    Args:
        min_sentence_length: The minimum length of a sentence to be included in a chunk.

    Returns:
        A function that splits text into chunks based on sentence boundaries.
    """

    def sentence_based_text_splitter(text_buffer: str) -> tuple[str, str]:
        """
        A function to split the text into chunks. This is useful if you want to split the text into
        chunks before sending it to the TTS model rather than waiting for the whole text to be
        processed.

        Args:
            text_buffer: The text to split.

        Returns:
            A tuple of the text to process and the remaining text buffer.
        """
        stripped_text = text_buffer.strip()
        # CJK sentences end with a full-width terminator and are not followed by whitespace, so
        # they need their own boundary that does not require a separator to match.
        boundaries = list(re.finditer(r"(?<=[.!?])\s+|(?<=[。！？])\s*", stripped_text))
        if boundaries:
            last_boundary = boundaries[-1]
            # Keep one space between the flushed sentences, as the previous single-space join of
            # the split sentences did, while leaving CJK text without an added separator.
            combined_sentences = re.sub(
                r"(?<=[.!?])\s+", " ", stripped_text[: last_boundary.start()]
            )
            if len(combined_sentences) >= min_sentence_length:
                remaining_text_buffer = stripped_text[last_boundary.end() :]
                if remaining_text_buffer:
                    # Carry any trailing whitespace over to the remainder. It separates the text
                    # held back from the next streamed delta, and stripping it concatenates the
                    # next word onto the last one, so "He " followed by "arrived" is spoken as
                    # "Hearrived". With nothing held back it would only leave a whitespace-only
                    # buffer, which is still synthesized at the end of the turn.
                    remaining_text_buffer += text_buffer[len(text_buffer.rstrip()) :]
                return combined_sentences, remaining_text_buffer
        return "", text_buffer

    return sentence_based_text_splitter
