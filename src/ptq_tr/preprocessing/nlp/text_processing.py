"""Text preprocessing helpers."""


def get_text_pair(batch, sentence1_key, sentence2_key=None):
    if sentence2_key is None:
        return batch[sentence1_key]
    return batch[sentence1_key], batch[sentence2_key]


def is_valid_text_pair(example, sentence1_key, sentence2_key=None):
    if example[sentence1_key] is None:
        return False
    if sentence2_key is not None and example[sentence2_key] is None:
        return False
    return True


def extract_answer_from_span(context, offsets, start_index, end_index):
    if start_index > end_index:
        return ""

    start_offset = offsets[start_index]
    end_offset = offsets[end_index]
    if start_offset is None or end_offset is None:
        return ""

    start_char = start_offset[0]
    end_char = end_offset[1]
    if start_char is None or end_char is None or end_char < start_char:
        return ""

    return context[start_char:end_char]
