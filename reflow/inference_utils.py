import math
from ast import literal_eval


def validate_infer_step(infer_step):
    if isinstance(infer_step, bool) or not isinstance(infer_step, int):
        raise TypeError("infer_step must be an integer")
    if infer_step <= 0:
        raise ValueError("infer_step must be positive")
    return infer_step


def parse_speaker_mix(value):
    """Safely parse and validate a speaker-weight mapping."""
    if value is None or value == 'None':
        return None
    if isinstance(value, dict):
        parsed = value
    else:
        text = str(value).strip().replace('，', ',').replace('：', ':')
        if not text.startswith('{'):
            text = '{' + text + '}'
        parsed = literal_eval(text)
    if not isinstance(parsed, dict):
        raise ValueError("speaker mix must be a dictionary")

    result = {}
    for speaker_id, weight in parsed.items():
        speaker_id = int(speaker_id)
        weight = float(weight)
        if speaker_id <= 0:
            raise ValueError("speaker IDs must be positive")
        if not math.isfinite(weight):
            raise ValueError("speaker weights must be finite")
        result[speaker_id] = weight
    return result
