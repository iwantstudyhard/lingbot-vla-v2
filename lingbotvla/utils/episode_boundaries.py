"""Pure timestamp arithmetic for CFR LeRobot v3 shared-video segments."""

import math


def bounded_timestamps(relative_times, start, end, fps):
    """Return in-segment frame-grid timestamps and clamping flags.

    End is exclusive. The audited dataset is constant-frame-rate and its
    segment boundaries are on the global frame grid. Fail on unsupported
    metadata instead of silently selecting an adjacent episode.
    """
    if not all(math.isfinite(float(x)) for x in (start, end, fps)) or fps <= 0:
        raise ValueError("Invalid video interval/fps")
    start_frame, end_frame = round(start * fps), round(end * fps)
    if abs(start * fps - start_frame) > 0.01 or abs(end * fps - end_frame) > 0.01:
        raise ValueError("Clean training requires CFR video boundaries aligned to the frame grid")
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("Empty or invalid episode video segment")
    result, clamped = [], []
    for relative in relative_times:
        if not math.isfinite(float(relative)):
            raise ValueError("Non-finite video query")
        requested = round((start + float(relative)) * fps)
        index = min(max(requested, start_frame), end_frame - 1)
        result.append(index / fps)
        clamped.append(index != requested)
    return result, clamped
