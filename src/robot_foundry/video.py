"""Decode by presentation timestamp; never assume concatenated v3 videos start at zero."""

import av
import numpy as np


def decode_frames(path, timestamps, size, fps):
    times = np.asarray(timestamps, dtype=np.float64)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Requested video timestamps must be finite and strictly increasing.")
    frames = []
    tolerance = 0.51 / fps + 1e-4
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if times[0] > 1:
            container.seek(int(max(0, times[0] - 0.5) / stream.time_base), stream=stream, backward=True)
        previous = None
        previous_time = None
        cursor = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video frame has no PTS.")
            time = float(frame.pts * frame.time_base)
            while cursor < len(times) and time >= times[cursor]:
                candidate, candidate_time = frame, time
                if previous is not None and abs(previous_time - times[cursor]) < abs(time - times[cursor]):
                    candidate, candidate_time = previous, previous_time
                if abs(candidate_time - times[cursor]) > tolerance:
                    raise ValueError(
                        f"Video/data mismatch: requested {times[cursor]:.4f}, found {candidate_time:.4f}"
                    )
                pixels = candidate.reformat(width=size, height=size, format="rgb24").to_ndarray()
                frames.append(pixels.transpose(2, 0, 1).copy())
                cursor += 1
            previous, previous_time = frame, time
            if cursor == len(times):
                break
        # Last requested timestamp can be slightly larger than the final PTS because of float32 storage.
        while (
            cursor < len(times) and previous is not None and abs(previous_time - times[cursor]) <= tolerance
        ):
            frames.append(
                previous.reformat(width=size, height=size, format="rgb24")
                .to_ndarray()
                .transpose(2, 0, 1)
                .copy()
            )
            cursor += 1
    if len(frames) != len(times):
        raise ValueError(f"Video is truncated: decoded {len(frames)} of {len(times)} requested frames.")
    return np.stack(frames)
