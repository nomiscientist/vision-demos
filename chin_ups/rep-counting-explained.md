# How a rep is found and timed

1. Run pose estimation on every frame to get the body's keypoints.
2. Measure the torso's up and down movement against the hands on the bar.
3. Count a rep at each peak, but only if the eyes clear the hands.
4. Time the pull on each one, from where the rise starts to the peak.

Code: [`src/reps.py`](src/reps.py). Every parameter, and why each default is
what it is: [`config.py`](config.py). Per-rep numbers land in `reps.json` and
`summary.txt`.

## The signal

- `hand_y - torso_y`, measured against the hands rather than the frame, so a
  handheld camera does not register as movement.
- Torso is the mean of both shoulders and both hips; averaging four joints
  cancels the noise the model makes solving each frame independently.
- Image `y` grows downward, so the signal *rises* as the body rises. Hanging is
  its low plateau, which is what calibrates zero.
- A median filter then a centered moving average: spikes, then jitter, without
  shifting the peak times.
- Frames where the wrists leave the bar are dropped, because the whole signal
  assumes they are a fixed point.

## The count

- A hysteresis state machine: armed at a valley, confirmed at a peak, closed on
  the way back down. Two thresholds rather than one, or noise wobbling across a
  single crossing counts several times.
- The peak also has to clear chin-over-bar, meaning the eyes above the hands,
  which is self-calibrating in a way an amplitude threshold is not.
- `displacement.png` plots the curve and that margin, so you can see *why* each
  rep counted.

## The timing

`REP_METRIC` picks the span the panel and the plot both report:

- **`moving`** (the default): onset to peak, the pull minus any dead hang.
- **`ascent`**: valley to peak, hang included.
- **`total`**: valley to valley, the whole cycle.

The onset is the foot of the rise, found by walking back from the fastest part
of the pull until the curve flattens.

## Limits

- One subject: the signal reads the tracked person, so a second body in frame is
  ignored.
- Form is not judged beyond chin-over-bar: no elbow angle, no kipping check.
- Zero and rep height are calibrated per clip, so heights do not compare across
  videos.
