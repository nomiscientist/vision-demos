"""Params for the chin-ups pose run.

Edit values here, then run `python main.py`.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
INPUT_VIDEO = DATA_DIR / "input" / "chin_ups.mov"

# ── Conversion (.mov -> .mp4) ────────────────────────────────────────────────
# Always converted: ffmpeg applies the source rotation, OpenCV ignores it.
INFERENCE_HEIGHT = 1080   # uploaded to the model; ViTPose caps at a 2048px long edge
EXPORT_HEIGHT = 1080      # rendered; None keeps the source resolution
TRIM_SECONDS = None       # e.g. 3.0 for a fast test run
CONVERT_CRF = 23
FORCE_RECONVERT = False   # True re-runs ffmpeg even when a cached MP4 matches
REUSE_POSES = False       # True reuses cached poses instead of calling the gateway

# ── Model ────────────────────────────────────────────────────────────────────
MODEL = "usyd-community/vitpose-plus-large"
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0   # seconds; video pose is minutes, not seconds

EVERY_FRAME = True       # detect on every frame; costs wall clock, not money
VIDEO_FPS = 2.0          # detector cadence, used only when EVERY_FRAME is False
VIDEO_MAX_FRAMES = None  # None -> the model's 900-frame default
PRECISION = 4            # decimal places on normalized coordinates (1-8)

# ── Render ───────────────────────────────────────────────────────────────────
DRAW_FACE = False             # face keypoints read as a scribble over the face
MAX_PERSONS = 1               # one body should be one subject
PERSON_IOU_THRESHOLD = 0.6    # above this, two boxes are the same person
DRAW_BBOX = True
BBOX_COLOR = (255, 160, 40)   # BGR, blue
DRAW_TRACK_LABEL = False      # the "id 0" tag above the box
DRAW_HUD = False              # the frame/person counter in the top-left
LINE_THICKNESS = 3            # px at a 720px-wide frame; scales with the export
POINT_RADIUS = 4
OUTPUT_CRF = 20

# ── Rep counting ─────────────────────────────────────────────────────────────
# The signal is `hand_y - torso_y`; see rep-counting-explained.md for the method.
COUNT_REPS = True

TORSO_JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
HAND_JOINTS = ("left_wrist", "right_wrist")
HEAD_JOINTS = ("left_eye", "right_eye")

REFERENCE_TO_HANDS = True         # False measures against the frame instead
HAND_OFF_BAR_TOLERANCE = 0.05     # wrist drift allowed before "hands off the bar"
SMOOTH_MEDIAN_FRAMES = 5          # median filter, to kill isolated spikes
SMOOTH_MEAN_FRAMES = 7            # then a centered moving average, to kill jitter
BASELINE_PERCENTILE = 10          # relative zero: the hanging plateau, not the minimum
REP_PEAK_FRACTION = 0.60          # rise past this share of rep height to confirm a rep
REP_RESET_FRACTION = 0.30         # fall back under this to arm the next one
REQUIRE_HEAD_ABOVE_HANDS = True   # gate each rep on chin-over-bar

# ── The reported number ──────────────────────────────────────────────────────
# Which span the panel and displacement.png both report.
#   "moving": onset to peak, excluding any dead hang at the bottom
#   "ascent": valley to peak, hang included
#   "total":  valley to valley, the whole cycle
REP_METRIC = "moving"

REP_ONSET_METHOD = "velocity"        # or "threshold"; see rep-counting-explained.md
REP_ONSET_VELOCITY_FRACTION = 0.10   # lower = starts earlier, nearer the base
REP_ONSET_FRACTION = 0.05            # used only by REP_ONSET_METHOD = "threshold"

# ── Side panel ───────────────────────────────────────────────────────────────
# The clip on the left, an equal-width stats panel on the right.
SIDE_PANEL = True
PANEL_PROGRESSIVE = True   # False draws every rep from frame 0 instead
PANEL_PLOT_SCALE = 0.95    # square graph edge, as a fraction of the panel's short side
PANEL_Y_TICKS = 4          # target y-axis tick count on the duration graph

# Text sizes and gaps are px at a 720px-wide panel and scale with the output.
PANEL_HEADER_SIZE = 40
PANEL_NOTE_SIZE = 30           # the closing note
PANEL_HEADER_LINE_GAP = 0.85   # line spacing, as a multiple of the font size
PANEL_NOTE_LINE_GAP = 0.85

REP_LABEL = "Chin-ups"
PANEL_GRAPH_TITLE = "Rep Duration (s)"
PANEL_AVG_LABEL = "Avg Duration"
PANEL_Y_LABEL = "Duration (s)"

PANEL_FONT = "auto"        # "auto", "opencv" to force Hershey, or a font path
PANEL_FONT_INDEX = None    # face index inside a .ttc; None uses the default

ATTRIBUTION = "Jeremy Park"   # credit line, bottom-right of the export; "" disables it
ATTRIBUTION_SIZE = 20
ATTRIBUTION_MARGIN = 25
ATTRIBUTION_OPACITY = 0.75    # 0-1; white, dialled back so it reads as a credit

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"     # converted MP4s + cached poses, reused across runs
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"

SAVE_JOINT_PLOT = False      # one joint's height over time, one line per track
PLOT_JOINT = "left_wrist"    # any name from src.skeleton.KPT_NAMES
