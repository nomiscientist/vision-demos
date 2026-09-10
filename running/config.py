"""Params for the running pose run.

Edit values here, then run `python main.py`. There are no command-line flags on
purpose: `run.json` in each output directory snapshots these values, so a result
can always be traced back to its settings.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
INPUT_VIDEO = DATA_DIR / "input" / "running.mov"

# ── Conversion (.mov -> .mp4) ────────────────────────────────────────────────
# Always converted: ffmpeg applies the source rotation, OpenCV ignores it.
INFERENCE_HEIGHT = 1080   # uploaded to the model; ViTPose caps at a 2048px long edge
EXPORT_HEIGHT = 1080      # rendered, and sets the panel's height with it
TRIM_SECONDS = None       # e.g. 3.0 for a fast test run
CONVERT_CRF = 23
FORCE_RECONVERT = False   # True re-runs ffmpeg even when a cached MP4 matches
REUSE_POSES = True      # True reuses cached poses instead of calling the gateway

# ── Model ────────────────────────────────────────────────────────────────────
MODEL = "usyd-community/vitpose-plus-large"
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0   # seconds; video pose is minutes, not seconds

# Detect on every frame, decimating nothing. A stride at 170 steps a minute
# lasts about 21 frames, so a foot strike is a two- or three-frame event: skip
# frames and the cadence is measured against a grid coarser than the thing being
# measured. Billing is per sampled frame, so this costs wall clock, not money.
EVERY_FRAME = True
VIDEO_FPS = 2.0          # detector cadence, used only when EVERY_FRAME is False
VIDEO_MAX_FRAMES = None  # None -> the model's 900-frame default
PRECISION = 4            # decimal places on normalized coordinates (1-8)

# ── Render ───────────────────────────────────────────────────────────────────
DRAW_FACE = False             # face keypoints read as a scribble over the face
MAX_PERSONS = 1               # one body should be one subject
PERSON_IOU_THRESHOLD = 0.6    # above this, two boxes are the same person
DRAW_BBOX = False             # the skeleton is enough on a single runner
BBOX_COLOR = (255, 160, 40)   # BGR
DRAW_TRACK_LABEL = False      # the "id 0" tag above the box
DRAW_HUD = False              # the frame/person counter in the top-left
LINE_THICKNESS = 3            # px at a 720px-wide frame; scales with the export
POINT_RADIUS = 4
OUTPUT_CRF = 20

# The two joints every number on the panel is measured from, enlarged in the
# panel's own per-foot colors so the trace can be checked against the leg it
# came from. The flash is a ring that expands and fades over each foot strike,
# marking the same instant the panel's dots do.
HIGHLIGHT_FEET = True
FOOT_MARKER_RADIUS = 9        # px at a 720px-wide frame
FLASH_ON_CONTACT = True
FLASH_SECONDS = 0.25
FLASH_THICKNESS = 3

# The comet trail: each ankle's recent path, drawn behind the marker in that
# foot's own color and fading out over the last TRAIL_SECONDS. The marker says
# where the ankle is; the trail says where it has just been, so the swing reads
# as a swing rather than as a dot that moved between frames. It is drawn under
# the skeleton, so it never covers a joint.
#
# The span is what to tune first. Around a third of a stride shows the swing
# arc while it is still an arc; a trail as long as the whole stride overlaps the
# next one and turns into a loop, which is a picture of the cycle rather than of
# the motion. At 180 spm a stride is 0.67s, so 0.32 is a little under half of it.
TRAIL_ON_ANKLES = True
TRAIL_SECONDS = 0.32
TRAIL_FEET = ("left", "right")
TRAIL_WIDTH = 9            # px at a 720px-wide frame, at the head
TRAIL_TAPER = 0.18         # tail width as a fraction of the head's
TRAIL_OPACITY = 0.85       # alpha at the head; the tail fades to nothing
# How the fade is distributed along the tail. 1.0 is linear; above it the head
# keeps its brightness and the tail drops away fast, which is the shooting-star
# look. Below 1.0 the whole tail stays bright and reads as a ribbon.
TRAIL_FADE = 1.35
TRAIL_GLOW = 2.6           # halo width, as a multiple of the core's
TRAIL_GLOW_OPACITY = 0.28
TRAIL_SOFTNESS = 0.6       # edge blur, as a fraction of the head width

# The wind. A comet's tail does not lie back along its path — it streams
# downwind of it — and the trail reads as motion rather than as a smear
# precisely because it does. Each point of the tail is pushed downwind in
# proportion to its age, so the head stays exactly on the ankle and only the
# spent part of the trail drifts. It also fixes what the path alone looked like
# during stance, where the ankle barely moves and the trail collapsed into a
# blob at the marker: raked, a planted foot streams a clean streak.
#
# Downwind is the direction of travel, reversed, and it is measured per clip
# rather than set per video: the planted foot is the part of a runner that is
# not moving with them, so relative to the hips it travels backwards through
# every stance — on a road or on a belt, where there is no displacement to read
# a direction from at all. See src/gait._heading; `heading_x` and `facing_x` in
# gait.json record what it found, and `quality` flags a clip where the motion
# and the runner's own facing disagree.
#
# "auto" follows that heading. "left"/"right" force a side (for a clip the
# detector gets wrong), and "none" switches the rake off, which puts the tail
# back exactly along the path the ankle took.
TRAIL_WIND = "auto"
TRAIL_WIND_SPEED = 220     # px/s at a 720px-wide frame; the drift, per second of age

# ── The gait signal ──────────────────────────────────────────────────────────
# One signal per leg: ankle height above that ankle's own stance level, divided
# by the runner's own leg length.
#
# The ankle, because COCO-17 has no foot. The gateway's ViTPose returns 17
# keypoints and the ankles are the lowest two — no heel, no toes — so nothing
# here can see the foot itself. See cadence-explained.md for what that costs.
ANALYZE_GAIT = True

# What the zero is. "ground" takes a fixed stance level from the whole clip,
# which is right for a static camera; "hip" measures each ankle against that
# frame's own hips instead, for a camera that moves with the runner.
FOOT_REFERENCE = "ground"

# The belt line, as a percentile of ankle height rather than its maximum: the
# lowest single frame of a clip is a bad pose as often as it is the ground.
GROUND_PERCENTILE = 90

# One belt line per foot, because the two ankles do not project to the same
# height while standing on the same belt. Measured on this clip: the right leg
# is the near one (it projects 3-6% larger) and the left ankle's stance level
# sits 0.9% of a leg length higher in frame. Share one line between them and the
# lower foot spends longer under the strike threshold for no reason to do with
# the runner. False shares one line, which is only right when both ankles are
# the same distance from the lens.
PER_FOOT_GROUND = True

# Light smoothing only. A swing peak is about ten frames wide, so a wide window
# would flatten the very feature the strike is timed against.
SMOOTH_MEDIAN_FRAMES = 3   # median filter, to kill isolated spikes
SMOOTH_MEAN_FRAMES = 3     # then a centered moving average, to kill jitter

# ── Foot strikes ─────────────────────────────────────────────────────────────
# Two fractions of the typical swing height. The foot is planted below
# CONTACT_FRACTION, and a rise has to clear SWING_PEAK_FRACTION to count as a
# swing at all — which is what stops an ankle wobbling during stance from
# reporting three strikes where there was one.
SWING_PEAK_FRACTION = 0.55
CONTACT_FRACTION = 0.18
# Shorter than this and the signal only clipped the threshold mid-swing. In
# SECONDS, not frames: as a frame count it silently meant 100 ms at 30 fps and
# 50 ms at 60 fps, and the safe range for it is set by the runner's stance
# duration (~240 ms here), which is a time. It also has the one real
# interaction in this config -- a lower CONTACT_FRACTION shortens stances, so
# this floor has to stay well under the shortest one the threshold creates.
MIN_CONTACT_SECONDS = 0.10

# How heavily the live cadence reading is smoothed: the weight given to the
# newest stride in an exponential average, so the effective window is about
# 2/alpha - 1 steps. 0.20 is roughly nine steps.
#
# Exponential rather than a trailing window of k samples, because a window
# lurches twice per reading -- once when a value enters and again when it
# leaves -- and the second lurch is pure artifact. And measured over
# *overlapping strides* rather than steps, so the left/right alternation
# cancels in every sample instead of needing an even window to average it out.
# See src/gait._stride_ema_samples.
CADENCE_SMOOTHING = 0.20

# The trailing window for each foot's own rate, in that foot's steps. Used only
# by gait.json and clearance.png -- per-foot rate is not on the panel, since the
# feet alternate and neither can be faster than the other.
CADENCE_FOOT_WINDOW_STEPS = 2

# ── Side panel ───────────────────────────────────────────────────────────────
# The clip on the left, an equal-width live panel on the right.
SIDE_PANEL = True

# The narrowest the cadence axis may get, in spm. A steady runner varies by
# about a spm, so an axis fitted to the data alone spans ~15 spm and turns every
# remaining wobble into a 20px lurch. See src/panel.GaitPanel.
PANEL_Y_MIN_SPAN = 20.0

# The graph draws one cadence series twice: over a trailing window ("now") and
# as the mean so far. There is no per-foot trace, and that is deliberate — the
# feet alternate, so each lands the same number of times and neither can have a
# higher rate than the other. See src/gait.GaitAnalysis.phase_split.
# Interpolated points per step-to-step span. The vertices are ~12px apart on
# this export, so a handful is plenty; the curve passes through every vertex
# either way, so this buys smoothness and never moves a reading.
PANEL_TRACE_SEGMENTS = 6

# The graph draws one series: the mean cadence so far. The live reading is the
# headline's own figure, and drawing it here as well needed a colour key to tell
# the two apart — a key for a distinction the title can simply make. There is no
# per-foot trace either, and that is deliberate: the feet alternate, so each
# lands the same number of times and neither can have a higher rate than the
# other. See src/gait.GaitAnalysis.phase_split.
PANEL_TRACE_THICKNESS = 3.0
PANEL_SIDE_MARGIN = 60    # px at a 720px-wide panel; the inset either side of
                          # the cadence graph, the knee block and the ankle view

# The cadence strip's width : height. It was a square, on the argument that a
# slope on screen should be a slope in the data -- and a wide box does compress
# slope, which is the cost this pays for the height. It is worth paying because
# of what the trace is: a running mean, which settles within a few strides and
# then drifts about a spm. Its job is to show that settling, not to be measured
# by eye; the live figure is the headline and every reading is in gait.json.
# PANEL_Y_MIN_SPAN still fixes the y range, so a flat-looking trace cannot be a
# rescaled wobble. 1.0 gives the old square back.
PANEL_GRAPH_ASPECT = 3.2

# Text sizes are px at a 720px-wide panel and scale with the output.
#
# The headline used to be 140px of digits and took the top third of the panel
# on its own. It is the one number that never needs help being found -- it is
# the only thing at the top, it is the only thing that large, and it changes
# while you watch it -- so the size was buying nothing that the position was
# not already giving it, and it was spending the height the ankle view now
# lives in. At 78 it still reads across a room.
PANEL_TITLE_SIZE = 26
PANEL_NUMBER_SIZE = 78
PANEL_UNIT_SIZE = 28
PANEL_SUB_SIZE = 22
PANEL_GRAPH_TITLE_SIZE = 26
PANEL_AXIS_LABEL_SIZE = 19
PANEL_TICK_SIZE = 15
PANEL_KNEE_KEY_SIZE = 22
PANEL_KNEE_VALUE_SIZE = 30

# Panel labels. The headline is the live cadence over its trailing average; the
# sub-line carries the step count. The graph y-axis spells the unit out once
# ("steps per minute (spm)"); the headline keeps the short form.
PANEL_TITLE = "CADENCE"
PANEL_UNIT = "spm"
PANEL_STEPS_LABEL = "{steps} steps"
PANEL_STEPS_LABEL_ONE = "{steps} step"    # the first second of any clip

PANEL_GRAPH_TITLE = "AVG CADENCE OVER TIME"
PANEL_Y_LABEL = "steps per minute (spm)"
PANEL_X_LABEL = "time (s)"

# ── The average knee ─────────────────────────────────────────────────────────
# Under the cadence graph, one leg either side. This is what the contact events
# buy: they give a phase to average at, so every strike stacks on the same
# instant of the cycle. The limb is drawn at its measured orientation (neither
# leg mirrored -- the point is to compare them) and the fan around the shank is
# one standard deviation of the knee angle across those strikes.
#
# Knee flexion is the angle at the knee between thigh and shank, 0 for a
# straight leg. Every frame of it lands in gait.json; see src/gait._knee_flexion.
#
# The shape is sampled at the knee's own most-extended point near each strike,
# searched this far either side, and not at the strike frame. A minimum landing
# on the edge of this window means the real one is probably outside it, so edge
# hits are rejected and counted into gait.json's `quality` block. The knee flexes at
# 150-420 deg/s just after contact, so sampling on that slope costs 5-14 deg per
# frame of timing error; its turning point is flat, so the same error costs
# almost nothing. Same reasoning as timing the steps on mid-stance.
KNEE_SEARCH_SECONDS = 0.10

# A planted knee does not bend past about this. Frames inside a detected ground
# contact that exceed it are counted into gait.json's `quality` block -- they
# are the pose failing, not the runner, and on this clip they are where ViTPose
# swaps the left and right leg as the legs scissor past each other. For scale,
# this runner's planted knee peaks at 36-44 deg and never passes 51 outside
# that one swapped frame, which reaches 90.
KNEE_STANCE_LIMIT_DEG = 85.0
# Which legs the knee block draws. The near leg alone by default, and the reason
# is what the panel would otherwise be claiming: two limbs side by side read as
# "here is your asymmetry", and the gap between them is not interpretable from a
# single side-on camera -- the two legs are viewed from slightly different
# angles and every left-versus-right figure inherits that. One limb makes no
# claim it cannot support.
#
# The near leg is also the cleaner one, measuring 20-58% less keypoint jitter
# and more repeatable stride to stride. That is the tiebreak, not the reason:
# both legs' segment proportions check out, so neither is geometrically wrong.
#
# ("left", "right") shows both. Either way both legs are measured in full and
# land in gait.json, summary.txt and knee.png.
PANEL_KNEE_FEET = ("right",)

PANEL_KNEE_SHARE = 0.40        # share of the spare height the knee block takes
PANEL_KNEE_FILL = 0.92         # limb length as a fraction of that block's height
PANEL_KNEE_THICKNESS = 6       # px at a 720px-wide panel
PANEL_KNEE_FAN_SD = 1.0        # how many standard deviations the shading spans
PANEL_KNEE_FAN_OPACITY = 0.30        # the cap, where every sampled limb overlaps
PANEL_KNEE_FAN_STEP_OPACITY = 0.05   # what one sampled limb contributes
PANEL_KNEE_ENVELOPE_STEPS = 11       # limbs laid down across that spread
#
# The two opacities above are the knobs for how heavy the shading looks. The cap
# is what the densest part of the band reaches; the step is what a single
# sampled limb adds, so the edges of the band sit near it. Lower both together
# to fade the whole thing: 0.55/0.09 is a distinct halo, 0.30/0.05 reads as a
# soft band, 0.18/0.03 is barely there.
#
# The band is zero-width at the hip and widest at the ankle, and that is
# geometry rather than a setting. Every strike is measured relative to its own
# hip, so the hip is the origin and cannot vary; spread grows with distance from
# it, and the knee-angle spread adds on past the knee. Measured as the distance
# between the two +/-1 sd limbs at each joint, this clip gives 0.000 leg lengths
# at the hip, 0.021 at the knee and 0.074 at the ankle.

# How fast the drawn shape and its number chase the current average, per frame.
# The average only changes when a strike lands, so undamped it is a static
# figure that jumps once a stride; 0.05 gives a continuous drift with a time
# constant near half a second. Causal -- it only ever reads its own last output.
PANEL_KNEE_EASING = 0.05

# How many recent strikes the drawn shape averages. A mean over every strike
# converges after a few strides and then sits still -- 1.0 degrees of movement
# across this clip, which reads as frozen rather than steady. Eight strikes
# (four strides) drifts 1.7 degrees and is still an average, just over a stated
# span. 0 or None averages everything. gait.json and summary.txt always report
# every strike.
PANEL_KNEE_WINDOW = 8
PANEL_KNEE_DECIMALS = 1        # a tenth of a degree, so the figure visibly moves

# How long the printed numbers hold between updates. The drawn shape keeps
# easing every frame regardless -- this is only the digits. At a tenth of a
# degree the eased value changes every frame, and a number flickering thirty
# times a second is not a reading; a second's hold gives about fifteen updates
# across a fifteen-second clip, which is legible and still moving.
PANEL_KNEE_LABEL_HOLD_SECONDS = 1.0

PANEL_KNEE_TITLE = "AVG KNEE SHAPE AT FOOT STRIKE"
PANEL_KNEE_LEFT = "LEFT"
PANEL_KNEE_RIGHT = "RIGHT"
PANEL_KNEE_FORMAT = "{deg}° ± {sd}°"
PANEL_KNEE_FORMAT_NO_SD = "{deg}°"   # one strike has no spread to quote

# ── The ankles, live ─────────────────────────────────────────────────────────
# The bottom block: the two ankle points lifted out of the pose and drawn on
# their own, in the space they actually move through. Same two joints every
# number on the panel is measured from, same two colors, with the body taken
# away -- so the loop each one traces, the distance between them and the belt
# line under them are all there to be read directly.
#
# Three layers, and they are deliberately not the same kind of thing:
#
#   * the dim cloud is recent history -- every earlier pass, painted on and
#     then left to fade, dense where the ankle lingers (stance, the turn at
#     the top of the swing) and bright wherever it has passed lately. A
#     picture of where the time has been going, without every stride since
#     frame one stacked into one picture.
#   * the comet is the live path, blown downwind, exactly as on the video.
#   * the dot is the measurement: this frame's ankle, nothing added.
#
# Both the cloud and the comet only reach back a bounded span of real time --
# see PANEL_ANKLE_MEMORY_HALFLIFE_SECONDS and PANEL_ANKLE_TRAIL_SECONDS -- and
# neither is ever drawn ahead of the current frame: the shape of the loop is
# the answer, and drawing it from frame one would be showing the future.
PANEL_ANKLE_TITLE = "ANKLE PATH VISUALIZATION"
PANEL_ANKLE_FEET = ("left", "right")

# What the positions are measured against -- the same question FOOT_REFERENCE
# answers for the signal, so "auto" follows it. "image" draws the ankle where
# it is in the shot, which is right for a fixed camera and is the only mode
# where the belt line is a fixed line (so it is the only one that draws it).
# "hip" measures against the mid-hip, for a camera that moves with the runner.
PANEL_ANKLE_FRAME = "auto"
PANEL_ANKLE_PAD = 0.10        # air around the path, as a share of its own span
# How closely the *trail* (comet + cloud) tracks the exact measured path, vs a
# lightly eased copy of it -- 1.0 is the exact path, lower values round off a
# genuine sharp reversal (toe-off can turn the ankle ~130 degrees in a frame
# or two) that no denser curve fit can smooth away without moving off the
# sample near it. The dot is never eased -- it is always the exact reading.
PANEL_ANKLE_PATH_EASE = 0.45
# How long the live comet takes to settle from the exact path onto the eased
# one, after a point stops being the head. Not instant: a point that jumped
# straight from exact to eased the moment the head moved past it disagreed
# with itself by tens of px on this clip, every single frame, which is what
# "jumpy" was -- a point already on screen relocating on its own, not the
# ankle moving. Settling it over a few frames instead keeps the head glued to
# the dot (age 0 is always exact) while spreading that same total correction
# out into a change too small, and too continuous, to read as a jump.
PANEL_ANKLE_PATH_SETTLE_SECONDS = 0.30
# The block is as tall as the path needs at the column's width, since an
# equal-scale fit cannot stretch to fill and any extra height is a dead band
# under the drawing. This caps that, as a share of the height the knee block
# and this one share: a clip with a tall path -- a runner crossing the frame,
# a high-knee drill -- must not crowd the knee out.
PANEL_ANKLE_MAX_SHARE = 0.55
PANEL_ANKLE_GROUND = True
PANEL_ANKLE_HIP_LABEL = "under the hip"   # hip mode only: a vertical at x = 0
PANEL_ANKLE_DOT = 7               # px at a 720px-wide panel

# Longer than the overlay's trail, because this view has room for it: at 180
# spm a stride is 0.67s, so 0.55 shows almost the whole loop with the head
# still visibly moving around it.
PANEL_ANKLE_TRAIL_SECONDS = 0.55
PANEL_ANKLE_TRAIL_WIDTH = 7
PANEL_ANKLE_TRAIL_TAPER = 0.20
PANEL_ANKLE_TRAIL_OPACITY = 0.95
PANEL_ANKLE_TRAIL_FADE = 1.35
PANEL_ANKLE_GLOW = 2.6
PANEL_ANKLE_GLOW_OPACITY = 0.30
PANEL_ANKLE_SOFTNESS = 0.6
# Sub-points per frame-to-frame span (see comet.Comet). This trail is wide and
# lives long enough to be looked at closely, unlike the overlay's -- so the
# turn at the top of the swing, where the ankle changes direction fastest,
# used to draw as a visible elbow. 1 turns it back off.
PANEL_ANKLE_TRAIL_SMOOTH = 5
# Gentler than the overlay's wind: the view is a few hundred px across, so the
# same drift would blow the tail clean out of the box. Direction comes from
# TRAIL_WIND, so both halves of the frame stream the same way.
PANEL_ANKLE_WIND_SPEED = 90    # px/s at a 720px-wide panel

# The cloud: everywhere the ankle has recently been, not everywhere it has
# ever been. It fades continuously rather than accumulating without limit --
# see GaitPanel._visited_cloud -- so old strides stop competing with the loop
# being run right now. HALFLIFE is the real time it takes an untouched spot to
# lose half its brightness; short enough that only the last two or three
# strides read as clearly present. OPACITY is the alpha a freshly stroked
# pixel is drawn at, full stop -- coverage is combined by *maximum*, not
# summed, so a pixel stroked twice inside the same frame (which the join
# between one frame's short stroke and the next always is, since a thick line
# covers a disk at both its ends) reads no brighter than one stroked once. 0
# switches the cloud off entirely.
PANEL_ANKLE_MEMORY_OPACITY = 0.35
PANEL_ANKLE_MEMORY_HALFLIFE_SECONDS = 0.8
PANEL_ANKLE_MEMORY_WIDTH = 5   # px at a 720px-wide panel

# One gap between every block on the panel, so the rhythm down it is even by
# construction rather than tuned per gap. Raising it spreads the blocks apart
# and the two drawings shrink to pay for it; the top and bottom air below stay
# exactly where they are set.
PANEL_SECTION_GAP = 48

# Air at the top and bottom of the panel, in px at a 720px-wide reference. These
# are reserved *before* the two plots are sized, so raising one shrinks the
# drawings rather than pushing anything off the edge. The bottom margin is the
# gap between the last label and the credit line's own band; the credit line is
# placed from the panel edge and does not move with any of this.
#
# As set, the content runs from 51px below the top edge to 978 of 1080, with
# 83px clear before the credit. The bottom used to read as dead space for a
# duller reason than the margin: the knee label was reserved here *and* inside
# the knee box, so 34px of it was booked twice and nothing could ever sit there.
PANEL_TOP_MARGIN = 60
PANEL_BOTTOM_MARGIN = 34

PANEL_FONT = "auto"        # "auto", "opencv" to force Hershey, or a font path
PANEL_FONT_INDEX = None    # face index inside a .ttc; None uses the default

ATTRIBUTION = "Jeremy Park"   # credit line, bottom-right of the panel; "" disables it
ATTRIBUTION_SIZE = 22
ATTRIBUTION_MARGIN = 22
ATTRIBUTION_OPACITY = 0.9     # 0-1; white, dialled back so it reads as a credit

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"     # converted MP4s + cached poses, reused across runs
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"

SAVE_CLEARANCE_PLOT = True   # clearance.png: both feet, every strike, the cadence
SAVE_STEPS_PLOT = True       # steps.png: step and contact time, per foot
SAVE_KNEE_PLOT = True        # knee.png: every detected knee, for debugging
SAVE_JOINT_PLOT = False      # one joint's height over time, one line per track
PLOT_JOINT = "left_ankle"    # any name from src.skeleton.KPT_NAMES
