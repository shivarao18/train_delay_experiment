"""Generate the 10-slide ASEE 2027 proposal deck with speaker notes.

Builds slides/asee_proposal_slides.pptx (16:9) from the proposal narrative and
the real pipeline results/figures in reports/figures, and writes the full
speaker script to slides/presentation_script.md.

Usage:
    python slides/make_slides.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT_ROOT / "reports" / "figures"
OUT_PPTX = PROJECT_ROOT / "slides" / "asee_proposal_slides.pptx"
OUT_SCRIPT = PROJECT_ROOT / "slides" / "presentation_script.md"

NAVY = RGBColor(0x1F, 0x38, 0x64)
ORANGE = RGBColor(0xC5, 0x5A, 0x11)
GRAY = RGBColor(0x40, 0x40, 0x40)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ---------------------------------------------------------------------------
# Slide content: (title, bullets, image, notes)
# bullets: list of (text, indent_level, bold)
# ---------------------------------------------------------------------------

SLIDES = [
    {
        "kind": "title",
        "title": "Teaching Trustworthy AI Decision-Making in Civil Engineering\nthrough Train-Delay Forecasting",
        "subtitle": "A project-based learning module on prediction reliability\n"
                    "ASEE 2027 Research Proposal  |  University of Florida",
        "notes": (
            "Good afternoon, everyone. Today I am presenting a proposal for a project-based "
            "learning module that teaches undergraduate civil-engineering students something "
            "our curricula rarely cover: not how to build an AI model, but how to decide "
            "whether an AI prediction is reliable enough to act on. The vehicle for this is "
            "train-delay forecasting on a real national railway network. Everything I will "
            "show you today - every number and every figure - comes from a working technical "
            "pipeline we have already built and validated, so this proposal is grounded in a "
            "functioning prototype, not just an idea."
        ),
    },
    {
        "title": "Motivation: accuracy is not reliability",
        "bullets": [
            ("AI models are spreading through civil engineering: transportation, infrastructure management, structural monitoring", 0, False),
            ("Standard training focuses on average error (e.g., Mean Absolute Error)", 0, False),
            ("But a model with low average error can still fail badly:", 0, False),
            ("during disruptions and congestion", 1, False),
            ("under unfamiliar operating conditions", 1, False),
            ("when the world drifts away from its training data", 1, False),
            ("In safety-critical infrastructure, a precise-looking number without uncertainty is dangerous", 0, True),
            ("Educational gap: students learn to train models, not to judge when to trust them", 0, True),
        ],
        "notes": (
            "Why does this matter? AI models are already embedded in transportation systems, "
            "asset management, and structural monitoring. Our courses teach students to train "
            "these models and evaluate them with average-error metrics like MAE. The problem "
            "is that average error tells you almost nothing about the cases that matter most. "
            "A model can look excellent on average and still fail during exactly the "
            "disruptions and unusual conditions where engineers depend on it. In "
            "safety-critical infrastructure, a precise-looking prediction that carries no "
            "statement of its own uncertainty is genuinely dangerous. So the gap we are "
            "addressing is educational: students graduate knowing how to build predictive "
            "models, but not how to judge when a prediction deserves their trust."
        ),
    },
    {
        "title": "Educational objective and learning outcomes",
        "bullets": [
            ("Objective: develop the technical understanding and professional judgment to evaluate AI systems used in civil infrastructure", 0, True),
            ("After the module, students can:", 0, False),
            ("distinguish prediction accuracy from prediction reliability", 1, False),
            ("interpret prediction intervals, coverage, calibration, and sharpness", 1, False),
            ("read reliability diagrams (stated vs observed coverage)", 1, False),
            ("spot models that do well overall but fail in specific operating conditions", 1, False),
            ("explain how temporal drift erodes reliability", 1, False),
            ("recommend engineering responses when predictions are uncertain or miscalibrated", 1, False),
        ],
        "notes": (
            "The objective is professional judgment, not just technique. After completing the "
            "module, students should be able to distinguish accuracy from reliability, "
            "interpret prediction intervals and their coverage, read a reliability diagram "
            "that compares what a model claims against what it delivers, and - most "
            "importantly - recognize when a model that looks good on average is quietly "
            "failing in specific operating regimes. They should also be able to explain how "
            "temporal drift degrades a deployed model, and recommend appropriate engineering "
            "responses when a prediction is too uncertain or too poorly calibrated to act on. "
            "These outcomes map directly onto the four modules I will walk through next."
        ),
    },
    {
        "title": "Learning context: the RIDE railway dataset",
        "bullets": [
            ("RIDE (Elliker et al., 2026): open, nationwide dataset of the Belgian railway network", 0, False),
            ("94.5M train events, 2023-2025; benchmark-ready release on Hugging Face", 1, False),
            ("Task: for each active train, predict the delay change at its next 15 scheduled stops (~40 min ahead)", 0, False),
            ("Forward-in-time design is built in: train on 2023-2024, test on 2025", 0, True),
            ("Temporal drift is inherent to the task, not simulated", 1, False),
            ("Students get starter code + partially prepared data (interpretable features, physical units)", 0, False),
            ("Focus stays on reliability evaluation, not data cleaning", 1, False),
        ],
        "notes": (
            "The technical foundation is the RIDE dataset - an open, nationwide-scale record "
            "of the Belgian railway network with about 95 million train events from 2023 "
            "through 2025. The prediction task is concrete and intuitive: for every active "
            "train, predict how its delay will change at each of its next fifteen scheduled "
            "stops, roughly forty minutes into the future. What makes RIDE ideal for teaching "
            "reliability is its forward-in-time design: models are developed on 2023 and 2024 "
            "operations and tested on 2025. Temporal drift is not something we simulate - it "
            "is baked into the benchmark. Students receive starter code and partially "
            "prepared data with interpretable features in physical units - seconds of delay, "
            "trains on nearby track segments - so their effort goes into evaluating "
            "reliability, not wrangling data."
        ),
    },
    {
        "title": "Module 1 - Prediction accuracy (and its blind spots)",
        "bullets": [
            ("Scaffolded XGBoost workflow vs a naive baseline: 'the delay stays the same'", 0, False),
            ("Pilot results on 2025 test data (2.1M predictions):", 0, True),
            ("XGBoost MAE: 87.8 s   |   naive baseline: 98.0 s", 1, True),
            ("The model wins in every forecast-horizon bin", 1, False),
            ("But the breakdown tells a richer story:", 0, False),
            ("in the most common regime (delay barely changes), the naive baseline is better: 14 s vs 32 s", 1, True),
            ("during large disruptions (>5 min change), the model wins big: 621 s vs 731 s", 1, False),
            ("Lesson: a single average number hides where a model is actually strong or weak", 0, True),
        ],
        "notes": (
            "Module one starts where most courses end: train a model, measure MAE. Students "
            "use a scaffolded XGBoost workflow and compare it against the humblest possible "
            "baseline - assume the train's current delay simply persists. In our pilot run on "
            "over two million 2025 predictions, the model beats the baseline: 88 seconds of "
            "mean absolute error versus 98, and it wins at every forecast horizon. But then "
            "students dig into the breakdown, and the story gets interesting. In the most "
            "common situation, where the delay barely changes, the naive baseline is more "
            "than twice as accurate as the machine-learning model. The model earns its keep "
            "during large disruptions. That is the first lesson: one average number cannot "
            "tell you where a model is strong and where it is weak."
        ),
    },
    {
        "title": "Module 2 - Prediction uncertainty via conformal prediction",
        "bullets": [
            ("From single values to intervals with statistical guarantees (MAPIE)", 0, False),
            ("Two deliberately contrasting methods:", 0, False),
            ("split conformal: near-constant width, marginal coverage guarantee", 1, False),
            ("conformalized quantile regression (CQR): adaptive width", 1, False),
            ("Pilot: both hit their stated coverage on 2025 data", 0, True),
            ("90% nominal -> 90.6% (split) and 90.1% (CQR)", 1, False),
            ("but CQR is sharper: median width 312 s vs 363 s", 1, True),
            ("Students learn coverage, sharpness, over/underconfidence from a real reliability diagram", 0, False),
        ],
        "image": FIGURES / "reliability_diagram.png",
        "notes": (
            "Module two replaces single-value predictions with prediction intervals, using "
            "conformal prediction through the MAPIE library. We deliberately give students "
            "two contrasting methods. Split conformal wraps the point model with intervals of "
            "essentially constant width and guarantees average coverage. Conformalized "
            "quantile regression adapts its width case by case. On the 2025 test year, both "
            "methods deliver almost exactly the coverage they promise - the reliability "
            "diagram on the right sits right on the diagonal - about ninety point five "
            "percent for split conformal and ninety point one for CQR at the ninety percent "
            "level. The difference is sharpness: CQR achieves the same coverage with "
            "meaningfully narrower typical intervals. Students learn that two honest models "
            "can price their uncertainty very differently, and sharper honesty is worth more."
        ),
    },
    {
        "title": "Module 3 - Temporal drift and conditional reliability",
        "bullets": [
            ("Key question: does '90% coverage' hold within specific operating conditions?", 0, True),
            ("Split conformal holds 90% on average but collapses conditionally:", 0, False),
            ("99% coverage at 0-5 min horizons -> 71% at 45+ min", 1, True),
            ("only 71% for heavily delayed trains (>5 min)", 1, False),
            ("CQR stays between 86-93% by adapting width (186 s -> 537 s across horizons)", 0, False),
            ("Monthly drift across 2025 is visible but mild (88-93%)", 0, False),
        ],
        "image": FIGURES / "conditional_reliability.png",
        "image_wide": True,
        "notes": (
            "Module three asks the question that separates reliability from accuracy: when a "
            "method says ninety percent coverage, does that hold within specific operating "
            "conditions, or only on average? The answer, shown in these plots, is striking. "
            "Split conformal keeps its ninety percent promise on average - but that average "
            "is a blend of ninety-nine percent coverage for five-minute-ahead predictions and "
            "only seventy-one percent for predictions more than forty-five minutes out. For "
            "heavily delayed trains, coverage also drops to seventy-one percent - exactly the "
            "trains a dispatcher cares most about. CQR, because it widens its intervals where "
            "the task is harder, holds between eighty-six and ninety-three percent across "
            "all conditions. Students see with their own data that a marginal guarantee can "
            "hide conditional failure, and that drift across the 2025 test year is real but "
            "modest compared to these regime effects."
        ),
    },
    {
        "title": "Module 4 - Engineering decision-making with real cases",
        "bullets": [
            ("Scenarios mined automatically from the 2025 predictions - real trains, real failures:", 0, False),
            ("A. Accurate but uncertain: error 12 s, but the 90% interval spans 588 s (24,000 such cases)", 1, False),
            ("B. Narrow but wrong: interval of 150 s misses the actual outcome (15,000 cases)", 1, False),
            ("C. Fine on average, unreliable in disruption: MAE nearly triples (236 s vs 88 s), coverage drops to 85.7%", 1, False),
            ("D. Same train, 0 min vs 27 min ahead: interval grows from 101 s to 1,256 s", 1, False),
            ("Students must decide: act on the prediction, hedge, or escalate to human judgment?", 0, True),
        ],
        "notes": (
            "Module four turns the analysis into engineering judgment. Instead of inventing "
            "hypothetical vignettes, we mine them from the model's own 2025 predictions - "
            "these are real trains on real days. Scenario A: the point prediction was off by "
            "just twelve seconds, but the interval spanned nearly ten minutes - accurate, yet "
            "the model knew it was uncertain. There are twenty-four thousand cases like it. "
            "Scenario B is the dangerous opposite: a confident, narrow interval that simply "
            "missed. Scenario C shows the aggregate illusion - during congested disruptions, "
            "error nearly triples and coverage sags. Scenario D follows one train: the "
            "half-minute-ahead interval was a hundred seconds wide; the twenty-seven-minute "
            "one, over twelve hundred. For each case students must choose: act on the "
            "prediction, hedge, or escalate to a human. That is the professional skill we "
            "are targeting."
        ),
    },
    {
        "title": "Research questions and assessment plan",
        "bullets": [
            ("Research questions:", 0, True),
            ("RQ1  Does the module improve understanding of accuracy vs reliability?", 1, False),
            ("RQ2  Can students interpret intervals, coverage, calibration, sharpness, reliability diagrams?", 1, False),
            ("RQ3  Can they apply these concepts to realistic infrastructure decisions?", 1, False),
            ("RQ4  Which concepts are hardest (drift, conditional coverage, calibration, width)?", 1, False),
            ("RQ5  Do students see uncertainty-aware AI as relevant to their careers?", 1, False),
            ("Mixed-methods assessment: pre/post tests, rubric-scored assignments, scenario-based decision tasks, attitudinal surveys, optional transfer task (e.g., structural-health monitoring)", 0, False),
            ("IRB approval before any student data is collected", 0, False),
        ],
        "notes": (
            "The study itself asks five research questions: whether the module improves "
            "students' understanding of the accuracy-reliability distinction; whether they "
            "can correctly interpret intervals, coverage, calibration, and reliability "
            "diagrams; whether they can carry those concepts into realistic decision "
            "scenarios; which concepts prove hardest to learn - our candidates are temporal "
            "drift, conditional coverage, calibration, and interval width; and how the "
            "experience shapes their attitudes toward AI in practice. Assessment is "
            "mixed-methods: pre- and post-tests - for example, choosing between a model with "
            "lower error but poor coverage and one with slightly higher error but honest "
            "intervals - plus rubric-scored assignments, scenario-based decision tasks, "
            "attitude surveys, and an optional transfer task to another domain such as "
            "structural-health monitoring. IRB approval will be in place before any student "
            "data is collected."
        ),
    },
    {
        "title": "Expected contribution and current status",
        "bullets": [
            ("A reproducible educational module for trustworthy-AI evaluation in civil engineering", 0, True),
            ("emphasis on recognizing unreliable predictions and deciding under uncertainty", 1, False),
            ("evidence on which reliability concepts need the most instructional support", 1, False),
            ("Technical pipeline is already built and validated:", 0, True),
            ("open data (RIDE), open tooling (XGBoost, MAPIE), fully scripted, runs end to end", 1, False),
            ("all figures and numbers in this talk come from that pipeline", 1, False),
            ("Next steps: finalize student-facing materials, pilot in [course name], collect assessment data", 0, False),
            ("Thank you - questions welcome", 0, True),
        ],
        "notes": (
            "To close: the contribution is a reproducible module that teaches civil engineers "
            "to evaluate AI the way the profession will need - recognizing unreliable "
            "predictions and making defensible decisions under uncertainty - together with "
            "evidence about which of these concepts students find hardest, which should be "
            "useful to anyone integrating trustworthy-AI topics into an engineering "
            "curriculum. I want to emphasize that the technical foundation is not "
            "speculative: the full pipeline - open data, open tooling, scripted end to end - "
            "already runs, and every figure and number you saw today came out of it. What "
            "remains is the educational study itself: finalizing the student-facing "
            "materials, piloting the module, and collecting the assessment data. Thank you - "
            "I would be glad to take questions."
        ),
    },
]


# ---------------------------------------------------------------------------
# Deck construction
# ---------------------------------------------------------------------------

def add_textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    return tf


def add_title(slide, text, top=Inches(0.35)):
    tf = add_textbox(slide, Inches(0.55), top, SLIDE_W - Inches(1.1), Inches(1.0))
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = NAVY
    # underline bar
    bar = slide.shapes.add_shape(1, Inches(0.6), top + Inches(0.85), Inches(2.6), Pt(3.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ORANGE
    bar.line.fill.background()


def add_bullets(slide, bullets, left, top, width, height):
    tf = add_textbox(slide, left, top, width, height)
    for i, (text, level, bold) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if level == 0 else "– ") + text
        p.level = level
        p.font.size = Pt(17 if level == 0 else 15)
        p.font.bold = bold
        p.font.color.rgb = GRAY if not bold else NAVY
        p.space_after = Pt(7)


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def build() -> None:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    for spec in SLIDES:
        slide = prs.slides.add_slide(blank)

        if spec.get("kind") == "title":
            tf = add_textbox(slide, Inches(0.9), Inches(2.1), SLIDE_W - Inches(1.8), Inches(2.2))
            p = tf.paragraphs[0]
            p.text = spec["title"]
            p.font.size = Pt(34)
            p.font.bold = True
            p.font.color.rgb = NAVY
            p.alignment = PP_ALIGN.CENTER
            tf2 = add_textbox(slide, Inches(0.9), Inches(4.6), SLIDE_W - Inches(1.8), Inches(1.4))
            p2 = tf2.paragraphs[0]
            p2.text = spec["subtitle"]
            p2.font.size = Pt(18)
            p2.font.color.rgb = ORANGE
            p2.alignment = PP_ALIGN.CENTER
            add_notes(slide, spec["notes"])
            continue

        add_title(slide, spec["title"])
        image = spec.get("image")

        if image and spec.get("image_wide"):
            # bullets on top, wide image below (sized by height, centered)
            add_bullets(slide, spec["bullets"], Inches(0.6), Inches(1.45), SLIDE_W - Inches(1.2), Inches(2.6))
            pic = slide.shapes.add_picture(str(image), Inches(0.65), Inches(4.3), height=Inches(2.95))
            pic.left = Emu(int((SLIDE_W - pic.width) / 2))
        elif image:
            # bullets left, image right
            add_bullets(slide, spec["bullets"], Inches(0.6), Inches(1.5), Inches(7.1), Inches(5.6))
            pic = slide.shapes.add_picture(str(image), Inches(7.95), Inches(1.6), width=Inches(4.9))
            if pic.height > Inches(5.4):
                scale = Inches(5.4) / pic.height
                pic.height = Inches(5.4)
                pic.width = Emu(int(pic.width * scale))
        else:
            add_bullets(slide, spec["bullets"], Inches(0.6), Inches(1.5), SLIDE_W - Inches(1.2), Inches(5.6))

        add_notes(slide, spec["notes"])

    OUT_PPTX.parent.mkdir(exist_ok=True)
    prs.save(OUT_PPTX)
    print(f"Wrote {OUT_PPTX}")

    lines = ["# Presentation Script - ASEE 2027 Proposal\n"]
    for i, spec in enumerate(SLIDES, 1):
        title = spec["title"].replace("\n", " ")
        lines.append(f"## Slide {i}: {title}\n")
        lines.append(spec["notes"] + "\n")
    OUT_SCRIPT.write_text("\n".join(lines))
    print(f"Wrote {OUT_SCRIPT}")


if __name__ == "__main__":
    build()
