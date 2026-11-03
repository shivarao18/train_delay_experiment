# Presentation Script — Professor Meeting
**"Beyond Accuracy: Teaching Trustworthy AI Evaluation to Undergraduate Civil Engineers"**
*Estimated time: 15–18 minutes + discussion*

---

## Before You Start
Set the PDF to full-screen. Have the abstract draft open on your phone or a second window in case questions come up. The goal of this meeting is **to get a course name, a semester, and a yes** — keep bringing the conversation back to that.

---

## Slide 1 — Title
**[~1 min]**

> "Thanks for making time. What I want to share today is a proposed teaching module — something I've been developing that I think could slot into one of our upper-division courses. The core idea is this: we teach students how to build and evaluate AI models, but we rarely teach them how to judge whether an AI prediction is actually trustworthy enough to act on in a professional context. This module addresses that gap, using a real railway dataset and a learning design that connects directly to what civil engineers will face in practice. I'll walk you through the design, show you some of the technical results I've already validated, and then I want to get your input on the best course to pilot this in."

*[advance slide]*

---

## Slide 2 — The Problem
**[~2 min]**

> "Let me start with a concrete result, because I think it illustrates the problem better than any argument I could make."

> "We have a prediction model — a well-trained XGBoost model — operating on real Belgian railway data. The model publicly claims 90% confidence in its prediction intervals. And on average, it actually delivers that: 90.6% marginal coverage across all test cases."

> "But when you slice by operating condition, that headline number starts to fall apart. For trains that are already heavily delayed — the ones where an engineer most needs a reliable prediction to make a decision — the actual coverage drops to 71.1%. Not 90%. 71. And for long-horizon predictions, 45 minutes or more ahead, it drops to 71.4%."

> "So a system that looks perfectly calibrated on paper is genuinely unreliable in the scenarios where you'd rely on it most. That's a professional judgment problem, not just a statistics problem."

> "And here's the issue: our students graduate knowing how to compute accuracy. They don't have the vocabulary or the mental models to detect this kind of conditional failure. That's what this module is designed to fix."

*[advance slide]*

---

## Slide 3 — Educational Objective
**[~1.5 min]**

> "So the educational objective is very specific. It's not 'understand machine learning' — it's this: *can a student evaluate whether a given AI prediction is sufficiently reliable to act on professionally?*"

> "I've broken that into five concrete learning outcomes on the slide. The ones I think are most novel for a civil engineering curriculum are these two: being able to identify when a model performs well overall but fails under specific conditions — the conditional reliability problem I just showed — and being able to recommend appropriate engineering responses when predictions are uncertain. That's a professional judgment skill, not a coding skill."

> "And importantly, this is designed for students who have seen basic statistics and regression, but haven't taken a machine learning course. No advanced prerequisites."

*[advance slide]*

---

## Slide 4 — Module Progression
**[~2 min]**

> "The module is structured as four sequential units, and each one targets a progressively higher level of Bloom's taxonomy."

> "Module 1 uses what learning scientists call cognitive conflict. Students will see XGBoost beat the naïve baseline overall — and then discover it loses badly in the most common real-world scenario. That surprise is what drives genuine inquiry."

> "Module 2 introduces prediction intervals through dual coding — pairing numerical tables with calibration diagrams — so students build two mental representations of the same concept, which research shows improves retention."

> "Module 3 is the situated cognition piece — students analyze real predictions under real operating conditions: disruptions, congestion, long forecast horizons. They're not working with a toy problem."

> "And Module 4 is the transfer test — can they apply what they've learned to infrastructure decision scenarios they haven't seen before? That's Bloom's level five: evaluation."

> "The whole progression takes three to four weeks, though I can compress it if the course schedule is tighter — I want to talk about that with you."

*[advance slide]*

---

## Slide 5 — The Dataset
**[~1.5 min]**

> "The data we're using is the RIDE Gold Lite benchmark — it's an open dataset of Belgian passenger railway operations, published earlier this year. Nationwide scale, three years of data, 2023 through 2025."

> "What makes it ideal for this module is that the temporal split is built in. You train on 2023–2024, you test on 2025. So temporal drift — the question of whether a model degrades over time as the world changes — is inherent to the task, not something I have to manufacture."

> "The dataset is openly licensed, so students can look at the raw data, verify what we're telling them, even extend it for projects. And we provide starter code and partially prepared data, so the learning focus stays on *evaluating* the predictions, not on data cleaning or model tuning."

> "The test set I'm showing results on today has 2.1 million real predictions. These aren't simulated examples."

*[advance slide]*

---

## Slide 6 — Module 1 Results
**[~2 min]**

> "This is the heart of Module 1, and I want to spend a moment on it because it's the most important pedagogical moment in the entire module."

> "Students run the model and see the overall MAE: XGBoost at 87.8 seconds, the naïve baseline at 98. XGBoost wins. They feel good about their model."

> "Then they slice by the type of delay change that actually occurred. And look at the highlighted column — the ±30 second bin. That's trains whose delay barely changed. It's the most common scenario: 727,000 predictions, 34.7% of the entire test set."

> "In that regime, the naïve baseline achieves 14.1 seconds MAE. XGBoost gets 32 seconds. The baseline wins by a factor of 2.3."

> "Now, XGBoost absolutely earns its keep in the high-stakes bins — large delays, big worsening — where the baseline fails badly. But that initial surprise, the moment students realize the 'smarter' model loses in the most common situation, is exactly the kind of productive confusion that forces them to think more carefully about *when* accuracy matters and *for whom*."

> "That's the cognitive conflict that sets up everything that follows."

*[advance slide]*

---

## Slide 7 — Module 2 Results
**[~1.5 min]**

> "In Module 2, students move from point predictions to prediction intervals. They compare two methods side by side."

> "The first is split conformal prediction — it gives intervals of nearly constant width, with a mathematical guarantee on marginal coverage. The second is conformalized quantile regression, or CQR — it produces adaptive intervals that widen when predictions are genuinely harder."

> "Looking at the table — both methods hit their stated nominal coverage. Split conformal at 90.6% against a 90% target, CQR at 90.1%. So both are honest on average."

> "But look at the median width column. CQR's median 90% interval is 312 seconds versus 363 for split conformal. And for short-horizon easy predictions, CQR narrows all the way to about 186 seconds. That's the adaptivity advantage — the interval actually tells you something about the *difficulty* of this specific prediction."

> "And this chart on the left is the reliability diagram — a calibration curve. If a method is perfectly calibrated, its points fall on that diagonal line. Students learn to read this diagram and ask: does the model's stated uncertainty actually match what it delivers?"

*[advance slide]*

---

## Slide 8 — Module 3: Regime Failures
**[~2 min]**

> "Module 3 is where everything comes together. We take those same 90% intervals from Module 2 and ask: do they hold *conditionally*, not just on average?"

> "These bar charts show coverage sliced by delay regime and by congestion level. The dashed line is the 90% target. Look at the heavily delayed bar for split conformal — 71.1%. CQR does better at 86.2%, but it's still below the stated 90%."

> "In the congestion panel, split conformal drops to 85% during high-congestion periods. Again, the method that looks fine overall has a real hole in exactly the operating conditions where infrastructure reliability matters most."

> "This is what I mean by conditional reliability versus marginal coverage. And this is the concept I want civil engineering students to internalize — because in practice, you don't operate in average conditions. You operate during disruptions. That's when you most need a reliable prediction, and that's exactly when this class of methods fails silently."

*[advance slide]*

---

## Slide 9 — Module 3: Temporal Drift
**[~1.5 min]**

> "This slide has an interesting result that I want to flag specifically."

> "Look at the right panel — coverage by calendar month across all of 2025. Both methods are stable: split conformal stays between 87.9% and 92.5%, CQR between 88.4% and 90.9%. There's no systematic degradation from January to December."

> "That challenges a common assumption: that models go stale over time as the world drifts away from the training data. For this dataset, that's not the main story. The main reliability risks are operating condition — delay regime and forecast horizon — not time since training."

> "I think that's a genuinely interesting and practical finding. It has implications for how you audit a deployed AI system: don't just look at when it was last retrained. Look at what conditions it's currently operating under."

> "And that kind of nuanced judgment is exactly what I want students to be able to make."

*[advance slide]*

---

## Slide 10 — Assessment Framework
**[~1.5 min]**

> "For the research side, I'm proposing a mixed-methods evaluation framework with four instruments."

> "A pre and post conceptual test — specific items targeting the threshold concepts from each module, like asking students to choose between two models given their calibration curves rather than their accuracy scores."

> "Rubric-scored assignments where students interpret actual reliability diagrams and justify interval comparisons."

> "Scenario-based decision tasks — this is Module 4's assessment — where students are given a real prediction and have to decide whether to act on it, with written justification."

> "And attitudinal surveys to measure how students' perceived relevance of trustworthy AI changes through the module."

> "I want to be upfront: the student data is pending. The technical pipeline is fully validated — all the results I've shown today are real. But to generate the educational outcome data, I need a course to run this in. Which is exactly what I want to talk about with you."

*[advance slide]*

---

## Slide 11 — The Ask
**[~1.5 min + open discussion]**

> "So here's where I need your help. I have five specific questions."

> "First — which course? I'm thinking something at the 3000–4000 level where students have seen regression and basic statistics, but aren't ML specialists. Does anything in our current curriculum come to mind?"

> "Second — which semester? ASEE 2027 abstract submissions are typically due in October 2026 — that's about six weeks from now. If I can confirm a Spring 2027 pilot, I can submit with a note that preliminary data will be available at the conference."

> "Third — scope. The full design is three to four weeks, but I can compress it to one or two weeks inside an existing data analysis unit if that's easier to fit."

> "Fourth — IRB. Is there an existing educational research protocol at UF I can join, or do I need to file a new expedited review?"

> "And fifth — and this one is practical — if the pre/post test is embedded as a low-stakes graded quiz, response rates are much higher and the data is more valid. Is that something you'd be comfortable with?"

> "I'm happy to take any of these questions in whatever order makes sense to you."

---

## Closing / If You Have Extra Time

If the professor asks about the novelty angle:
> "What distinguishes this from existing ML or data literacy courses is that we're not teaching students to train models better. We're teaching them to *audit* models — to ask whether a deployed AI system is reliable enough to trust in a specific context. That's a professional engineering judgment skill, and to my knowledge there's no structured module for civil engineers that puts conformal coverage and conditional reliability at the center of that judgment."

If asked about generalizability:
> "The railway context is the vehicle, but the concepts — prediction intervals, calibration, conditional coverage — transfer directly to structural health monitoring, traffic prediction, flood forecasting, any domain where civil engineers are consuming AI outputs. Module 4 actually asks students to make that transfer explicitly."

---

## Timing Guide

| Slide | Target Time | Cumulative |
|---|---|---|
| 1 — Title | 1 min | 1 min |
| 2 — Problem | 2 min | 3 min |
| 3 — Objective | 1.5 min | 4.5 min |
| 4 — Module Progression | 2 min | 6.5 min |
| 5 — Dataset | 1.5 min | 8 min |
| 6 — Module 1 | 2 min | 10 min |
| 7 — Module 2 | 1.5 min | 11.5 min |
| 8 — Module 3 Regime | 2 min | 13.5 min |
| 9 — Module 3 Drift | 1.5 min | 15 min |
| 10 — Assessment | 1.5 min | 16.5 min |
| 11 — The Ask | open | 18+ min |
