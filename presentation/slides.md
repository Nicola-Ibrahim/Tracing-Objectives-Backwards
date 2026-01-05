<!--
  Slide content lives here. Use `---` for horizontal slides and `--` for vertical stacks.
  Speaker notes start with "Note:".
-->

<!-- .slide: class="title-slide" data-transition="zoom" -->
<p class="eyebrow">Master Thesis</p>
<h1>Tracing the Objectives Backwards</h1>
<h3>Data-Driven Inverse Exploration of Multi-Objective Problems</h3>

<p class="title-meta"><span class="title-name">Nicola Ibrahim</span></p>

<p class="title-meta">M.Sc. Computer Engineering · Signal, Image and Speech Processing</p>
<p class="title-meta">Department of Electrical Engineering and Information Technology</p>
<p class="title-meta"><span class="title-accent">Paderborn University</span> · Filing date: January 5, 2026</p>

Note: One-line pitch: we learn an inverse map so designers can query targets directly.

---

## Agenda
1. Motivation & problem statement <!-- .element: class="fragment" -->
2. Inverse-design workflow <!-- .element: class="fragment" -->
3. Models and evaluation <!-- .element: class="fragment" -->
4. Experiments & results <!-- .element: class="fragment" -->
5. Contributions & outlook <!-- .element: class="fragment" -->

---

<!-- .slide: data-auto-animate -->
## Motivation: Inverse Design in Practice
- We often start with *desired outcomes*, not parameters. <!-- .element: class="fragment" -->
- Forward simulation is easy; inverse decision-making is hard. <!-- .element: class="fragment" -->
- Multi-objective targets add non-uniqueness and trade-offs. <!-- .element: class="fragment" -->

Note: Give a signal-processing example: filter response, beam pattern, rate-distortion.

---

<!-- .slide: data-auto-animate -->
## Problem Statement
Given a forward map `f: X -> Y` and samples `(x_i, y_i)`:

- Learn an inverse rule `g: Y -> X` that proposes **multiple candidates** for a target `y*`. <!-- .element: class="fragment" -->
- Verify each candidate by forward evaluation and select the best. <!-- .element: class="fragment" -->

Note: Stress ill-posedness: one-to-many, noisy, sometimes infeasible.

---

## Forward vs. Inverse View
<div class="grid-2">
  <div>
    <h3>Key Tension</h3>
    <p>Forward: decisions -> outcomes (well-posed).</p>
    <p>Inverse: outcomes -> decisions (ill-posed, ambiguous).</p>
    <div class="callout">
      Inverse exploration must be **data-driven** and **verified forward**.
    </div>
  </div>
  <div>
    <img src="assets/forward-inverse.svg" alt="Forward and inverse mapping diagram" />
  </div>
</div>

---

## Research Questions
1. How do we map a new target `y*` to candidate decisions efficiently? <!-- .element: class="fragment" -->
2. How do we score candidates with explicit, reproducible metrics? <!-- .element: class="fragment" -->
3. Which modeling choices generalize best to unseen targets? <!-- .element: class="fragment" -->

---

<!-- .slide: data-auto-animate -->
## Inverse-Querying Workflow
<div class="grid-2">
  <div>
    <ol>
      <li class="fragment">Collect forward evaluations `(x, y)`.</li>
      <li class="fragment">Train inverse model `g` on outcome space.</li>
      <li class="fragment">Query with `y*` to produce candidate set.</li>
      <li class="fragment">Forward-check and rank with outcome discrepancy.</li>
    </ol>
    <p class="muted">Offline training, fast online querying.</p>
  </div>
  <div>
    <img src="assets/pipeline.svg" alt="Propose-check workflow" />
  </div>
</div>

---

<!-- .slide: data-auto-animate -->
## Modeling Choices
<div class="grid-2">
  <div>
    <ul>
      <li class="fragment"><strong>MDN</strong>: probabilistic multi-modal outputs</li>
      <li class="fragment"><strong>CVAE</strong>: conditional generative latent space</li>
      <li class="fragment"><strong>INN</strong>: invertible mapping with latent noise</li>
    </ul>
    <p class="muted">Goal: generate diverse candidates for each target.</p>
  </div>
  <div>
    <img src="assets/models.svg" alt="Model families diagram" />
  </div>
</div>

---

<!-- .slide: data-auto-animate -->
## Evaluation Protocol
<div class="grid-2">
  <div>
    <ul>
      <li class="fragment"><strong>Outcome discrepancy</strong> `E_Y = d(f(x), y*)`</li>
      <li class="fragment"><strong>Best-of-K</strong> selection from candidate set</li>
      <li class="fragment"><strong>Tolerance success</strong> `Succ_ε` rates</li>
      <li class="fragment"><strong>Calibration</strong> with PIT and CRPS</li>
    </ul>
  </div>
  <div class="card">
    <p class="eyebrow">Reporting principle</p>
    <p>What matters is the set the user gets, not a single point.</p>
  </div>
</div>

---

<!-- .slide: data-auto-animate -->
## Experimental Benchmarks
<div class="grid-2">
  <div>
    <h3>Bi-objective COCO BBOB</h3>
    <p>Controlled synthetic tasks for objective trade-offs.</p>
    <h3>Real-world Signal Processing</h3>
    <p>dEchorate dataset for acoustic room geometry design.</p>
  </div>
  <div>
    <img src="assets/datasets.svg" alt="Datasets overview" />
  </div>
</div>

---

<!-- .slide: data-auto-animate -->
## Results: Pareto Trade-offs
<div class="grid-2">
  <div>
    <ul>
      <li class="fragment">Inverse models reach targets with low outcome error.</li>
      <li class="fragment">Best-of-K improves quality and diversity.</li>
      <li class="fragment">Coverage limits are the dominant failure mode.</li>
    </ul>
  </div>
  <div>
    <img src="assets/pareto-front.svg" alt="Pareto front illustration" />
  </div>
</div>

---

## Contributions
1. Clear formulation of inverse decision mapping for multi-objective targets. <!-- .element: class="fragment" -->
2. Reproducible propose-check workflow with forward verification. <!-- .element: class="fragment" -->
3. Set-based evaluation protocol reflecting user-facing outputs. <!-- .element: class="fragment" -->
4. Empirical comparison of modeling strategies and generalization. <!-- .element: class="fragment" -->

---

## Limitations & Future Work
- Coverage-aware reliability remains critical for deployment. <!-- .element: class="fragment" -->
- Extend to higher-dimensional outcome spaces. <!-- .element: class="fragment" -->
- Add calibration layers for guaranteed outcome bounds. <!-- .element: class="fragment" -->

---

# Thank You
Questions?

Note: End by restating the key takeaway: fast inverse querying + forward verification.
