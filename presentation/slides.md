<!--
  Slide content lives here. Use `---` for horizontal slides and `--` for vertical stacks.
  Speaker notes start with "Note:".
-->

<!-- .slide: class="title-slide" data-transition="zoom" -->
<div class="space-y-8 py-6">
  <div class="space-y-3">
    <p class="eyebrow fade-up">Master Thesis Presentation</p>
    <div class="space-y-1">
      <h1 class="font-display leading-tight text-ink">Tracing the Objectives Backwards</h1>
      <h3 class="font-sans text-[0.9em] tracking-[0.02em] font-medium text-slate-600">Data-Driven Inverse Exploration of Multi-Objective Problems</h3>
    </div>
  </div>
  
  <div class="glass-card max-w-xl border-l-8 border-accent">
    <div class="grid grid-cols-2 gap-6 text-[0.6em]">
      <div class="space-y-1">
        <p class="text-muted uppercase tracking-widest font-bold">Author</p>
        <p class="font-semibold text-ink text-base">Nicola Ibrahim</p>
        <p class="text-slate-500">M.Sc. Computer Engineering</p>
      </div>
      <div class="space-y-1 text-right">
        <p class="text-muted uppercase tracking-widest font-bold">Institution</p>
        <p class="font-semibold text-ink text-base">Paderborn University</p>
        <p class="text-slate-500">January 5, 2026</p>
      </div>
    </div>
  </div>
</div>

Note: Welcome everyone. Today I'm presenting my thesis on inverse design—learning to map desired performance targets back to the decisions that produce them.

---

<h2 class="slide-title">Agenda</h2>
<div class="grid grid-cols-2 gap-8 mt-4">
  <div class="space-y-4">
    <div class="glass-card fragment fade-up" data-fragment-index="1">
      <p class="eyebrow">01</p>
      <p class="font-bold text-ink">Motivation & Problem</p>
    </div>
    <div class="glass-card fragment fade-up" data-fragment-index="2">
      <p class="eyebrow">02</p>
      <p class="font-bold text-ink">Inverse-Design Workflow</p>
    </div>
    <div class="glass-card fragment fade-up" data-fragment-index="3">
      <p class="eyebrow">03</p>
      <p class="font-bold text-ink">Models & Evaluation</p>
    </div>
  </div>
  <div class="space-y-4">
    <div class="glass-card fragment fade-up" data-fragment-index="4">
      <p class="eyebrow">04</p>
      <p class="font-bold text-ink">Experiments & Results</p>
    </div>
    <div class="glass-card fragment fade-up" data-fragment-index="5">
      <p class="eyebrow">05</p>
      <p class="font-bold text-ink">Contributions & Outlook</p>
    </div>
  </div>
</div>

---

<!-- .slide: data-auto-animate -->
<h2 class="slide-title">Motivation: Turning Design Upside-Down</h2>
<div class="grid-2">
  <div class="space-y-6">
    <p>Designers start with <strong>performance templates</strong>, not parameters:</p>
    <ul class="space-y-4">
      <li class="fragment fade-up"><strong>Audio:</strong> Prescribed frequency-response shape.</li>
      <li class="fragment fade-up"><strong>Antennas:</strong> Specific spatial beam patterns.</li>
      <li class="fragment fade-up"><strong>Robotics:</strong> Target rate-distortion operating points.</li>
    </ul>
    <div class="callout fragment zoom-in">
      Forward simulation is easy; <strong>inverse decision-making</strong> is the bottleneck.
    </div>
  </div>
  <div class="space-y-4">
    <div class="card bg-slate-50 border-slate-200">
      <p class="text-sm font-semibold mb-2">The Inverse Challenge</p>
      <p class="text-xs text-slate-600">Ill-posedness, non-uniqueness, and high-dimensional search spaces make "manual" trial-and-error impractical.</p>
    </div>
    <img src="assets/forward-inverse.svg" alt="Inverse mapping challenge" class="rounded-xl shadow-lg" />
  </div>
</div>

Note: In practice, we know what we want (outcomes) but not how to get there (decisions).

---

<!-- .slide: data-auto-animate -->
<h2 class="slide-title">Problem Statement</h2>
<div class="space-y-8">
  <div class="glass-card">
    <p class="font-mono text-accent mb-4">Given forward map $f: X \to Y$ and samples $(x_i, y_i) \in \mathcal{D}$:</p>
    <h3 class="text-xl font-bold">The Goal</h3>
    <p>Learn a data-driven inverse rule $\hat{g}: Y \to X$ that proposes <strong>multiple candidate decisions</strong> $\hat{x}$ for a target $y^*$.</p>
  </div>

  <div class="grid grid-cols-3 gap-4">
    <div class="card border-l-4 border-brand">
      <p class="font-bold text-brand text-sm">Non-uniqueness</p>
      <p class="text-[0.6em]">Many $x$ can yield the same $y^*$.</p>
    </div>
    <div class="card border-l-4 border-accent">
      <p class="font-bold text-accent text-sm">Infeasibility</p>
      <p class="text-[0.6em]">Some $y^*$ have no preimage in $X$.</p>
    </div>
    <div class="card border-l-4 border-slate-500">
      <p class="font-bold text-slate-500 text-sm">Stability</p>
      <p class="text-[0.6em]">Small $\Delta y^*$ can cause large $\Delta x$.</p>
    </div>
  </div>
</div>

Note: We need models that can output multiple distinct candidates to handle this ambiguity.

---

<h2 class="slide-title">The Propose-Check Workflow</h2>
<div class="grid-2">
  <div class="space-y-4">
    <ol class="space-y-2">
      <li class="fragment fade-left" data-fragment-index="1">
        <span class="font-bold text-brand">Offline:</span> Collect evaluations and train $\hat{g}(y)$.
      </li>
      <li class="fragment fade-left" data-fragment-index="2">
        <span class="font-bold text-accent">Query:</span> Input $y^*$, sample $K$ candidates $\{\hat{x}_1, \dots, \hat{x}_K\}$.
      </li>
      <li class="fragment fade-left" data-fragment-index="3">
        <span class="font-bold text-ink">Verify:</span> Evaluate $f(\hat{x}_k)$ forward.
      </li>
      <li class="fragment fade-left" data-fragment-index="4">
        <span class="font-bold text-slate-500">Rank:</span> Select best candidate by discrepancy.
      </li>
    </ol>
    <div class="glass-card mt-6">
      <p class="text-xs italic text-muted">"Computational effort shifts from query-time search to one-time offline training."</p>
    </div>
  </div>
  <img src="assets/pipeline.svg" alt="Workflow diagram" class="rounded-2xl" />
</div>

---

<!-- .slide: data-auto-animate -->
<h2 class="slide-title">Modeling Multi-Valued Maps</h2>
<div class="grid grid-cols-3 gap-6 mt-4">
  <div class="glass-card space-y-4 border-t-4 border-brand">
    <h3 class="font-display text-lg">MDN</h3>
    <p class="text-[0.8em]"><strong>Mixture Density Networks</strong></p>
    <p class="text-[0.6em] text-slate-500 line-clamp-3">Approximates $p(x|y)$ as a mixture of Gaussians. Captures distinct modes explicitly.</p>
  </div>
  <div class="glass-card space-y-4 border-t-4 border-accent">
    <h3 class="font-display text-lg">CVAE</h3>
    <p class="text-[0.8em]"><strong>Conditional VAEs</strong></p>
    <p class="text-[0.6em] text-slate-500 line-clamp-3">Learns a latent distribution $z$ conditioned on $y$. High diversity through latent sampling.</p>
  </div>
  <div class="glass-card space-y-4 border-t-4 border-ink">
    <h3 class="font-display text-lg">INN</h3>
    <p class="text-[0.8em]"><strong>Invertible Neural Nets</strong></p>
    <p class="text-[0.6em] text-slate-500 line-clamp-3">Uses bijective architecture to preserve information. Latent space padding handles ill-posedness.</p>
  </div>
</div>
<div class="mt-4">
  <img src="assets/models.svg" alt="Model families" class="h-40 mx-auto" />
</div>

---

<!-- .slide: data-auto-animate -->
<h2 class="slide-title">Evaluation: Outcome-Space Metrics</h2>
<div class="grid-2">
  <div class="space-y-6">
    <div class="card border-l-4 border-brand">
      <p class="eyebrow">Outcome Discrepancy</p>
      <p class="font-mono text-xl text-ink">$E_Y = \|f(\hat{x}) - y^*\|^2$</p>
      <p class="text-xs text-muted">Primary measure of target alignment.</p>
    </div>
    <div class="card border-l-4 border-accent">
      <p class="eyebrow">Success Rate ($Succ_\epsilon$)</p>
      <p class="text-ink font-semibold">Targets within tolerance $\epsilon$.</p>
      <p class="text-xs text-muted">Fraction of queries considered "close enough".</p>
    </div>
  </div>
  <div class="glass-card space-y-4">
    <p class="eyebrow text-slate-400">The "Best-of-K" Principle</p>
    <p class="text-sm">We assess the <strong>best possible design</strong> the user can choose from the candidate set.</p>
    <div class="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
      <div class="h-full bg-accent" style="width: 85%"></div>
    </div>
    <p class="text-[0.6em] text-muted italic text-center">Quality scales with candidate set size $K$.</p>
  </div>
</div>

Note: Emphasize that we evaluate in objective space because that's what the user cares about.

---

<!-- .slide: data-auto-animate -->
<h2 class="slide-title">Experimental Benchmarks</h2>
<div class="grid-2 gap-12">
  <div class="space-y-8">
    <div class="glass-card flex items-start space-x-4">
      <div class="bg-brand text-white p-2 rounded-lg font-bold">01</div>
      <div>
        <h3 class="text-ink mb-1">COCO BBOB (Synthetic)</h3>
        <p class="text-xs text-slate-500">Bi-objective test suite with complex trade-off surfaces and non-convex Pareto fronts.</p>
      </div>
    </div>
    <div class="glass-card flex items-start space-x-4">
      <div class="bg-accent text-white p-2 rounded-lg font-bold">02</div>
      <div>
        <h3 class="text-ink mb-1">dEchorate (Real-World)</h3>
        <p class="text-xs text-slate-500">Acoustic room geometry design. Mapping reverb/echo profiles to 3D sensor positions.</p>
      </div>
    </div>
  </div>
  <img src="assets/datasets.svg" alt="Datasets overview" class="rounded-xl" />
</div>

---

<!-- .slide: data-auto-animate -->
<h2 class="slide-title">Results: Pareto Alignment</h2>
<div class="grid-2">
  <div class="space-y-4">
    <ul class="space-y-4">
      <li class="fragment fade-right">
        <strong>High Precision:</strong> Inverse models reach targets with error lower than grid-search baseline.
      </li>
      <li class="fragment fade-right">
        <strong>Power of K:</strong> Increasing $K$ from 1 to 50 significantly boosts $Succ_\epsilon$.
      </li>
      <li class="fragment fade-right">
        <strong>Generalization:</strong> CVAE shows superior robustness on out-of-distribution targets.
      </li>
    </ul>
    <div class="callout fragment zoom-in">
      Models learn the "Pareto skeleton" of the design space.
    </div>
  </div>
  <div class="space-y-4">
    <img src="assets/pareto-front.svg" alt="Pareto front results" class="h-64 mx-auto" />
    <p class="text-[0.5em] text-center text-muted">Comparison of target specs vs. forward-evaluated results.</p>
  </div>
</div>

---

<h2 class="slide-title">Key Contributions</h2>
<div class="space-y-6 mt-8">
  <div class="glass-card fragment fade-up hover:border-brand transition-colors">
    <p class="font-bold text-ink">01. Formalization</p>
    <p class="text-sm text-slate-600">Unified framing of data-driven inverse design for multi-objective signal processing.</p>
  </div>
  <div class="glass-card fragment fade-up hover:border-accent transition-colors">
    <p class="font-bold text-ink">02. Protocol</p>
    <p class="text-sm text-slate-600">Reproducible "Best-of-K" evaluation pipeline with explicit forward verification.</p>
  </div>
  <div class="glass-card fragment fade-up hover:border-slate-500 transition-colors">
    <p class="font-bold text-ink">03. Empirical Insights</p>
    <p class="text-sm text-slate-600">Detailed characterization of trade-offs between alignment, diversity, and computational cost.</p>
  </div>
</div>

---

<h2 class="slide-title">The Road Ahead</h2>
<div class="grid grid-cols-2 gap-8">
  <div class="space-y-6">
    <div class="card border-l-4 border-accent">
      <h3 class="text-accent">Reliability</h3>
      <p class="text-xs">Integrating <strong>Operational Domain Validators</strong> (ODD) to detect out-of-distribution targets.</p>
    </div>
    <div class="card border-l-4 border-brand">
      <h3 class="text-brand">Scalability</h3>
      <p class="text-xs">Extending to outcome spaces with $D > 10$ using hierarchical latent models.</p>
    </div>
  </div>
  <div class="glass-card bg-slate-900 text-white">
    <p class="eyebrow text-slate-400">Final Takeaway</p>
    <p class="text-sm leading-relaxed">
      Inverse modeling turns passive datasets into <strong>active design assistants</strong>, enabling designers to trace their objectives backwards with mathematical confidence.
    </p>
  </div>
</div>

---

<div class="h-full flex flex-col items-center justify-center text-center space-y-12">
  <h1 class="text-5xl font-display text-ink">Thank You</h1>
  <div class="space-y-4">
    <h3 class="text-accent uppercase tracking-widest font-bold">Questions?</h3>
    <p class="text-muted italic">Tracing the Objectives Backwards</p>
  </div>
  
  <div class="grid grid-cols-3 gap-8 pt-12 text-[0.4em] uppercase tracking-widest font-bold text-slate-400">
    <p>Theory</p>
    <p>Data</p>
    <p>Design</p>
  </div>
</div>

Note: Thank you for your attention. I'm now open for any questions regarding the methodology or results.
