# Face-Audit — Roadmap

Cohesive plan from current state (a working video blur-audit pipeline on one
machine) to a censorship-integrity platform and eventual startup.
Status markers: `[x]` done · `[~]` in progress · `[ ]` not started.

## Phase 0 — Existence & credibility (weeks 1–2)

The project currently can't prove it exists: no site, no license, sparse
profile, no demo artifact a stranger can run.

- [ ] Repo hygiene: commit or discard the local `status_server.py` changes;
      fast-forward local `main` to `origin/main`; review `origin/relay`.
- [ ] Public identity: GitHub profile (avatar, bio, location), repo
      description + topics (`privacy`, `face-detection`, `redaction-audit`),
      an OSI license (MIT or Apache-2.0), and pinned repo.
- [ ] One-page landing site (GitHub Pages is enough): what it does, one GIF
      of the dashboard flagging a failed blur, link to repo. No roadmap
      promises on the public site.
- [ ] Demo pack: a scripted end-to-end run on **synthetic** footage only
      (club_a/club_b + a synthesized-censorship sample) producing the flagged
      gallery + report.csv — reproducible by anyone with `requirements.txt`.
- Acceptance: a stranger can go from the site → repo → running demo in
  under 15 minutes.

## Phase 1 — Video audit to product quality (weeks 2–8)

The current SCRFD + blur/edge/flat metrics + tracking + review loop is the
seed; harden it into something reliable.

- [ ] Benchmark suite: fixed public video set (synthesized censorship at
      known severities) with precision/recall gates in CI on every PR.
- [ ] Ship the learned classifier as default tiebreaker: `train_classifier.py`
      (MobileNetV3-Small, ONNX) promoted from optional to standard path once
      it beats the heuristic F1 on the benchmark set.
- [ ] Hard cases: pixelation + black-box over *moving* faces, partial-box
      (eyes visible), inpainting-style censorship, faces at small scale.
- [ ] Review-loop maturation: from single-reviewer accept/deny to
      label-export → retrain → redeploy cycle documented as a loop.
- [ ] CLI one-liner output contract: `Veilaudit check video.mp4` →
      machine-readable verdict (JSON) + human report. This is what partners
      would actually call.
- Acceptance: benchmark P/R ≥ agreed targets on pixelation and blur classes;
  documented false-positive behavior on motion blur.

## Phase 2 — Document / text-layer auditor (weeks 6–12)

The Epstein failure mode generalized: black boxes drawn over the *text layer*
of a PDF, "redacted" text still selectable, OCR-recoverable scans.

- [ ] PDF auditor: detect text under redaction boxes, extractable text
      behind rectangles, mismatched box/word geometry, metadata leaks
      (author, revision history, embedded original pages).
- [ ] Image-of-document auditor: detect OCR-recoverable text under boxes or
      marker strokes (heavy degradation still leaks structure).
- [ ] Same verdict contract as Phase 1: `Veilaudit check doc.pdf`.
- Acceptance: catches every failure class above on a synthetic redaction
  test corpus built from public-domain documents.

## Phase 3 — Audio distortion auditor (months 3–5)

Where voice distortion/pitch-shifting fails: speaker identity survives.

- [ ] Speaker-verification-based audit: enroll "redacted" speaker segments,
      test whether two redacted segments across a corpus match each other
      (re-identification *without* naming — proves the anonymization failed).
- [ ] Detect machine-distortion artifacts: vocoder/pitch-shift signatures
      that distinguish "anonymized" audio from natural speakers.
- Acceptance: proves linkability of "anonymized" speakers in a test corpus
  with zero ground-truth names — matches the project's "prove censorship
  failed" thesis.

## Phase 4 — Research bet: feature estimation under censorship (months 4–12)

The differentiator: estimating facial features and proportions even when
censorship *succeeds* — from degradation patterns, surrounding geometry,
temporal context, and residual high-frequency structure.

- [ ] Build the eval set first: pairs of (original, censored) faces at
      graded severities; metrics = landmark drift, identity-embedding
      cosine similarity, attribute (age/sex/glasses) accuracy post-censor.
- [ ] Baseline: how much identity leaks through strong pixelation via
      super-resolution + embedding matching (public results suggest: a lot).
- [ ] Model path: restoration-network + embedding-probing pipeline; only
      escalate complexity if the baseline shows headroom.
- [ ] Publish as a preprint + demo regardless of outcome — negative results
      ("pixelation at strength X still leaks Y% identity") are the startup's
      strongest marketing evidence.
- Data strategy for this phase is documented in a separate internal note
  (kept private).
- Acceptance: a quantified leak curve (identity recoverability vs.
  censorship strength) for the top 3 censorship methods.

## Phase 5 — Formation & first revenue (month 9+)

- [ ] Value wedge: sell the *audit*, not the detection — "prove your
      redaction release is clean" for newsrooms, law firms, FOIA shops.
- [ ] Conversations once Phase 1 benchmark is public: legal-aid orgs,
      investigative journalists, privacy NGOs.
- [ ] Entity, license dual-tracking (open core vs. hosted audit service),
      and a name that isn't a placeholder.

## Sequencing logic

Phase 0 costs days and unblocks credibility — do it first. Phase 1 and 2
share the verdict contract and review tooling, so they interleave. Phase 3 and
4 are the risky/differentiating bets; both consume the eval infrastructure
built in Phase 1, which is why the benchmark suite is the true critical path.
