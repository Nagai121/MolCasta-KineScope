# KineScope findings — free to take

This page collects the findings of KineScope, a systematic study of what a
short molecular dynamics (MD) trajectory tells a predictor about future
molecular events. The findings are released for anyone to use: no permission
needed, no payment, no obligation to collaborate, cite, or report back.
Take what is useful and build on it in any way you like.

## Setup in brief

- **Task.** Predict whether a molecular event (hydrogen-bond formation or
  dissociation, contact formation or dissociation, salt-bridge formation or
  dissociation) will occur a given time (1–90 ns) after a prediction origin,
  for candidate atom/residue pairs in a protein.
- **"Guidance".** We call the improvement in prediction accuracy obtained by
  adding the real trajectory history (the first 0–1 ns) as input, compared to
  static structure alone, the *guidance* of the trajectory. On 100 unseen
  proteins × 3 replicas, real history improved AUPRC by +0.0712 at 1 ns and
  +0.0273 at 90 ns (protein-mean differences against a structure-only
  correction model; against the base model itself the candidate-pooled gain
  is +0.0364 — see `docs/facts.md` for the exact baselines).
- **Data.** Trajectories come from the ATLAS database (see "Data and
  license" below). Experiments use three bases: a 6-protein base with
  retrained predictors (2 test proteins), a 30-protein salt-bridge base, and
  a 100-protein base. Every experiment was run with its success/failure
  criteria fixed in advance, and the full records (code, per-experiment
  READMEs, result files) are in `experiments/` in this repository.

## The six findings

### 1. Even 100 ps of real trajectory informs predictions 90 ns ahead

Using only the first 0.1 ns of real history recovers at least half of the
full-history gain in almost every phenomenon × horizon combination tested
(recovery ≥ 0.5). For hydrogen-bond formation at 90 ns, a 0.1 ns prefix
recovers 0.590 of the gain of the full 1 ns history (experiment G03; the
full prefix map is the G series, closed at G180).

### 2. Guidance compresses into a single physical quantity

For hydrogen-bond formation, the pair's mean distance over the history
window alone recovers 0.87–1.08 of the full-history gain (K21). Most of what
the trajectory tells the predictor, for this phenomenon, is carried by one
scalar per pair. (For contact formation and salt-bridge formation the
information is distributed over several statistics instead — see finding 6.)

### 3. The information lives in the part of the mean distance that static structure cannot predict

The per-pair *offset* — the actual window-mean distance minus the mean
distance predicted from static structure alone — carries 0.68–0.94 of the
guidance (K42/K44). The offset has a spread of about 0.49 Å, persists across
the two halves of the window (correlation 0.59–0.73 for all six
phenomenon/direction combinations), and is set by the pair and its initial
configuration rather than by the thermal realization: another replica of the
same pair mostly preserves it (K03).

### 4. Generated trajectories fail because they reproduce what static structure already knows

The mean distances of trajectories generated from a single structure are
~97% explainable by static features (GBM R² 0.976), versus ~63% for real
trajectories (K39), and correlate 0.96–0.98 with the static prediction
itself (K43). A static correction model already uses that signal, so adding
generated history adds almost no new information. This is a measurable
diagnostic any trajectory generator can be run through: *what fraction of
your generated pair-mean distances is already predictable from the static
structure?*

### 5. Guidance is a level, not a trend — and longer history is not always better

Instantaneous pair distance decorrelates within 100 ps (K2-02), yet the
offset (finding 3) persists across window halves: guidance behaves like a
persistent level of the pair, not a temporal pattern. Consistently,
extending the real-history prefix does not monotonically increase recovery
(e.g., salt-bridge formation at 10 ns: recovery 0.868 with a 0.1 ns prefix,
0.393 with a 1.0 ns prefix), and at long horizons the *first* half of the
history can carry more guidance than the second half (K2-05). Adding an
explicit drift term (first-half minus second-half mean or fraction) helps
only in scattered cells: 11 of 36 combinations for the mean, 3 of 36 for the
fraction, with no overlap pattern by horizon — at 90 ns it never helps.

### 6. What carries guidance depends on the phenomenon

- Hydrogen-bond formation/dissociation: the pair mean distance is a
  near-sufficient summary (finding 2).
- Contact formation and salt-bridge formation: the information is
  distributed over the six history statistics; no single summary suffices
  (K27, K31).
- The drift results (finding 5) also split by phenomenon, not by formation
  vs dissociation: drift adds for hydrogen-bond dissociation and contact
  formation at several horizons, never for hydrogen-bond formation or
  salt-bridge dissociation.

## An open task, with the success line fixed in advance

**Predict the offset (finding 3) from static structure alone.**
Target: the per-pair window-mean distance over 0–1 ns, specifically its
static-unpredictable component. Current status: static features leave a
residual std of ≈0.49 Å; the best direct attempt (G01) reached 0.474 Å.
Success criteria, fixed in advance: residual std ≤ 0.30 Å (the replica-to-
replica floor is 0.249 Å), and, using the predicted means as predictor
input, recovery of ≥ 0.5 of the real-history guidance gain on the
100-protein base. Candidate approaches noted in the records include
relational (pair-neighborhood) representations and local coordinate frames
for pairs. Everything needed — data definitions, baselines, and the
evaluation protocol — is in this repository.

## Cost note (measured)

Short real MD is cheaper than it looks, but fixed costs dominate: on the
measurement machine, preparation and equilibration cost ≈2,800 s regardless
of length, and production ran at ≈1.0 ns/day, so a 0.1 ns run costs ≈3.2 h
against ≈24.6 h for 1.0 ns (G04). Combined with finding 1, the practical
route "run 0.1 ns of real MD per system" buys most of the guidance at about
one eighth of the cost of a 1 ns run.

## Data and license

- The underlying trajectories are from **ATLAS**: Vander Meersche, Y. et al.
  (2024) *Nucleic Acids Research* 52(D1), D384–D392,
  doi:10.1093/nar/gkad1084. ATLAS data is released under CC BY-NC 4.0.
  This repository does not redistribute ATLAS trajectories; derived
  quantities computed from ATLAS data inherit the non-commercial condition.
  If you use the data side of this work, cite ATLAS.
- The findings, text, and result files in this repository are released
  under CC0: use them for any purpose, commercial or not, with no
  obligation. Attribution is appreciated but not required.
- Code in this repository is released under the MIT license.

## Where things are

- `docs/facts.md` — verified facts and cross-experiment condition notes
  (Japanese).
- `docs/g-series-summary.md` — summary of the short-real-MD recovery map
  (Japanese).
- `docs/hypotheses.md` — hypothesis registry and experiment index
  (Japanese).
- `experiments/` — one directory per experiment, each with a README stating
  the hypothesis, the conditions changed, the result, and what remains
  unresolved (mostly Japanese).

A citable DOI will be added when the repository snapshot is archived.
