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
  vs dissociation: drift adds for hydrogen-bond dissociation (1, 10, 30 ns)
  and contact formation (1, 5, 10, 30, 50 ns), for salt-bridge formation
  only at 5 and 10 ns (the two largest drift gains in the whole mean-drift
  map, +0.0437 and +0.0413), and for contact dissociation only at 1 ns; it
  never adds for hydrogen-bond formation or salt-bridge dissociation, and
  never at 90 ns for any combination.

## An open task, with the success line fixed in advance

**Predict the offset (finding 3) from static structure alone.**
Target: the per-pair window-mean distance over 0–1 ns, specifically its
static-unpredictable component. Current status: static features leave a
residual std of ≈0.49 Å; the best direct attempt (G01) reached 0.474 Å.
A separate line of attempts that predicts the offset with learned models
(the GA series, run on the Codex side) plateaued at a residual correlation
of about 0.61–0.62 in validation, and its best model (GA386) scored 0.620
on a held-out blind panel (3,980 pairs), versus 0.622 for its parent
baseline — the validation gain did not survive blind scoring, and the 0.700
line set for that series was not reached (GAm01).
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

## Public Verification Update (E01–E04, 2026-11-26)

Four 100% public, leak-free verification experiments (`E01–E04`, zero proprietary `T1I2006`/`T1I2007` dependencies, 5-fold protein-grouped CV) were run directly on public ATLAS trajectories, OpenMM GPU simulations, and official `MDGen` weights to audit and update the six findings above:

- **Finding 1 & Cost Note revised ([E01](experiments/E01_true_short_md_guidance/README.md))**:
  - The G/L series claim that "0.1 ns recovers ≥ 0.5 at all horizons" was inflated by a **1.0 ns time gap** between early frames `[0, 1]` and origin frame `10` (even single frame `f0` alone scored `0.38–0.54`). Under a strict steady-state adjacent window (`[9, 10]`, 100 unseen proteins, 100% public 20-feature baseline), true `0.1 ns` recovers only **0.36–0.44** (HB formation) and **0.28–0.45** (HB dissociation), failing `0.50` on 6/6 horizons.
  - A **true zero-gap forward prefix** (`0.0 → 0.1 ns`, moving origin `t = L`) recovers **0.48–0.60** (`0.579` at 90 ns), revealing an *initial-relaxation vs steady-state asymmetry*; in steady state, `0.2–0.5 ns` is the practical sweet spot (`0.5 ns` = **0.75–0.88**), and `1.0 ns mean` alone captures **0.90–0.96** of 6-stats guidance.
  - On a real consumer GPU (`RTX 4060 Ti`, `123–188 ns/day` explicit solvent AMBER14 + TIP3P + standard 200 ps equilibration), fixed overhead is **66–69%** of a 0.1 ns run, giving a true `0.1 ns / 1.0 ns` cost ratio of **0.27–0.28 (~1/3.6, not 1/8)** (`1mix_A`: 3.8 min vs 14.2 min).
- **Finding 3 & K3 static-elimination wall established 100% publicly ([E02](experiments/E02_public_k3_and_replica_floor/README.md))**:
  - First direct measurement of the **100-protein replica floor** across 4 phenomena (`484,049` pairs): for HB formation, `static20` residual SD is **0.561 Å** vs single-replica thermal noise **0.201 Å** (3-replica diff SD `0.285 Å`, **2.8× S/N**), and independent replicas agree on the *static residual itself* (`r = +0.388` HB form, `+0.568` HB diss, `+0.249` contact, `+0.416` salt bridge).
  - Combining all 4 K3 static physical families simultaneously (`68` features, `T1I2006`-free, 5-fold CV) reduces residual SD only from `0.602` to `0.574 Å` (`9.1%` of residual variance); protein-atom `K3-02` H-bond network features (`10` features) also fail (`0.561 → 0.559 Å`, `max|r| = 0.025`).
- **Open Task Breakthrough — water-free 100 ps implicit-solvent MD shatters the K3 wall ([E03](experiments/E03_implicit_md_relaxation_vs_k3/README.md))**:
  - Starting strictly from single static frame `10` (`d0_err = 0.0000 Å`, 25 unseen ATLAS proteins, `N = 13,687`, **55 s/protein** on RTX 4060 Ti), `4 × 100 ps` OpenMM GBn2 implicit-solvent MD (zero explicit water) recovers **`r = +0.520`** of the explicit-solvent residual (`SD: 0.603 → 0.513 Å`, `-0.090 Å`) and **55.4–85.7% (mean-only, ≥55% on 6/6 horizons)** / **up to 115.6% (all 7 stats at 90 ns)** of explicit 1 ns MD's downstream 1–90 ns H-bond formation guidance.
- **Finding 4 refined on real SOTA generative MD (`MDGen`, [E04](experiments/E04_public_generator_static_redundancy/README.md))**:
  - On 25 strictly unseen `splits/atlas_test.csv` proteins (avoiding the 88/100 overlap between KineScope's test set and MDGen's `atlas_train.csv`, **40 s/protein**), official `MDGen` (`atlas.ckpt`) is **not** statically redundant ($R^2 = 0.594$ [1.2 ns] → `0.485` [100 ns] vs real `0.662`); the $R^2 \approx 0.96$ collapse was specific to KineScope's internal generators.
  - Instead, E03 vs E04 exposes a **two-regime split** at equal ~40–55 s/protein GPU cost: implicit physics (E03) dominates sub-ns local offset (`r = +0.520`, `1 ns rec = 58.2%`), whereas MDGen is weak at `1 ns` (`r = +0.266`, `-5.2% to +24.6%`) but genuinely recovers **55.0–77.8%** of explicit 1 ns MD guidance across **5–90 ns** via its 100 ns rollout.

## Where things are

- `docs/facts.md` — verified facts and cross-experiment condition notes
  (Japanese).
- `docs/g-series-summary.md` — summary of the short-real-MD recovery map
  (Japanese).
- `docs/hypotheses.md` — hypothesis registry and experiment index
  (Japanese).
- `docs/literature/` — prior literature review and novelty audit
  (`01`–`04`, including post-E01–E04 update).
- `experiments/` — one directory per experiment (`K*`, `K2*`, `K3*`, `G*`, `L*`,
  and public verification series `E01`–`E04`), each with a README, code, and
  result files.

A citable DOI will be added when the repository snapshot is archived.
