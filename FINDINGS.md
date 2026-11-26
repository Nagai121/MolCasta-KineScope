# KineScope findings — free to take

This repository summarizes an empirical study of **what a short molecular dynamics (MD) trajectory tells a predictor about future local protein events (1–90 ns ahead)**, why static structural features fail to capture that signal, and how far lightweight implicit-solvent physics and generative MD models can recover it.

All findings and result files are released under **CC0** (code under **MIT**): use them for any purpose, with no obligation to collaborate, cite, or reply.

---

## The question

Take a single protein snapshot and a pair of donor/acceptor atoms that are currently close (3.5–6.0 Å) but not yet hydrogen-bonded. Now try to predict whether that pair will form a hydrogen bond (or break an existing one, or form a contact or salt bridge) **1 to 90 ns later**.

Adding just **1 ns of explicit-solvent MD history** substantially improves prediction accuracy all the way out to 90 ns—even on unseen proteins and even on top of a 20-feature static structural baseline (for example, +0.094 AUPRC at 1 ns and +0.054 at 90 ns for hydrogen-bond formation; +0.215 at 1 ns and +0.144 at 90 ns for contacts, measured via 5-fold protein-grouped cross-validation across 100 unseen proteins × 3 replicas from the ATLAS database, 484,049 candidate pairs).

**Why does a 1 ns trajectory help predict what happens 90 ns later, and can we obtain that signal without running explicit-solvent MD?**

---

## Four main findings

### 1. Almost all of the 90 ns predictive gain comes from one number: the pair's local equilibrium distance

For hydrogen-bond formation and dissociation, the pair's **mean distance over the 1 ns window alone** recovers **90–96%** of the full 6-statistic trajectory gain across all horizons (1–90 ns). Fluctuation amplitude, minimum distance, and short-term trend add very little.

Why does a 1 ns mean stay informative 90 ns later? Not because 1 ns of motion encodes long-term trajectory memory, but because a single static snapshot contains local thermal and packing distortion. Short relaxation moves the pair toward the center of its local metastable basin—an equilibrium offset that persists across halves of the window (`r = 0.58–0.67`) and across independent MD replicas starting from the same structure (`r = 0.39` for H-bond formation, `0.57` for H-bond dissociation, `0.42` for salt bridges). Across 100 unseen proteins, the residual error left by static features (`0.56 Å` for H-bond formation) is **2.8× larger** than the single-replica thermal noise floor (`0.20 Å`).

### 2. Static local physics features cannot predict this offset

We systematically tested six physical feature families computed from the static starting structure across the same 100 unseen proteins:

- sidechain dihedral angles (χ)
- backbone dihedral angles (φ, ψ)
- local electrostatic field at the pair midpoint
- surrounding hydrogen-bond network and competing partners (including water)
- bridging water molecules between the pair
- full atomistic electrostatic and van der Waals interaction energies (CHARMM36m)

None of them explains the offset individually, and **combining all of them simultaneously explains only 9–12% of the residual variance** (reducing residual SD from `0.60 Å` to `0.56–0.57 Å`, far above the `0.20 Å` noise floor). The residuals also show almost no spatial clustering across nearby pairs unless they share a residue (`r = 0.10` for shared-residue pairs; `0.02` for nearby non-shared pairs). In short, the offset appears to be a genuine many-body relaxation property rather than a local static descriptor.

### 3. Water-free implicit-solvent relaxation (4 × 100 ps, ~55 s/protein) breaks the static wall

Starting strictly from the same single static frame—with **no explicit water molecules**—we ran four short (100 ps) implicit-solvent MD relaxations per protein (GBn2 continuum solvent with the AMBER14 force field, averaging ~55 seconds per protein on a consumer RTX 4060 Ti GPU across 25 unseen proteins, 13,687 pairs).

This lightweight relaxation breaks through the barrier where all six static feature families failed:

- Its pair-mean offset correlates with the true explicit-solvent residual at **`r = 0.52`**, cutting the prediction residual from `0.60 Å` down to `0.51 Å`.
- Using **only the single mean distance** from implicit-solvent MD recovers **55–86%** of explicit 1 ns MD's downstream H-bond prediction gain across every horizon from 1 ns to 90 ns.
- Using all summary statistics reaches **67–69% at 10–50 ns** and **116% at 90 ns** (averaging four short trajectories suppresses single-replica thermal noise).

Explicit water molecules are therefore *not* required to capture most of the local equilibrium offset: fast many-body steric and electrostatic relaxation in a continuum solvent already recovers the majority of the signal at roughly 1/10–1/15 the wall-clock cost of explicit 1 ns MD.

### 4. Implicit physics vs. a generative MD model (MDGen) split cleanly into two time regimes

At a comparable GPU budget (~40 seconds per protein on the same GPU), we ran the published all-atom generative MD model **MDGen** (`atlas` checkpoint, 100 ns / 250-frame rollouts) starting from the same single frame. Because 88 of our 100 proteins appear in MDGen's training set, we evaluated MDGen strictly on **25 held-out proteins from its own official test split** (11,285 pairs, zero training overlap).

| Method (from single static frame) | GPU time / protein | Residual alignment with explicit 0–1 ns MD (`r`) | 1 ns H-bond gain recovered | 5–90 ns H-bond gain recovered |
|---|---:|---:|---:|---:|
| 6 static physical feature families combined | <1 s | weak (<12% variance) | — | — |
| **4 × 100 ps implicit-solvent MD (GBn2)** | **~55 s** | **+0.52** | **58%** (mean only) | **55–86%** (mean) / **59–116%** (all stats) |
| **MDGen (100 ns generative rollout)** | **~40 s** | **+0.27** | **−5% to +25%** | **55–78%** |

Two clear lessons emerge:

- **Short-horizon local relaxation (1 ns):** Implicit-solvent physics clearly wins. MDGen is trained on 400 ps-stride trajectories rather than sub-nanosecond force relaxation, so it captures only about half as much of the local offset (`r = 0.27`) and adds little at 1 ns.
- **Long-horizon conformational drift (5–90 ns):** MDGen is *not* merely repeating static structure (its static explainability R² drops to `0.49`, below real MD's `0.66`), and its 100 ns rollout genuinely recovers **55–78%** of explicit 1 ns MD's predictive gain at 5–90 ns.

---

## Short explicit-solvent MD: true scaling and GPU cost

When running explicit-solvent MD directly (without any time gap between the history window and the prediction origin):

- **Initial relaxation vs. steady state:** A 0.1 ns window immediately following the starting structure (`0.0 → 0.1 ns`) recovers **48–60%** of the 1 ns gain (58% at 90 ns), whereas a steady-state 0.1 ns window (`0.9 → 1.0 ns`) recovers only **36–44%** for H-bond formation and **28–45%** for H-bond dissociation.
- **Practical window length:** In steady state, **0.2–0.5 ns** is the reliable sweet spot (`0.5 ns` recovers **75–88%** across horizons).
- **Real GPU cost:** On an RTX 4060 Ti running explicit AMBER14 + TIP3P (`123–188 ns/day`) with standard 200 ps equilibration, fixed preparation and equilibration account for **66–69%** of a 0.1 ns run. The true `0.1 ns / 1.0 ns` wall-clock cost ratio is **0.27–0.28 (~1/3.6)**—e.g., 3.8 min vs. 14.2 min for a 206-residue protein (`1mix_A`).

---

## Open directions

If you want to build on these findings, two concrete directions stand out:

1. **Direct equilibrium-offset prediction from 3D structure:** Because 90–96% of the downstream gain compresses into one reproducible scalar per pair (with 2.8× signal-to-noise across replicas), training a 3D equivariant model directly to predict the sub-ns equilibrium distance offset—instead of generating full trajectories—may be a much cleaner target.
2. **Short implicit physics + generative MD hybrids:** Combining ~55 s of continuum-solvent relaxation (strong at 0–1 ns) with a ~40 s generative rollout (strong at 5–90 ns) is a natural next benchmark.

---

## Revision note (what changed from earlier versions of this repository)

Earlier internal experiments in this repository (preserved in `experiments/` for transparency) used a proprietary base predictor, a CPU-only 20 ps equilibration benchmark, internal prototype generators, and a history window (`[0, 1]`) separated by a 1.0 ns gap from the origin frame (`10`). When re-tested under 100% public, leak-free conditions (`experiments/E01`–`E04`), three earlier claims were revised or retired:

- **"0.1 ns steady-state MD recovers ≥50% at all horizons"** → retired (inflated by the 1.0 ns time gap; true steady-state 0.1 ns is 36–44%, while initial `0.0 → 0.1 ns` relaxation is 48–60%).
- **"0.1 ns explicit MD costs ~1/8 of 1.0 ns"** → revised (true GPU cost with 200 ps equilibration is ~1/3.6, ratio `0.27–0.28`).
- **"Generative MD models suffer ~97% static redundancy and zero guidance"** → retired as a general claim (true of our internal prototype generators, not of published `MDGen`, which shows static R² = `0.49` and 55–78% recovery at 5–90 ns).

---

## Data and license

- Underlying trajectories come from **ATLAS**: Vander Meersche, Y. et al. (2024) *Nucleic Acids Research* 52(D1), D384–D392, doi:10.1093/nar/gkad1084 (CC BY-NC 4.0). Trajectories are not redistributed here; derived quantities inherit ATLAS's non-commercial condition. Please cite ATLAS if using the data side.
- Findings, text, and result files in this repository: **CC0**.
- Code in this repository: **MIT**.

## Experiment index

| Series | What it contains | Link |
|---|---|---|
| Public verification (short explicit MD & GPU cost) | 100-protein zero-gap scaling (4 phenomena) + RTX 4060 Ti explicit-solvent timings | [experiments/E01_true_short_md_guidance](experiments/E01_true_short_md_guidance/README.md) |
| Public verification (static wall & replica floor) | 100-protein replica noise floor + simultaneous 5-family static elimination (no proprietary features) | [experiments/E02_public_k3_and_replica_floor](experiments/E02_public_k3_and_replica_floor/README.md) |
| Public verification (implicit-solvent relaxation) | 4 × 100 ps OpenMM GBn2 relaxation vs static features & explicit 1 ns MD (25 unseen proteins) | [experiments/E03_implicit_md_relaxation_vs_k3](experiments/E03_implicit_md_relaxation_vs_k3/README.md) |
| Public verification (generative MD diagnostic) | MDGen evaluated on 25 strictly held-out test proteins | [experiments/E04_public_generator_static_redundancy](experiments/E04_public_generator_static_redundancy/README.md) |
| Static-physics elimination & spatial clustering | Individual physical feature tests (χ, H-bond network, E-field, φ/ψ, bridging water, CHARMM36m energy, combined, clustering) | `experiments/K3-01_*` – `K3-08_*` |
| Literature review & novelty audit | Prior work in biophysics, MD datasets, and generative MD | [docs/literature/README.md](docs/literature/README.md) |
