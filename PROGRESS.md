# riffle — internal progress notes
_Last updated: 2026-05-13. Not for submission — engineering scratchpad._

---

## Current state

Training pipeline is on the `unsloth` branch. `dev` branch has the dataset.

| stage | status |
|---|---|
| Dataset (JSONL) | ✅ committed to git (`data/chordonomicon_*.jsonl`) |
| Base model eval | ✅ done (see numbers below) |
| SFT smoke test | run it |
| Full SFT | run it |
| GRPO smoke test | run it |
| Full GRPO | run it |
| Three-way eval | run after GRPO |
| Report + slides | not started |

---

## Empirical findings not in the planning docs

### Dataset

- **Actual throughput**: full 679k-row Chordonomicon filter ran in ~28 seconds on a Mac CPU — not the ~24 min originally estimated. Root cause: `@lru_cache` on `is_parseable` exploits the tiny chord vocabulary (~hundreds of unique symbols across 680k songs). Cache hit rate is extremely high.
- **Genre imbalance is severe**: pop/rock/country each cap at 8000; soul hits only 4776; jazz 4283; blues 214. Blues is functionally unusable as a standalone eval class. Jazz is thin. The eval grid uses jazz+pop so the comparison is still valid.
- **Raw retention is 21%**, but per-style cap is what drives the final 4.9% — not the quality filters. Most filtered rows are `wrong_genre` (459k) then `no_sections` (83k).
- **`/` tokens in model output**: Chordonomicon uses `/` as a repeat-bar symbol. The model (having seen similar notation in pretraining) emits `/` as chord tokens. The parser now strips these; they were causing false `unparseable_chord` errors and bloating section lengths.

### Prompt engineering

- **Original prompt** showed only `<intro_1> C G Am F <verse_1> ...` — two sections. The base model interpreted this as "generate one or two sections then stop." It would loop `/` forever filling `max_new_tokens` within a single section.
- **Fixed prompt** explicitly lists every tag with correct indices (`<intro_1> ... <verse_1> ... <chorus_1> ... <verse_2> ... <chorus_2> ... <outro_1>`) so the model sees the full required structure upfront. Training data was rebuilt with this new prompt format.

### Base model evaluation (Qwen3-1.7B, zero-shot, `unsloth` branch prompt)

```
mean_reward (compliance):  0.192  → but this was with gaming penalty active
mean_presence:             0.860  ← model finds most sections
mean_order:                0.899  ← sections appear in right order
gaming_rate:               80%    ← model copies same chords across all sections
pass_rate:                 16%
```

After removing the gaming penalty:
```
new effective baseline:  0.5 * 0.860 + 0.5 * 0.899 ≈ 0.88 compliance
```

Key insight: **the model understands structure but not chord diversity**. Presence/order are high. The 80% gaming rate is the primary failure mode — model fills every section with the same 4-chord loop. This is exactly what SFT (real varied progressions) + GRPO (reward doesn't penalise diversity-of-structure) should fix.

Structures without intro/outro (`verse-chorus-verse-chorus-bridge-chorus`) scored 2/5 perfect — the model consistently runs out of tokens before generating `outro`. Max_new_tokens may need bumping to 768 for eval.

### Reward function evolution

1. **Original design**: `reward = r_compliance * (1 - gaming_penalty)` where gaming_penalty = 1.0 if max pairwise Jaccard > 0.9. Binary cliff, multiplicative — no gradient, and when triggered it nuked compliance entirely.
2. **Intermediate design (compliance-only)**: `reward = 0.5 * r_presence + 0.5 * r_order`. Cleaner, but SFT saturated it immediately — GRPO smoke showed reward=1.0, reward_std=0.0, loss=0.0 across all rollouts. No gradient signal.
3. **Current design (post-SFT-saturation, 2026-05-13)**: `reward = 0.35 * presence + 0.35 * order + 0.3 * diversity`.
   - `diversity = clip((1 - mean_jaccard) / 0.4, 0, 1)` — linear ramp over the **mean** pairwise Jaccard between distinct section types (verse vs chorus vs bridge, etc.). Same-name repeats (`chorus_1`, `chorus_2`) are already merged into one set before pairing.
   - **Mean, not max**: real songs share diatonic material between tonic-area sections (verse/chorus often overlap on I-V-vi-IV). Max would flag those as "gamed" when overall diversity is healthy. Mean rewards song-wide variety, lets one similar pair slide.
   - **Calibration constant 0.4** chosen from `scripts/analyze_section_jaccard_dataset.py` re-run with mean: dataset mean-Jaccard ≈ 0.62 (mean), 0.61 (median), consistent across train/val/test. Picking 0.4 puts dataset-typical jaccard right at full diversity credit (jaccard 0.6 → 1.0, jaccard 0.8 → 0.5, jaccard 1.0 → 0.0).
   - **Dataset itself has 15% "gamed" rows** (mean_jaccard > 0.9) — single-section songs and minimalist progressions exist legitimately. Worth noting but not actionable; the reward still pushes the model toward the bulk of the distribution.
   - Compliance still dominates (0.7 weight) so the model can't trade structure for diversity. Diversity (0.3) is enough to give GRPO non-zero advantages now that SFT nails compliance.
   - Additive, not multiplicative — gaming a section no longer zeros the whole reward.

### Model selection history

- Started: `Qwen/Qwen3.5-2B` (old baseline) → `Qwen/Qwen3.5-0.5B` → `Qwen/Qwen3.5-0.8B` (user-corrected via linter)
- **Final**: `unsloth/Qwen3-1.7B` — confirmed real Qwen3 series model, well-supported by Unsloth, fits on T4 with LoRA
- The 0.8B series (Qwen3.5) has hybrid linear attention requiring `flash-linear-attention` + `causal-conv1d` for the fast path; without them it's ~5-10x slower. This was the primary cause of slow eval runs.

### Training stack switch: Unsloth

- **Why**: vLLM-backed GRPO rollout generation is ~5-10x faster than HF generate. LoRA allows 1.7B to fit on T4 with headroom.
- **LoRA config**: rank=32, alpha=64, standard attention+MLP target modules.
- **Saving strategy**: both SFT and GRPO use `save_pretrained_merged` (merged 16-bit) so `eval.py` loads checkpoints with standard HuggingFace — no Unsloth dependency at eval time.
- **Old baseline API** (`generate`, `generate_batch`, `generate_grid` + `GenerationResult`, `BatchResult`, `GridResult`) was deleted — it tested a flat-list format with heuristic key/mode/voice rewards, a different task from what SFT/GRPO train on.

### HuggingFace caching

- Model weights cached to Google Drive via `os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"` — avoids re-downloading on Colab session restarts.
- Dataset Arrow cache in `~/.cache/huggingface/datasets/` (local) — the 679k-row Chordonomicon loads in ~2s after first pull.

---

## Architectural decisions not to revisit

- **Full song → per-song JSONL rows** (not per-section): keeps context intact and avoids artificial section boundaries in training.
- **Sectional tag format** (`<verse_1> Cmaj7 G Am F ...`) not JSON/CSV — token-efficient and parseable with a single regex.
- **Compliance reward only** (presence + order), not style/voice-leading heuristics — style heuristics are brittle, hard to verify, and have noisy gradients. SFT handles style transfer via demonstrations.
- **No LoRA for eval** — merged checkpoints; eval.py is dependency-free from training stack.

---

## Known risks going into training

1. **Blues** (214 examples) will be dominated by pop/rock/country in SFT. If eval ever includes blues, expect poor results.
2. **Jazz** (4283 examples) is thin. The eval grid uses jazz+pop; jazz SFT performance may lag.
3. **Gaming rate may not drop much from SFT alone** — SFT teaches format and chord diversity from demonstrations, but doesn't explicitly reward non-gaming. That's GRPO's job via the compliance reward (which now doesn't penalise same-sounding sections — so GRPO needs chord-level diversity to emerge implicitly or we accept gaming as a known limitation).
4. **Outro missing** — base model runs out of tokens before outro. Watch for this in SFT output too; if it persists, bump `max_new_tokens` in `generate_sectional` from 512 to 768.
5. **Colab disconnect during long training** — checkpoints save every 100 steps (GRPO) / 500 steps (SFT) to Drive. Worst case: lose last partial epoch.
