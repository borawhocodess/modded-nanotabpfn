# INTEGRITY NOTES — what a reviewer should check, and what we got wrong

## Verified before any number was produced
- Upstream pin `687cfd9b5777` is current HEAD; our baseline file is byte-identical (sha256 `a63654348e4c3e6d`).
- Base arm applies NO patch (`_prep` guards on `if patch_text.strip():`), so it is upstream byte-for-byte.
- `--seed` is upstream argparse with Config default 11; passing `--seed 11` is a no-op. We added nothing.
- Both patches apply cleanly to the pinned file and reproduce our stored variants exactly.

## Errors we made and caught ourselves
1. **Compile-cache asymmetry (affected correctness).** Shared warm cache created once, never merged ->
   warmed only the first-running arm. fast2 paid a cold compile every round (epoch-1 35.05 s vs 11.03 s)
   and measured *slower overall* while being 14% faster per epoch. Fixed by warming every arm and
   merging; confirm-pass epoch-1 = base 9.82 s / fast2 9.10 s. **This bug pointed toward the null**, i.e.
   it would have produced a believable wrong negative.
2. **Modal Volume concurrent-append data loss (cost money, not correctness).** Seven containers appended
   to one volume path; Volume commits are whole-file, so all but the last writer were lost. 21 rounds ran,
   3 survived, ~$20 wasted. Fixed by returning records through the function result AND writing unique
   per-chunk filenames. We discarded the inconsistent partial pulls rather than splicing them into an
   accumulation that never existed.
3. **Regex over-match.** `e_t:([0-9.]+)s` also matches the running-mean field. Fixed with a negative
   lookbehind; symptom was alternating real/mean values.

## Claims we deliberately do NOT make
- No reproduction of their 0.92 min. Our L40S absolute base median is 1.032 min.
- The single fast2 run at 0.83 min (under 0.92) is NOT headlined: on that container the pinned baseline
  ran 0.99 min, so it is one container's absolute, not an apples-to-apples record.
- No L40S number for fast4 — we never ran it there.
- 14/15 wins on L40S, not 15/15. The loss is reported in full with its cause.
- The ~0.76-0.79 min figure for their hardware is a **prediction** for them to re-time, not a measurement.

## Spend
Approved $40. Actual ~$38: staging ~$0, diagnostics ~$6, lost 21-round run ~$20, final 15-round run ~$12.
