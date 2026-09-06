# Papers → Code

Research papers in AI/ML, reimplemented from scratch and checked against their own reported results.

## What this is

Each paper here was read closely enough to rebuild end to end. The implementation is then evaluated the same way the paper evaluates itself, and the result is reported as it actually came out, including the cases where it doesn't match what the paper claims. A discrepancy is treated as a finding worth explaining, not a failure to hide.

This started with one paper, handed to me by a professor at an AI summit, which led to a great discussion. It has since grown into a standing habit: it's a more demanding way to read a paper than working through it on the page, and a better test of whether I actually understood it.

## How it's organized

Every paper lives in its own numbered folder:

```
01-paper-slug/
├── README.md        the paper's core claim, the translation decisions, and where the results landed
├── paper/            the original paper (PDF)
├── src/              the implementation, as importable modules
├── notebooks/        a walkthrough notebook tying src/ together 
├── results/          the metrics and figures this code actually produced
└── requirements.txt
```

The rule that matters most: each implementation is judged against the paper's actual claim, not against "does something run." If a result doesn't match what's reported, the README says why (different dataset, fewer resources, a design choice the paper left open... etc.).

## Papers implemented so far

| # | Paper | Area | Status |
|---|-------|------|--------|
| 01 | [RN-SMOTE: Reduced Noise SMOTE based on DBSCAN for Enhancing Imbalanced Data Classification](./01-rn-smote) — Arafa et al., *J. King Saud University – CIS*, 2022 | Imbalanced classification / noise reduction | Done |

More will be added as they're completed — the goal is breadth across classical ML, multi-modal frameworks, and LLMs, with a lean toward whatever's feeding my own research work at the time.

## About

- Contact: [ahmed.essam1418@gmail.com]

## License

MIT - Fork it, break it, tell me what's wrong with it.

