# T55 control analysis — paired, continuous primary
Results directory `results_pub`, tau = 5 %.  All conditions present.
## m0 vs base (60 paired designs)
- **primary** Wilcoxon signed-rank p = 0.1756 (Holm over the two primaries 0.1756; over all 10 control tests 1.0000); Hodges–Lehmann shift **+0.217 pp** [-0.092, +0.576]; median +0.135 pp [-0.069, +0.443] (bootstrap); mean +0.065 pp; negative/positive 27/33, sign test p = 0.5190; range [-8.42, +3.96]
- secondary pretenders 6 → 4; discordant 4/2, exact McNemar p = 0.6875
- at a base rate of 10.0 % and n = 60, the binary comparison reaches 80 % power against a control rate of max concordance 23 %, independence 32 %, min concordance 35 %
| structure | n | median Δ base | median Δ control | median shift | HL shift [95 %] | mean shift | neg/pos | Wilcoxon p (Holm, all tests) | pretenders |
|---|---|---|---|---|---|---|---|---|---|
| B | 20 | +2.34 pp | +2.32 pp | -0.356 pp | -0.439 [-2.046, +0.458] | -0.925 pp | 13/7 | 0.2162 (1.0000) | 3 → 1 |
| A | 20 | +1.45 pp | +2.99 pp | +0.748 pp | +0.920 [+0.315, +1.675] | +0.956 pp | 3/17 | 0.0017 (0.0169) | 1 → 2 |
| C | 20 | +0.21 pp | +0.35 pp | -0.033 pp | +0.103 [-0.153, +0.457] | +0.163 pp | 11/9 | 0.6215 (1.0000) | 2 → 1 |
## feas vs base (60 paired designs)
- **primary** Wilcoxon signed-rank p = 0.0071 (Holm over the two primaries 0.0141; over all 10 control tests 0.0635); Hodges–Lehmann shift **-0.181 pp** [-0.585, -0.003]; median -0.001 pp [-0.210, -0.000] (bootstrap); mean -0.793 pp; negative/positive 40/20, sign test p = 0.0135; range [-10.66, +4.61]
- secondary pretenders 6 → 3; discordant 4/1, exact McNemar p = 0.3750
- at a base rate of 10.0 % and n = 60, the binary comparison reaches 80 % power against a control rate of max concordance 23 %, independence 33 %, min concordance 35 %
| structure | n | median Δ base | median Δ control | median shift | HL shift [95 %] | mean shift | neg/pos | Wilcoxon p (Holm, all tests) | pretenders |
|---|---|---|---|---|---|---|---|---|---|
| B | 20 | +2.34 pp | +1.19 pp | -0.545 pp | -1.394 [-2.909, -0.156] | -1.711 pp | 15/5 | 0.0121 (0.0966) | 3 → 1 |
| A | 20 | +1.45 pp | +1.08 pp | -0.155 pp | -0.223 [-0.873, +0.101] | -0.706 pp | 14/6 | 0.1536 (1.0000) | 1 → 0 |
| C | 20 | +0.21 pp | +0.21 pp | -0.000 pp | +0.000 [-0.001, +0.002] | +0.038 pp | 11/9 | 0.8408 (1.0000) | 2 → 2 |
