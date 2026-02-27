# QTS OANDA XAU_USD — Embargo A/B/C (60d & 30d, proxy)

Method: proxy from closed roundtrips (`qts-oanda-minute-forensics-longonly-plusfvg-2026-02-26.json`).

| Case | Window | Closed RT | Wins | Ann.Return | Max DD | PF | Worst trade | Final capital |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_control | 60d_15m_proxy | 54 | 42 | 0.88% | -1.42% | 1.08 | -550.52 | 100144.42 |
| B_hard_embargo | 60d_15m_proxy | 49 | 42 | 7.08% | -0.60% | 2.32 | -351.29 | 101131.49 |
| C_hard_plus_soft | 60d_15m_proxy | 49 | 42 | 8.09% | -0.30% | 3.56 | -175.64 | 101286.25 |
| A_control | 30d_15m_proxy | 37 | 26 | -9.53% | -1.44% | 0.54 | -550.52 | 99180.35 |
| B_hard_embargo | 30d_15m_proxy | 32 | 26 | 2.06% | -0.60% | 1.21 | -351.29 | 100167.42 |
| C_hard_plus_soft | 30d_15m_proxy | 32 | 26 | 4.71% | -0.30% | 1.86 | -175.64 | 100378.90 |

- A_control 60d_15m_proxy: blocked_08:45=0, blocked_pattern=0, soft_scaled=0
- B_hard_embargo 60d_15m_proxy: blocked_08:45=4, blocked_pattern=1, soft_scaled=0
- C_hard_plus_soft 60d_15m_proxy: blocked_08:45=4, blocked_pattern=1, soft_scaled=17
- A_control 30d_15m_proxy: blocked_08:45=0, blocked_pattern=0, soft_scaled=0
- B_hard_embargo 30d_15m_proxy: blocked_08:45=4, blocked_pattern=1, soft_scaled=0
- C_hard_plus_soft 30d_15m_proxy: blocked_08:45=4, blocked_pattern=1, soft_scaled=12
