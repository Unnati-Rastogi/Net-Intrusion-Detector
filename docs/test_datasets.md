# Test datasets for NetGuard

The files in `data/test_datasets/` are real, sampled records from the Friday
capture in the corrected CIC-IDS2017 release. Friday was held out from model
selection. They keep the original CIC flow columns and `Label` values, so each
file can be uploaded directly through **Scan dataset**.

| File | Situation | What to look for |
|---|---|---|
| `01_normal_traffic.csv` | Normal traffic baseline | The detector should return a no-significant-activity conclusion. A few individual low-score review flags may still appear. |
| `02_ddos_heavy.csv` | Major DDoS incident | A clear incident with 90% DDoS traffic. The alert count should be high. |
| `03_low_prevalence_attack.csv` | Rare DDoS among normal traffic | Only 1% of flows are attacks. This tests whether the detector can find a small suspicious subset without treating the whole dataset as malicious. |
| `04_portscan_reconnaissance.csv` | Port-scan reconnaissance | A balanced mix of normal and Portscan flows. Use the explanation view to inspect the highest-scoring flows. |
| `05_rare_botnet_challenge.csv` | Rare Botnet activity | A real, uncommon attack family mixed with normal flows. Treat this as a difficult-case demonstration, not proof that any particular traditional NIDS would fail. |

All five files include labels for demonstration, so NetGuard will show accuracy,
precision, and recall after scanning. The detector itself does not need labels;
a real upload may omit the `Label` column.

Do not use the metrics from these files as a new independent headline claim:
they are all sampled from the same held-out Friday capture that produced the
existing CIC validation result.
