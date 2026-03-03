# QA Protocol

- Randomly sample 5% of labeled data for re-check.
- Verify split leakage by session_id (no same session across splits).
- Track label distribution (avoid extreme imbalance).
- Keep baseline metrics for comparison and debugging.