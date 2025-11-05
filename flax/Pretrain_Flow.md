1. Parse CLI/JSON Arguments
2. Telemetry & Output Dir Check
3. Setup Logging & Seed
4. Setup HF Hub Repo (Optional)
5. Load Dataset (Hub or Local)
6. Load Tokenizer & Config
7. Tokenize Text & Compute Span Mask Lengths
8. Group Text into Fixed Windows
9. Initialize Model (Pretrained or Scratch)
10. Create Data Collator (Span Masking)
11. Setup Training State & Optimizer
12. Define train_step & eval_step (pmap)
13. Run Training Loop (Logging / Eval / Save)
14. Final Evaluation & Save Metrics
