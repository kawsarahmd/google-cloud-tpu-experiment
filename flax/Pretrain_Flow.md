#### Main Functionality
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


#### Compact Pseudocode

#### hyperparams
input_length, target_length
noise_density, mean_noise_span_length

#### step 0: tokenize
tokens = tokenize(sentence)
L = len(tokens)

#### step 1: (precomputed) choose tokens_length so that post-corruption lengths fit
#### (using compute_input_and_target_lengths)

#### step 2: counts
num_noise_tokens     = round(L * noise_density)
num_nonnoise_tokens  = L - num_noise_tokens
num_noise_spans      = round(min(num_noise_tokens, num_nonnoise_tokens) / mean_noise_span_length)
num_noise_tokens     = clamp(num_noise_tokens, 1, L-1)
num_noise_spans      = max(num_noise_spans, 1)

#### step 3: random segmentation for span lengths
noise_span_lengths    = RANDOM_SEGMENT(num_noise_tokens,    num_noise_spans)
nonnoise_span_lengths = RANDOM_SEGMENT(num_nonnoise_tokens, num_noise_spans)
interleaved = INTERLEAVE(nonnoise_span_lengths, noise_span_lengths)  #### starts with non-noise

#### step 4: build is_noise mask from interleaved spans
is_noise = BUILD_MASK_FROM_SPANS(interleaved, start_with_non_noise=True)  #### length L

#### step 5a: encoder sentinel map (on masked positions)
enc_sentinel_map = CREATE_SENTINEL_IDS(is_noise)        #### E_k at span starts, -1 for span continuations, 0 else

#### step 5b: labels sentinel map (on non-masked positions)
lbl_sentinel_map = CREATE_SENTINEL_IDS(~is_noise)

#### step 6: encoder input
enc_tmp = WHERE(enc_sentinel_map != 0, enc_sentinel_map, tokens)   #### replace starts with E_k; continuations are -1
enc_fused = DROP_NEGATIVES(enc_tmp)                                #### remove -1
encoder_input = APPEND_EOS(enc_fused)                              #### ... + </s>
ASSERT(len(encoder_input) == input_length)

#### step 7: decoder labels
lbl_tmp = WHERE(lbl_sentinel_map != 0, lbl_sentinel_map, tokens)
labels_fused = DROP_NEGATIVES(lbl_tmp)
labels = APPEND_EOS(labels_fused)
ASSERT(len(labels) == target_length)

#### step 8: decoder_input_ids (teacher forcing)
decoder_input_ids = [<dec_start>] + labels[:-1]

