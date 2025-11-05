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

```
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
```


# One Toy Example

---

# Hyperparams (chosen to make lengths clean)

```
input_length  = 12        # final encoder length (after corruption + EOS)
target_length = 7         # final decoder length (after building labels + EOS)
noise_density = 0.30      # ~30% tokens will be masked
mean_noise_span_length = 2
```

# Step 0: tokenize

Sentence (13 words → 13 “tokens”):

```
S = "Cats are cute and friendly while dogs are loyal and very playful today"

tokens (L=13):
[1] Cats
[2] are
[3] cute
[4] and
[5] friendly
[6] while
[7] dogs
[8] are
[9] loyal
[10] and
[11] very
[12] playful
[13] today
```

# Step 1: choose tokens_length so post-corruption fits input_length

Using the same arithmetic as `compute_input_and_target_lengths`:

* L = 13
* noise tokens = round(13 × 0.30) = **4**
* non-noise tokens = 13 − 4 = **9**
* noise spans = round(4 / 2) = **2**
* ⇒ **encoder length** = non-noise (9) + sentinels (2) + EOS (1) = **12** ✅
* ⇒ **decoder length** = noise (4) + sentinels (2) + EOS (1) = **7** ✅
  So L=13 matches `input_length=12`, `target_length=7` perfectly.

# Step 2: counts

```
num_noise_tokens    = 4
num_nonnoise_tokens = 9
num_noise_spans     = 2
# already within clamps, ≥1, ≤(L-1)
```

# Step 3: random segmentation (picked deterministically for this demo)

We need two noise spans totaling 4, and two non-noise spans totaling 9.

```
noise_span_lengths     = [1, 3]          # sums to 4
nonnoise_span_lengths  = [3, 6]          # sums to 9
interleaved            = [3, 1, 6, 3]    # start with non-noise, then noise, etc.
```

# Step 4: build `is_noise` mask (length = 13)

Interpreting [3,1,6,3] over indices 1..13:

* non-noise(3): idx 1–3
* noise(1):     idx 4
* non-noise(6): idx 5–10
* noise(3):     idx 11–13

Boolean `is_noise` (T = masked, F = kept):

```
idx:      1    2    3    4    5    6    7    8    9    10   11    12    13
token:   Cats  are  cute  and  friendly while dogs  are  loyal  and  very  playful today
is_noise  F    F    F    T      F       F     F     F     F     F     T      T       T
```

# Step 5a: encoder sentinel map (on masked positions)

* Sentinel at **start** of each noise span; continuations become **-1** (to delete); others 0.
* Let `E0 = <extra_id_0>` and `E1 = <extra_id_1>` (we don’t need actual IDs; just names).

Noise spans:

* Span 0 starts at idx 4 (length 1): put `E0` at 4 (no continuation here).
* Span 1 starts at idx 11 (length 3): put `E1` at 11, and `-1` at 12–13.

Encoder sentinel map:

```
idx:        1  2  3   4   5  6  7  8  9  10  11  12  13
enc_map:    0  0  0  E0   0  0  0  0  0   0  E1  -1  -1
```

# Step 5b: labels sentinel map (on non-masked positions, i.e., ~is_noise)

We place sentinels at **starts of non-noise spans**; their continuations become **-1** (to delete).
Non-noise spans:

* Span A starts at idx 1 (length 3): put `E0` at 1; `-1` at 2–3.
* Span B starts at idx 5 (length 6): put `E1` at 5; `-1` at 6–10.

Labels sentinel map:

```
idx:        1   2   3   4  5   6   7   8   9   10  11  12  13
lbl_map:    E0  -1  -1   0  E1  -1  -1  -1  -1  -1   0   0   0
```

# Step 6: encoder input

Apply `enc_map` (replace where non-zero, keep token where 0), drop negatives, append EOS:

```
original:   [Cats, are, cute, and, friendly, while, dogs, are, loyal, and, very, playful, today]
enc_map:    [  0 ,  0 ,  0 , E0,    0     ,   0 ,   0 ,  0 ,   0  ,  0 ,  E1 ,  -1    ,  -1 ]

enc_tmp = WHERE(enc_map != 0, enc_map, original)
         = [Cats, are, cute,  E0, friendly, while, dogs, are, loyal, and,  E1,  -1,   -1]

enc_fused = DROP_NEGATIVES(enc_tmp)
         = [Cats, are, cute, E0, friendly, while, dogs, are, loyal, and, E1]

encoder_input = APPEND_EOS(enc_fused)
              = [Cats, are, cute, E0, friendly, while, dogs, are, loyal, and, E1, </s>]

len(encoder_input) = 12  ✅ matches input_length
```

# Step 7: decoder labels

Apply `lbl_map`, drop negatives, append EOS.
(Decoder labels should be: `<E0> (masked span 0 tokens) <E1> (masked span 1 tokens) </s>`.
Here the masked spans were:

* span 0 (idx 4): `["and"]`
* span 1 (idx 11–13): `["very","playful","today"]`)

Using the sentinel map approach:

```
lbl_tmp = WHERE(lbl_map != 0, lbl_map, original)
        = [ E0,  -1,  -1,  and,  E1,  -1,  -1,  -1,  -1,  -1, very, playful, today]

labels_fused = DROP_NEGATIVES(lbl_tmp)
             = [ E0, and, E1, very, playful, today ]

labels = APPEND_EOS(labels_fused)
       = [ E0, and, E1, very, playful, today, </s> ]

len(labels) = 7  ✅ matches target_length
```

# Step 8: decoder_input_ids (teacher forcing)

Right-shift `labels` and prefix a decoder start token (denote `<dstart>`):

```
decoder_input_ids = [<dstart>] + labels[:-1]
                  = [<dstart>, E0, and, E1, very, playful, today]
```

---

## Final Outputs (this run)

**Encoder input (length 12):**

```
[Cats, are, cute, E0, friendly, while, dogs, are, loyal, and, E1, </s>]
```

**Decoder labels (length 7):**

```
[E0, and, E1, very, playful, today, </s>]
```

**Decoder input ids (length 7):**

```
[<dstart>, E0, and, E1, very, playful, today]
```

---

```





---

# Example:

* **How many tokens will be masked** → `num_noise_tokens`
* **How many tokens will be kept** → `num_nonnoise_tokens`
* **How many masked chunks (spans)** we want → `num_noise_spans`
* We must split both the masked and unmasked parts into spans.

But instead of masking random single tokens, **T5 masks continuous chunks (spans)**.

So we need to decide:

> **How many tokens are in each span**.

This is what `noise_span_lengths` and `nonnoise_span_lengths` represent.

---

# ✅ Example 1 (The One You Already Saw)

We determined:

```
num_noise_tokens = 4
num_nonnoise_tokens = 9
num_noise_spans = 2
```

### Step 1: Split masked tokens into spans:

```
noise_span_lengths = [1, 3]     # length of masked span 1 = 1 token, span 2 = 3 tokens
```

### Step 2: Split unmasked tokens into spans:

```
nonnoise_span_lengths = [3, 6]  # keep 3 tokens, then later keep 6 tokens
```

### Step 3: Interleave them ALWAYS like:

```
(non-noise span), (noise span), (non-noise span), (noise span), ...
```

So:

```
interleaved = [3, 1, 6, 3]
```

This means:

| Span | Type        | Span Length |
| ---- | ----------- | ----------- |
| 1    | Keep tokens | 3 tokens    |
| 2    | Mask tokens | 1 token     |
| 3    | Keep tokens | 6 tokens    |
| 4    | Mask tokens | 3 tokens    |

---

# ✅ Example 2 (Different Random Split)

Assume:

```
num_noise_tokens = 5
num_nonnoise_tokens = 8
num_noise_spans = 2
```

Possible splits:

```
noise_span_lengths = [2, 3]      # 2 masked tokens, then 3 masked tokens
nonnoise_span_lengths = [4, 4]   # 4 kept tokens, then another 4 kept tokens
```

Interleave:

```
interleaved = [4, 2, 4, 3]
```

Meaning:

| Span | Action | Length   | Explanation                             |
| ---- | ------ | -------- | --------------------------------------- |
| 1    | Keep   | 4 tokens | Leave first 4 tokens untouched          |
| 2    | Mask   | 2 tokens | Replace this span with a sentinel       |
| 3    | Keep   | 4 tokens | Keep next 4 tokens untouched            |
| 4    | Mask   | 3 tokens | Replace this span with another sentinel |

---

# ✅ Example 3 (Very Different Shapes)

Suppose:

```
num_noise_tokens = 6
num_nonnoise_tokens = 10
num_noise_spans = 3   # More spans now
```

Possible random splits:

```
noise_span_lengths = [1, 3, 2]       # total 6
nonnoise_span_lengths = [5, 2, 3]    # total 10
```

Interleave ALWAYS pattern:

```
interleaved = [5, 1, 2, 3, 3, 2]
```

Mapping:

| Span | Type | Length | Meaning             |
| ---- | ---- | ------ | ------------------- |
| #1   | Keep | 5      | Keep first 5 tokens |
| #2   | Mask | 1      | Mask 1 token        |
| #3   | Keep | 2      | Keep 2 tokens       |
| #4   | Mask | 3      | Mask 3 tokens       |
| #5   | Keep | 3      | Keep 3 tokens       |
| #6   | Mask | 2      | Mask final 2 tokens |

---

# 💡 Why Interleaving Matters

T5’s masking pattern *always* alternates segments:

```
KEEP → MASK → KEEP → MASK → KEEP → MASK → ...
```

The masking does **not** start with masked tokens — it **always** starts with unmasked tokens.
This ensures the input is mostly readable while still hiding chunks.

---



```
Tokens:   A B C D E F G H I J K L M N O P ...

Keep (3): A B C
Mask (1): D
Keep (6): E F G H I J
Mask (3): K L M
```

Produces `is_noise` mask:

```
[ F, F, F, T, F, F, F, F, F, F, T, T, T ]
```






