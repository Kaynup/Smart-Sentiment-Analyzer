# 🧬 Real-World Dataset for Text Classification with Sarcasm Awareness

### 🧾 Author: **Punyak**

⋆｡°✩──────────────✩°｡⋆

## ✦ Aim

To create and label a **real-world dataset** that enhances both **vocabulary coverage** and **sarcasm detection** in **pretrained** or **custom neural models** for **multi-class** and **multi-label** text classification.

Supports both **one-vs-one** and **one-vs-many** classification setups, with diverse tones, slang usage, and contextual intent types.

---

## 🗓️ Dataset Overview

> **Total Samples**: ` > 2,000,000+`  
> **Source**: Reddit (multiple subreddits)  
> **Collection Period**: `2025-07-09` → `2025-07-14`

Content reflects natural, user-generated posts and comments, rich in sarcasm, humor, context ambiguity, and informal structures.

---

## ⚓ Meta-tag Schema

Each sample contains the following structured fields:

| 🏷️ **Field**         |  **Description / Values** |
|----------------------|-----------------------------|
| `text`               | Raw text (Reddit post or comment) |
| `text_label`         | Multi-label sentiment: `[1, 0]` = *Positive + Neutral* <br> Values: `Positive:1`, `Negative:-1`, `Neutral:0` |
| `text_tone`          | Multi-label tone scores: `[-3, 2]` = *Sarcastic + Mildly Sincere* <br> Range: `Sarcastic: -1 to -5`, `Sincere: 1 to 5`, `Neutral: 0` |
| `contains_slang`     | `'True'` / `'False'` |
| `nsfw_slang`         | `'True'` / `'False'` |
| `causality`          | `'True'` / `'False'` |
| `contains_emojies`   | `'True'` / `'False'` |
| `context_fields`     | Multi-label: e.g. `["Advice", "Joke-Meme"]` <br> Options: `Advice`, `Review`, `Comment`, `Compliment`, `Joke-Meme`, `Question`, `Reply` |
| `usability`          | Confidence/quality score for the entire sample: <br> `Great:3`, `Good:2`, `Low:1`, `Bad:0` |

---