# Natural Language Policy Dataset (`data/policy/`)
**Dataset Domain:** Industrial Occupational Safety & Health Standards (OSHA, MUTCD, ANSI/ISEA, UK HSE).  
**Task:** Sentence-Level Personal Protective Equipment (PPE) Mandate Classification.

---

## 1. Class Taxonomy
1. `helmet_required`: Explicit mandate for protective hard hat or industrial safety helmet (e.g. 29 CFR 1926.100).
2. `vest_required`: Explicit mandate for high-visibility reflective apparel or warning vests (e.g. MUTCD 6D.03, ANSI/ISEA 107).
3. `both_required`: Explicit co-mandate for both hard hat and high-visibility vest in the same operational zone.
4. `other_ppe`: Mandate for other protective gear (safety goggles, hearing protection, boots, gloves, fall arrest harnesses).
5. `none`: Scope definitions, administrative clauses, disclaimers, or explicit exemption clauses.

---

## 2. Split Strategy (Zero Cross-Section Leakage)
Splits are segregated strictly by **source document or standard section** to prevent boilerplate sentence leakage:
- **`train`**: OSHA 29 CFR 1926 Subpart E (Head, Foot, Eye, General), OSHA 1910.132, Commercial Site Orientations.
- **`val`**: MUTCD Part 6 Section 6D.03 (High-Visibility Apparel), OSHA 1926.106 (Water Safety).
- **`test`**: OSHA 29 CFR 1926.201 (Signaling/Flaggers), ANSI/ISEA 107-2020, UK HSE Construction Regulations.

---

## 3. Human Verification Notice
All test set sentence labels have been hand-verified against original regulatory sources. No synthetic template sentences are used in train, val, or test splits.
