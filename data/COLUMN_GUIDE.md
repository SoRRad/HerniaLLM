# Data Column Guide

## cases.csv — Field Definitions

Fill this file with your de-identified retrospective hernia cases.
Open `cases_template.csv` in Excel, populate it, and save as `cases.csv`.

---

### Identifiers

| Column | Description | Example |
|--------|-------------|---------|
| `case_id` | Unique case identifier | `CASE_001` |
| `hernia_type` | Actual hernia type (from surgical record) | `incisional`, `umbilical`, `epigastric`, `paraumbilical`, `spigelian` |
| `case_complexity` | Complexity tier | `simple`, `moderate`, `complex` |

---

### Presentation Features (fed to patient simulator)

| Column | Description | Example |
|--------|-------------|---------|
| `chief_complaint` | Patient's own words | `bulge in my belly button` |
| `duration_symptoms` | How long symptoms present | `3 weeks`, `6 months` |
| `location` | Anatomical location | `umbilical`, `midline scar`, `right lower quadrant` |
| `size_description` | Size as patient would describe | `golf ball sized`, `small grape` |
| `reducible` | Can it be pushed back? | `yes`, `no`, `sometimes` |
| `pain_character` | Type of pain | `dull ache`, `sharp`, `burning`, `none` |
| `pain_severity` | Scale 0–10 | `4` |
| `aggravating_factors` | What makes it worse | `lifting, coughing`, `none` |
| `relieving_factors` | What makes it better | `lying down`, `none` |
| `nausea_vomiting` | Present or not | `yes`, `no` |
| `bowel_changes` | Any bowel changes | `constipation`, `none` |
| `fever` | Fever present | `yes`, `no` |

---

### Clinical Context

| Column | Description | Example |
|--------|-------------|---------|
| `prior_hernia_repair` | Previous hernia surgery | `yes`, `no` |
| `prior_abdominal_surgery` | Other abdominal surgery | `appendectomy 2010`, `none` |
| `anticoagulation` | On blood thinners | `warfarin`, `none` |
| `pregnancy` | Current pregnancy | `yes — 28 weeks`, `no` |
| `bmi_category` | BMI category | `normal`, `overweight`, `obese` |
| `relevant_comorbidities` | Key comorbidities | `diabetes, COPD`, `none` |

---

### Red Flags (pre-specify before running pipeline)

| Column | Values | Notes |
|--------|--------|-------|
| `red_flag_pain` | `yes` / `no` | Severe or sudden onset pain |
| `red_flag_skin_changes` | `yes` / `no` | Erythema, discolouration over hernia |
| `red_flag_obstructive_symptoms` | `yes` / `no` | Vomiting, absolute constipation |
| `red_flag_irreducible` | `yes` / `no` | Cannot be reduced manually |
| `red_flag_systemically_unwell` | `yes` / `no` | Fever, tachycardia, sepsis signs |

---

## ground_truth.csv — Field Definitions

This file contains the correct answers, filled in by your clinical adjudicator
**before** any LLM sees the cases.

| Column | Description | Example |
|--------|-------------|---------|
| `case_id` | Must match cases.csv | `CASE_001` |
| `ground_truth_diagnosis` | Correct diagnosis | `incisional hernia` |
| `ground_truth_hernia_type` | Specific type | `incisional — midline` |
| `correct_imaging` | Appropriate test | `CT abdomen`, `ultrasound`, `none required` |
| `correct_referral` | Who to refer to | `general surgery`, `hernia specialist` |
| `referral_urgency` | Urgency level | `routine`, `urgent`, `emergency` |
| `red_flags_present` | Which flags present | `red_flag_pain, red_flag_irreducible` |
| `critical_red_flags` | Any life-threatening flags | `yes — incarceration suspected`, `no` |
| `correct_management_summary` | Free text summary | `Urgent surgical referral, nil imaging needed` |
| `adjudicator` | Who filled this in | `Dr Smith` |
| `adjudication_date` | When adjudicated | `2024-10-01` |
