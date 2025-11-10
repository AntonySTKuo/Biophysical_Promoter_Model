# Biophysical Promoter Model

This modeling project aims to quantify the sequence contributions of the −35 and −10 elements in bacterial promoters using a thermodynamic framework. Detailed model framework, interpretation, and methodologies are described in the accompanying article.

---

## Experimental Overview

In this study, we experimentally characterized:

- **Constitutive promoter libraries**: PL<sub>C17</sub> and PL<sub>C16</sub>  
- **Transcription factor (TF)-regulated promoter libraries**: PL<sub>TetR</sub>, PL<sub>LuxR</sub>, and PL<sub>CueR</sub>  

Here, TetR, LuxR, and CueR are transcription factors that regulate promoter activity.  
The TF-regulated libraries were characterized under two conditions:
- **Basal** (absence of inducer)
- **Induced** (presence of the corresponding saturating inducer controlling TF activity)

All processed datasets are available in the folder: `tables/`

---

## Modeling Approaches

### Constitutive Promoter Libraries
For constitutive promoters, we adopted our previously developed **PAS model** ([Promoter_Architecture_Scanner](https://github.com/AntonySTKuo/Promoter_Architecture_Scanner)) to quantify the sequence contributions of the −35 and −10 elements.  
Implementation: `scripts/ModelFit_PAS.py`

### TF-Regulated Promoter Libraries
For TF-regulated promoters, we employed two fitting approaches:

1. **Independent fitting**  
   Models were fitted separately to the basal and induced datasets, resulting in two distinct parameter sets.  
   Implementation: `scripts/ModelFit_TF-separate.py`

2. **Shared-parameter fitting**  
   Models were jointly fitted to the basal and induced datasets but constrained to share a common parameter set.  
   Implementation: `scripts/ModelFit_TF-shared.py`

---

## Output Files

- **Model results:** `models/`  
- **Visualization scripts:** Jupyter notebooks in `scripts/`  
- **Generated figures:** `figures/`

