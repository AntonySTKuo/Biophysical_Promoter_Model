# Biophysical Promoter Model

This project aims to quantify the sequence contributions of the -35 and -10 elements in bacterial promoters within a thermodynamic framework. Detailed model formulations, interpretations, and methodologies are described in the accompanying article.

---

## Experimental Datasets

All experimental datasets are available in the `tables/` folder, which contains two types of promoter libraries differing in their regulatory mechanisms:

* **Constitutive promoter libraries** <br>
  Constitutive promoters drive transcription without regulatory control. <br>
  * PL<sub>C17</sub>
  * PL<sub>C16</sub>

* **Transcription factor (TF)-regulated promoter libraries** <br>
  These promoters require specific TFs to modulate transcriptional activity in response to chemical inducers.
  Each library was characterized under two conditions:
  **Basal** — absence of the corresponding inducer;
  **Induced** — presence of a saturating concentration of the inducer controlling TF activity. <br>
  * PL<sub>TetR</sub>
  * PL<sub>LuxR</sub>
  * PL<sub>CueR</sub>

Here, TetR, LuxR, and CueR are the associated TFs that regulate promoter activity.

---

## Model Fitting

### Constitutive Promoter Libraries

For constitutive promoters, we applied the **PAS model** ([Promoter_Architecture_Scanner](https://github.com/AntonySTKuo/Promoter_Architecture_Scanner)) to quantify the sequence contributions of the -35 and -10 elements.
Implementation: `scripts/ModelFit_PAS.py`

### TF-Regulated Promoter Libraries

For TF-regulated promoters, we employed two fitting approaches:

1. **Independent fitting**
   Models were fitted independently to the basal and induced datasets, resulting in two distinct parameter sets.
   Implementation: `scripts/ModelFit_TF-separate.py`

3. **Shared-parameter fitting**
   Models were fitted to the basal and induced datasets, constrained to share a single common parameter set.
   Implementation: `scripts/ModelFit_TF-shared.py`

All fitted models are available in the `models/` folder.

---

## Visualization

Model results can be visualized using the provided Jupyter notebooks in `scripts/`, which generate the figures available in the `figures/` folder.

---

## Environment

If you want to run the modeling project on your own, please set up the Python environment using conda to create it from the configuration file `environment.yml`:

```
# bash
conda env create -f environment.yml
```

