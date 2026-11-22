# Conformal Prediction Under Distribution Shift

This project studies how split conformal prediction behaves when the test distribution differs from the distribution used for calibration.

I became interested in this problem after reading work from the NeurIPS literature on conformal prediction under distribution shift, including research on covariate shift and adaptive conformal inference. I then built my own experimental framework to reproduce and explore these ideas under controlled synthetic settings.

## Research Question

How does distribution shift affect the empirical coverage of conformal prediction intervals, and what simple methods can restore coverage when the exchangeability assumption no longer holds?

## Method

I implemented a regression based split conformal prediction pipeline in Python.

The experiments study three types of distribution shift:

Covariate shift

Noise shift

Subgroup and mixture shift

For each setting, I measure:

Empirical prediction interval coverage

Average interval width

How coverage changes as the shift becomes stronger

I also evaluate several possible responses to distribution shift:

Recalibration using data from the shifted distribution

Group specific calibration

Increasing calibration set size

Changing the underlying regression model

Weighted conformal prediction, using both the exact likelihood ratio and one estimated from unlabeled data with logistic regression

Locally adaptive intervals, using normalized (scaled residual) scores and conformalized quantile regression, on a variant of the data where noise grows with |x3|

Online prediction on drifting data streams, using adaptive conformal inference, a rolling calibration window, and the two combined

Every setting is repeated over 100 independent calibration and test draws, and results report mean coverage together with its trial to trial standard deviation. Within a trial, all methods are scored on the same test set.

## Main Findings

Under no distribution shift, the conformal intervals achieve approximately their nominal 90 percent coverage.

Under distribution shift, coverage can deteriorate substantially. For example, under strong covariate shift, coverage falls far below its target while interval width remains fixed.

Recalibration can restore coverage close to 90 percent, but it may require much wider intervals. This reveals an important tradeoff between statistical coverage and practical usefulness.

Group specific calibration performs particularly well when subgroup structure is known, restoring coverage without the extreme interval width growth seen under generic recalibration.

Weighted conformal prediction keeps coverage at or above target without any new labels, and estimated weights perform almost exactly like the true ones. However, as the shift grows, a growing share of test points get infinite intervals, about 16 percent at a covariate shift of 1.0 and nearly all at 2.5 and above.

When noise depends on the inputs, normalized scores give close to 90 percent coverage in every region of the input space and hold coverage as inputs shift toward the noisy region, while standard intervals fall to 67 percent. Conformalized quantile regression helps less. Adaptivity cannot help when difficulty is not visible in the inputs, as with the latent subgroups.

On data streams, adaptive conformal inference keeps long run coverage near 90 percent in every scenario, where standard intervals fall to 72 to 78 percent. Combining it with a rolling calibration window gives the best results, with no infinite intervals.

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python Main.py   # about 3 minutes; writes all figures and CSVs to results/
```

## Tools

Python

scikit learn

NumPy

pandas

Matplotlib

## Repository Structure

`Main.py` main entry point

`conformal.py` conformal prediction methods (standard, group specific, weighted, normalized, conformalized quantile regression)

`online.py` online methods for data streams (adaptive conformal inference, rolling window)

`data_generation.py` synthetic data and distribution shift generation

`experiments.py` experimental pipeline

`plots.py` visualization utilities

`results/` generated experimental results

`report.pdf` full written report

## AI Tool Usage

AI tools, including Claude, were used during development for coding assistance, debugging, and discussing possible implementation improvements.

I independently implemented and tested the experimental framework, selected the experimental comparisons, ran the analyses, verified the outputs, and interpreted the results. AI generated suggestions and code were reviewed and tested before being incorporated.

The underlying research direction was inspired by prior work in the conformal prediction literature rather than being an original research question developed from scratch.

## Related Work

This project was motivated in part by work such as:

Tibshirani et al., *Conformal Prediction Under Covariate Shift*, NeurIPS 2019

Gibbs and Candès, *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS 2021

## Author

Thinh Truong Nguyen

University of California, Berkeley

Mathematics and Statistics
