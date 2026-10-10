# RCPS with fractional pixel-miscoverage loss

This guide describes the RCPS experiment in [lodopab_small_subset_rcps.ipynb](lodopab_small_subset_rcps.ipynb). It now follows the loss in the risk-controlling prediction set definition: **the loss for one image is the fraction of its pixels missed**. RCPS controls the expected loss on a fresh test image with a high-probability statement over calibration samples.

## Dataset and prediction model

The notebook uses 700 images to train a CNN reconstruction function f. For each image, X is the degraded input and Y is the clean target. The CNN predicts Y_hat = f(X). It uses 300 further images for calibration and keeps 100 images untouched for test evaluation.

The target images come from a small LoDoPaB-derived subset mirrored by GLIMPSE. Inputs are simulated from 50-view sinograms with Gaussian noise and filtered back projection; these are not the official LoDoPaB measurement files. The subset does not document patient IDs.

## Pixel prediction sets

For radius λ ≥ 0, the set for pixel (m,n) of input image X_i is the interval

$$
T(X_i)_{mn} = [ max(0, Y_hat_{i,mn} − λ), min(1, Y_hat_{i,mn} + λ) ].
$$

A target pixel is covered when Y_{i,mn} ∈ T(X_i)_{mn}. The image prediction set is the collection of these intervals over all M × N pixels. Pixels inside an image may be dependent; this method does not assume pixel independence.

## The RCPS loss in this notebook

For calibration or test image i, define

$$
L_i(T) = 1 − (number of covered pixels in image i)/(M × N)
       = (number of missed pixels in image i)/(M × N).
$$

Thus L_i is in [0,1]. An image with 10% missed pixels has loss 0.10; it is not automatically counted as a full miss. This differs from the stricter whole-image indicator loss 1{at least one pixel missed}.

For a fresh example (X,Y) drawn from the target population P, the population risk is

$$
R(T) = E_{(X,Y)∼P}[L(T(X),Y)].
$$

It is the expected fraction of pixels missed, averaging over new images. It is not the probability of any miss in an image, and it does not promise that every individual image has low loss.

## Calibration procedure

The 300 calibration images are split randomly into two independent portions:

1. **Radius-selection set:** 150 images choose the smallest radius on a fixed grid whose observed mean loss leaves room for the confidence-bound margin.
2. **Risk-check set:** 150 different images estimate the selected radius's mean loss and calculate the upper bound. The radius selection does not use this set.

For a selected radius λ_hat, the risk-check mean loss is

$$
L_bar_risk = (1/n_risk) ∑_{i=1}^{n_risk} L_i(λ_hat).
$$

With confidence failure probability δ, the notebook uses the one-sided Hoeffding upper bound

$$
U(λ_hat) = min(1, L_bar_risk + sqrt(log(1/δ)/(2 n_risk))).
$$

The reported formal target is

$$
P_{C_risk}( R(T) > α ) ≤ δ.
$$

The inner expectation in R(T) is over a new test pair (X,Y). The outer probability is over repeated random risk-check calibration samples, conditional on the trained model and the radius selected on the separate optimization set. This is the distinction in the quoted definition: α limits the population expected loss, while δ limits the probability that calibration fails to establish that limit.

The notebook calls a radius **certified for α** only when

$$
U(λ_hat) ≤ α.
$$

For example, if α = 0.10, δ = 0.10, and U = 0.08, the calibration result supports the claim that expected pixel miscoverage on a fresh image is at most 10%, with 90% confidence. If U > α, the result is not certified; this does not prove the true risk is above α.

## α and δ in plain language

- **α (risk target):** the largest expected fraction of missed pixels per fresh image that is allowed. For α = 0.10, the target expected pixel miscoverage is at most 10%.
- **δ (confidence failure level):** how often the bound may fail across repeated risk-check samples. δ = 0.10 means a 90% confidence level under the assumptions below.
- **U (computed upper bound):** the number calculated from the risk-check losses. It is neither α nor the confidence level.

This α is the RCPS risk target. It is not split conformal's α, which is a nominal miscoverage level in a different method.

## Why small α may not be certifiable here

With n_risk = 150 and δ = 0.10, the Hoeffding margin is

$$
h = sqrt(log(10)/300) ≈ 0.0876.
$$

The upper bound cannot be below this margin, even when the observed risk-check loss is zero. Therefore α = 0.01 and α = 0.05 cannot be certified with this sample size and bound. “Not certified” means the evidence is insufficient for that claim; it does not mean the actual expected loss is necessarily above α.

The table reports test pixel coverage, test whole-image coverage, and mean interval width as empirical diagnostics. Test pixel coverage is the fraction of all test pixels covered. Test whole-image coverage is the fraction of images with every pixel covered; it is useful to see, but it is **not** the loss controlled here. Neither test metric is used for selecting the radius or computing the RCPS bound.

## Assumptions and limitations

The Hoeffding bound treats the per-image fractional losses in the risk-check set as independent, identically distributed, bounded observations from the same population as future images. Pixels within an image need not be independent because they are summarized as one bounded loss. Dependence between images, such as multiple images from one patient, can violate the independent-image assumption. Patient IDs are unavailable in this subset, so patient-level separation cannot be verified.

The notebook evaluates real LoDoPaB-derived target images with simulated measurements. It is a small exploratory stress test, not an evaluation on official LoDoPaB observations or a patient-level medical validation. Each risk target in the table is assessed with δ = 0.10 individually; the four rows do not provide a simultaneous 90% guarantee for all targets together.

## Run in Google Colab

Open [lodopab_small_subset_rcps.ipynb](lodopab_small_subset_rcps.ipynb), select Runtime → Change runtime type → GPU if available, and run all cells from top to bottom. CPU execution is possible but slower.
