# Tutorial: RCPS on the small LoDoPaB-derived dataset

This guide explains the **RCPS (risk-controlling prediction set)** experiment in [`lodopab_small_subset_rcps.ipynb`](lodopab_small_subset_rcps.ipynb).

There are now two standalone notebooks using the same small LoDoPaB-derived data and reconstruction setup: [`lodopab_small_subset_rcps.ipynb`](lodopab_small_subset_rcps.ipynb) for RCPS and [`lodopab_small_subset_split_conformal.ipynb`](lodopab_small_subset_split_conformal.ipynb) for split conformal. This guide explains the RCPS notebook. The notebook trains an image reconstruction model, chooses an interval radius using calibration images, then checks whether it can certify a requested image-level miss risk.

## In one sentence

For a new CT image, the model predicts a reconstruction and places a uniform interval around every pixel; RCPS chooses the interval radius so that, with a stated confidence level, the probability that **any pixel is missed** is at most a target risk `epsilon`.

## What data does it use?

The notebook downloads a small LoDoPaB-derived image subset from a Hugging Face copy associated with GLIMPSE. It uses 700 images for fitting the reconstruction model, 300 for RCPS calibration, and 100 untouched images for final evaluation.

The clean target images come from the LoDoPaB-derived subset. For each target, the notebook **simulates** a new 50-view sinogram, adds Gaussian noise, and reconstructs an FBP input. Therefore, the experiment uses real CT image targets, but its inputs are simulated; these are not the official LoDoPaB low-dose measurements. The copy also does not document patient IDs, so the notebook cannot enforce or verify patient-level separation.

## The workflow

### 1. Make paired input and target images

Each clean image is `Y`. The notebook computes simulated sparse-view projections, adds noise, and applies filtered back projection (FBP) to create the degraded input `X`:

`X = FBP(noisy 50-view projections of Y)`

### 2. Train the reconstruction model

A small convolutional neural network is trained on the 700 fitting pairs `(X, Y)` using pixel-averaged mean squared error. It learns a central reconstruction `f(X)`. Training loss measures fit to the training images; it does **not** certify uncertainty or coverage.

### 3. Define one score and one loss per image

For each of the 300 calibration images, the notebook computes the largest absolute pixel residual:

`S_i = max over pixels j of |Y_ij - f(X_i)_j|`

For a candidate radius `lambda`, the interval at pixel `j` is

`[max(0, f(X_i)_j - lambda), min(1, f(X_i)_j + lambda)]`.

The image-level loss is

`L_i(lambda) = 1{S_i > lambda}`.

It is 1 if **at least one pixel** in image `i` lies outside its interval, and 0 if every pixel is covered. This is a whole-image miss event, not an average pixel error or a fraction of missed pixels.

### 4. Split calibration into selection and risk-check portions

The 300 calibration images are randomly divided into 150 **optimization** images and 150 **risk-check** images.

- The optimization half chooses a radius from a fixed grid. The radius is selected without using the risk-check outcomes.
- The risk-check half is used once to estimate the selected radius's miss risk and compute an upper confidence bound.
- The 100 test images are kept out of both steps. They provide empirical evaluation only.

This extra split is needed here because RCPS selects a setting and then needs separate data to certify its risk. Reusing the same observations for both selection and certification would invalidate this simple bound.

### 5. Four terms to keep separate

**Image-level miss risk, `R(lambda)`:** For a fixed model and radius, this is the probability that one image drawn from the target population has at least one pixel outside its interval. It is a population quantity, not the observed test-set percentage.

**Epsilon, `epsilon`:** The largest miss risk we are willing to allow. For example, `epsilon = 0.10` means the goal is `R(lambda) <= 0.10`; in population terms, at least 90% of images should have all pixels covered. Epsilon is about the performance on images drawn from the population.

**Delta, `delta`, and confidence:** Delta is the allowed chance that the calibration procedure fails to produce an upper bound that is above the true `R(lambda)`. The confidence level is `1 - delta`. Here `delta = 0.10` means 90% confidence. This is about repeated risk-check calibration samples, not the percentage of images covered.

**RCPS upper bound:** This is the number computed from the risk-check images. If `n` is their count and `r_hat` is their observed mean image-level miss loss, the notebook computes

`UCB = min(1, r_hat + sqrt(log(1/delta) / (2*n)))`.

The upper bound is an estimate with a confidence statement attached; it is **not another name for epsilon or confidence**. The notebook says a radius is **certified for epsilon** only when `UCB <= epsilon`.

For example, if `epsilon = 0.10`, `delta = 0.10`, and the computed upper bound is `0.08`, the result is certified: with 90% confidence from the risk-check sample, the population whole-image miss risk is at most 10%. If the upper bound is `0.13`, it is not certified for epsilon 0.10. That does not prove the true risk is above 10%; it means this calibration result did not establish that it is below 10%.

The optimization step chooses a narrow radius using the separate optimization half. The Hoeffding margin can be large with only 150 risk-check images. In this notebook it is about 0.088, so the requested risks 0.01 or 0.05 cannot be certified by this bound and sample size—even if the observed risk-check miss rate is zero. That is a limitation of the available evidence, not proof that the actual miss risk is that high.

### 6. Evaluate on untouched test images

For a selected radius, the notebook reports:

- **Risk-check empirical risk:** fraction of risk-check images with at least one missed pixel.
- **Risk upper bound:** the RCPS certificate; compare it with `epsilon`.
- **Test coverage:** fraction of the 100 held-out images for which every pixel is inside the interval. This is an empirical diagnostic, not the certificate.
- **Test mean width:** average interval width over test images and pixels. Wider intervals generally improve coverage but are less informative.

It also plots a test reconstruction, interval width, and missed pixels for an example image. A mostly blank missed-pixel plot does not imply whole-image coverage: a single missed pixel makes that image a whole-image miss.

## What guarantee is being targeted?

The RCPS statement is: for a radius selected using the optimization half, the separate risk-check half gives an upper confidence bound on the probability that an image from the target population has **at least one pixel outside its interval**. A certified line means `UCB <= epsilon`. The confidence statement concerns repeated risk-check samples; the risk concerns one image drawn from the population. Neither says every image will be covered.

The simple Hoeffding calculation treats the image-level losses in the risk-check sample as independent, identically distributed, bounded observations from the same population as the future image. It does not require pixels within an image to be independent: all pixels have already been combined into one binary loss for that image. However, dependence between images—for example, multiple images from the same patient—can undermine this independence assumption. Since this subset copy has no documented patient IDs, interpret the result as an image-level exploratory stress test, not a patient-level medical guarantee.

## RCPS and split conformal are different

The notebook also contains an ordinary split-conformal calculation as a comparison. It uses the same maximum-pixel score but selects its threshold by a conformal order statistic. Its `alpha` is a nominal miscoverage level: for example, `alpha = 0.10` targets 90% marginal whole-image coverage under exchangeability of calibration and future examples. This `alpha` is separate from RCPS `epsilon` and `delta`.

The RCPS section asks: *Does the computed upper bound fall at or below the allowed miss risk `epsilon`?* It uses `delta` to specify the confidence of that bound and splits calibration data between radius selection and risk checking. Do not read the conformal `alpha` column as RCPS confidence or RCPS risk.

## Running in Google Colab

1. Open [`lodopab_small_subset_rcps.ipynb`](lodopab_small_subset_rcps.ipynb) in Colab.
2. Choose **Runtime → Change runtime type → GPU** if a GPU is available. The notebook also supports CPU, but processing the simulated projections will take longer.
3. Run the cells from top to bottom. The first cell installs the needed packages; the next downloads the subset.
4. Read the RCPS output table separately from the split-conformal table. Check whether `risk upper bound <= epsilon` before calling an RCPS result certified.

The subset is intentionally small for a cloud notebook. Results can vary with the random split and model training, and the test set has only 100 images, so test coverage estimates are coarse.
