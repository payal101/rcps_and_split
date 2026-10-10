# Tutorial: RCPS on the small LoDoPaB-derived dataset

This guide explains the **RCPS (risk-controlling prediction set)** experiment in [`lodopab_colab_small_subset.ipynb`](lodopab_colab_small_subset.ipynb). The notebook trains an image reconstruction model, chooses an interval radius using calibration images, then checks whether it can certify a requested image-level miss risk.

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

### 5. Compute the risk upper bound

For a chosen `lambda`, let `n` be the number of risk-check images and `r_hat` their mean image-level loss. The notebook uses the one-sided Hoeffding bound

`risk upper bound = min(1, r_hat + sqrt(log(1/delta) / (2*n)))`.

Here `delta = 0.10`, so the bound has 90% confidence under the sampling assumptions below. The result is called **certified** for target `epsilon` only if this upper bound is at most `epsilon`.

The optimization step looks for a narrow radius whose optimization-half risk leaves room for this Hoeffding margin. If the margin is too large, an epsilon may be impossible to certify with the available calibration sample. That is a meaningful result: it says this sample and bound do not establish the requested risk level.

### 6. Evaluate on untouched test images

For a selected radius, the notebook reports:

- **Risk-check empirical risk:** fraction of risk-check images with at least one missed pixel.
- **Risk upper bound:** the RCPS certificate; compare it with `epsilon`.
- **Test coverage:** fraction of the 100 held-out images for which every pixel is inside the interval. This is an empirical diagnostic, not the certificate.
- **Test mean width:** average interval width over test images and pixels. Wider intervals generally improve coverage but are less informative.

It also plots a test reconstruction, interval width, and missed pixels for an example image. A mostly blank missed-pixel plot does not imply whole-image coverage: a single missed pixel makes that image a whole-image miss.

## What guarantee is being targeted?

For a radius selected using the optimization half, the separate risk-check half gives a high-confidence upper bound on the probability that a future image has **at least one pixel outside its interval**. A certified line means the computed upper bound is no larger than the requested `epsilon`.

The simple Hoeffding calculation treats the image-level losses in the risk-check sample as independent, identically distributed, bounded observations from the same population as the future image. It does not require pixels within an image to be independent: all pixels have already been combined into one binary loss for that image. However, dependence between images—for example, multiple images from the same patient—can undermine this independence assumption. Since this subset copy has no documented patient IDs, interpret the result as an image-level exploratory stress test, not a patient-level medical guarantee.

## RCPS and split conformal are different

The notebook also contains an ordinary split-conformal calculation as a comparison. It uses the same maximum-pixel score but selects its threshold by a conformal order statistic. Split conformal targets marginal coverage under exchangeability of calibration and future examples.

The RCPS section answers a different question: *Can this chosen interval rule be certified to have image-level miss risk at most `epsilon`, with confidence `1 - delta`?* Its Hoeffding certificate uses independent image-level losses and spends calibration data between selection and risk checking. Do not treat the two outputs as the same guarantee.

## Running in Google Colab

1. Open [`lodopab_colab_small_subset.ipynb`](lodopab_colab_small_subset.ipynb) in Colab.
2. Choose **Runtime → Change runtime type → GPU** if a GPU is available. The notebook also supports CPU, but processing the simulated projections will take longer.
3. Run the cells from top to bottom. The first cell installs the needed packages; the next downloads the subset.
4. Read the RCPS output table separately from the split-conformal table. Check whether `risk upper bound <= epsilon` before calling an RCPS result certified.

The subset is intentionally small for a cloud notebook. Results can vary with the random split and model training, and the test set has only 100 images, so test coverage estimates are coarse.
