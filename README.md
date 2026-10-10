# RCPS on the small LoDoPaB-derived dataset

This README explains the RCPS experiment in [`lodopab_small_subset_rcps.ipynb`](lodopab_small_subset_rcps.ipynb), including the image-level loss, the risk target, the confidence bound, and how the notebook decides whether a result is certified.

## 1. Data and prediction model

Each example is an input image `X` and a clean target image `Y`. The notebook uses 700 examples to train a CNN reconstruction model

$$
\hat{Y} = f(X),
$$

where `f(X)` is the predicted image. The model is trained before calibration; the calibration and test images are not used to fit its weights.

The target images come from a small LoDoPaB-derived subset mirrored by GLIMPSE. The notebook creates inputs by simulating 50-view sinograms, adding Gaussian noise, and applying filtered back projection. Thus, targets are real dataset images, but the inputs are simulated, not the official LoDoPaB measurement files. The subset copy does not document patient IDs.

## 2. Pixel intervals and image-level score

For a candidate radius $\lambda \geq 0$, the prediction set for image $X_i$ is a band around the predicted image. At pixel $j$:

$$
\ell_{ij}(\lambda) = \max(0, \hat{Y}_{ij} - \lambda), \qquad
u_{ij}(\lambda) = \min(1, \hat{Y}_{ij} + \lambda).
$$

Pixel $j$ is covered when

$$
\ell_{ij}(\lambda) \leq Y_{ij} \leq u_{ij}(\lambda).
$$

The notebook combines all pixels in one image into a single score:

$$
S_i = \max_{j=1,\ldots,d} \left|Y_{ij} - \hat{Y}_{ij}\right|,
$$

where $d$ is the number of pixels. The image-level loss for radius $\lambda$ is

$$
L_i(\lambda) = \mathbf{1}\{S_i > \lambda\}
             = \mathbf{1}\{\text{at least one pixel in image }i\text{ is missed}\}.
$$

So the notebook makes an interval at **every pixel**, but RCPS counts one binary loss per **image**. It does not require pixels within an image to be independent.

## 3. Population miss risk and `epsilon`

For a fixed trained model and radius, define the population whole-image miss risk as

$$
R(\lambda) = \mathbb{E}_{(X,Y)\sim P}[L(\lambda;X,Y)]
            = \Pr_{(X,Y)\sim P}\!\left(\exists j:\,Y_j\notin[\ell_j(\lambda),u_j(\lambda)]\right).
$$

This is the probability that **one image sampled from the target population** has any missed pixel. It is not the test-set fraction itself.

`epsilon` is the largest population miss risk the experiment is asked to allow. For example,

$$
\epsilon = 0.10 \quad\Longrightarrow\quad \text{target } R(\lambda) \leq 0.10,
$$

which corresponds to at least 90% of images being fully covered on average over the population. It does not promise that every image is covered.

## 4. Calibration selection and risk check

The notebook has 300 calibration images. It randomly splits them into:

- $C_{\mathrm{opt}}$: 150 images for selecting a radius;
- $C_{\mathrm{risk}}$: 150 different images for checking the risk bound.

For each radius in a fixed grid $\Lambda$, the optimization half estimates

$$
\widehat{R}_{\mathrm{opt}}(\lambda)
 = \frac{1}{n_{\mathrm{opt}}}\sum_{i\in C_{\mathrm{opt}}}L_i(\lambda).
$$

The notebook selects the smallest grid radius whose optimization risk is at most $\max(0,\epsilon-h)$, where $h$ is the Hoeffding margin below. This selection does not use the risk-check outcomes.

## 5. Confidence, `delta`, and the RCPS upper bound

For the selected radius $\hat{\lambda}$, the risk-check half computes the empirical miss rate

$$
\widehat{R}_{\mathrm{risk}}(\hat{\lambda})
 = \frac{1}{n_{\mathrm{risk}}}\sum_{i\in C_{\mathrm{risk}}}L_i(\hat{\lambda}),
$$

and the one-sided Hoeffding upper bound

$$
U(\hat{\lambda}) = \min\!\left(1,
\widehat{R}_{\mathrm{risk}}(\hat{\lambda}) +
\sqrt{\frac{\log(1/\delta)}{2n_{\mathrm{risk}}}}\right).
$$

The two settings have different meanings:

- **`epsilon` is the allowed miss risk over images.** It says how small $R(\lambda)$ should be.
- **`delta` is the allowed failure probability of the upper-bound procedure.** Confidence is $1-\delta$. With `delta = 0.10`, the bound has 90% confidence under the assumptions below.
- **`U` is the computed upper bound.** It is neither `epsilon` nor confidence.

The formal statement is

$$
\Pr_{C_{\mathrm{risk}}}\!\left(R(\hat{\lambda}) \leq U(\hat{\lambda})\right)
\geq 1-\delta.
$$

This probability is over repeated fresh risk-check calibration samples, conditional on the trained model and the radius chosen using the separate optimization split. It is **not** a claim that $1-\delta$ of images are covered.

The notebook labels the selected radius **certified for `epsilon`** only when

$$
U(\hat{\lambda}) \leq \epsilon.
$$

For example, if `epsilon = 0.10`, `delta = 0.10`, and $U=0.08$, the result says: “With 90% confidence from the risk-check sample, population whole-image miss risk is at most 10%.” If $U=0.13$, it is not certified for `epsilon = 0.10`; that does not prove the true risk is above 10%.

## 6. Why this sample may not certify small risks

Here $n_{\mathrm{risk}}=150$ and `delta = 0.10`, so

$$
h = \sqrt{\frac{\log(10)}{300}} \approx 0.0876.
$$

Therefore, targets such as `epsilon = 0.01` and `0.05` cannot be certified by this bound and sample size, even if the risk-check sample has zero misses: the upper bound cannot be smaller than the margin $h$. “Not certified” means the available sample and bound did not establish the target; it does not prove the model's actual miss risk exceeds it.

## 7. Test-set metrics

The 100 untouched test images are used only after selecting and checking the radius. If `inside[i,j]` is 1 when target pixel $j$ in test image $i$ is inside its interval, then:

$$
\text{pixel coverage}
= \frac{1}{n_{\mathrm{test}}d}\sum_{i=1}^{n_{\mathrm{test}}}\sum_{j=1}^{d}\text{inside}_{ij},
$$

while

$$
\text{whole-image coverage}
= \frac{1}{n_{\mathrm{test}}}\sum_{i=1}^{n_{\mathrm{test}}}
\mathbf{1}\{\text{all pixels in image }i\text{ are inside their intervals}\}.
$$

The first fraction counts pixels; the second counts images. The test metrics are empirical diagnostics and are not used to select the radius or calculate the RCPS bound. The notebook also reports mean interval width and visualizes a test prediction, its width map, and missed pixels.

## 8. Assumptions and limits

The Hoeffding calculation assumes the per-image losses in the risk-check sample are independent, identically distributed, bounded observations from the same population as future images. The risk-check examples must also be separate from the data used to choose the radius. Pixels within an image may be dependent; they have already been combined into one image-level loss. Dependence between images, for example multiple images from one patient, may violate the independent-image assumption. The mirror has no documented patient IDs, so this notebook cannot verify patient-level independence or separation.

This is an exploratory image-level stress test. Its simulated measurement inputs, small sample sizes, and missing patient metadata limit what can be concluded from the results.

## Run in Google Colab

Open [`lodopab_small_subset_rcps.ipynb`](lodopab_small_subset_rcps.ipynb), select **Runtime → Change runtime type → GPU** if available, and run all cells from top to bottom. CPU execution is possible but slower. The first cells install dependencies and download the small subset.
