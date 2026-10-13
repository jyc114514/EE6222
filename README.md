# Dimensionality Reduction for Face Classification Using PCA and LDA

This repository contains a student coursework project for NTU EE6222 Assignment 1. It is an individual coursework project and is not published by or endorsed by NTU.

## Dataset

The experiment uses the 400-image, 40-identity Olivetti Faces dataset exposed by [sklearn.datasets.fetch_olivetti_faces](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.fetch_olivetti_faces.html); each image is represented as 4096 pixel values. The program stores the scikit-learn cache under data/sklearn_cache/ and downloads the dataset through the loader when the cache is absent. A first run therefore requires internet access and availability of the upstream download service.

[AT&T Laboratories Cambridge Database of Faces page](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) requests credit to AT&T Laboratories Cambridge but does not state redistribution terms on that page. This repository does not redistribute the face images, the scikit-learn cache, or processed image arrays. Please credit AT&T Laboratories Cambridge when using the images and review the source terms for your use.

## Methods

- PCA followed by Euclidean 1-nearest-neighbour classification.
- Direct LDA followed by the same classifier.
- PCA-to-LDA followed by the same classifier.
- Raw pixels with Euclidean 1-nearest-neighbour as the reference.

The PCA candidates are 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, and 200 components. Direct LDA and final LDA output dimensions use 1, 2, 5, 10, 15, 20, 25, 30, 35, and 39. PCA-to-LDA pre-dimensions are selected separately from 50, 75, 100, 150, and 200. The PCA pre-dimension and final LDA output dimension are distinct settings.

## Experimental protocol

The evaluation uses shuffled stratified 5-fold outer cross-validation. Within each outer-training fold, shuffled stratified 4-fold cross-validation selects dimensions by mean validation accuracy. The selected pipeline is refit on the complete outer-training fold and evaluated once on its held-out fold. The outer seed is 20261012; inner seeds are 20261020 + fold. The five outer test folds partition the 400 images, producing 400 out-of-fold predictions. Fold standard deviations describe variation across these five folds; they are not confidence intervals.

The saved v2 results were recorded with CPython 3.10.12, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, pandas 2.3.3, and Matplotlib 3.10.9. OMP, OpenBLAS, and MKL thread counts were each set to one.

## Main results

| Method | Selected settings across folds | Outer mean accuracy ± sample SD |
|---|---|---:|
| Raw 1-NN | 4096 pixels | 0.9400 ± 0.0311 |
| PCA + 1-NN | 40 (2 folds), 75, 150, 200 components | 0.9350 ± 0.0298 |
| Direct LDA + 1-NN | 25, 30 (2 folds), 35, 39 outputs | 0.9875 ± 0.0217 |
| PCA-to-LDA + 1-NN | PCA pre-dimensions 50/75/100; final LDA outputs 30/35/39 | 0.9850 ± 0.0224 |

Direct LDA classified 395/400 images correctly; the selected PCA-to-LDA models classified 394/400. This one-prediction difference is descriptive and does not establish a precise ranking.

The fixed PCA280-to-LDA20 diagnostic averaged 0.5000 ± 0.2095 under raw Euclidean distance and 0.9375 ± 0.0319 after standardizing LDA coordinates using outer-training outputs. The PCA319-to-LDA25 diagnostic averaged 0.6325 ± 0.0716 before standardization and 0.6250 ± 0.0884 after it. This scale intervention was post-hoc. All analyses reuse the same dataset and do not provide an independent sample-level test set.

## Repository structure

- EE6222_Assignment_Standalone.py: complete executable experiment.
- build_submission_pdf.py: PDF builder; defaults to the anonymized public report source.
- EE6222_Final_Report_Public.md: public report source without the matriculation number.
- [EE6222_Assignment1_Report.pdf](EE6222_Assignment1_Report.pdf): public PDF; the matriculation number is omitted from the page and PDF author metadata.
- requirements.txt: Python dependencies for the experiment, report builder, and tests.
- src/, tests/: supporting modules and unit tests.
- results/second_pass/: selected-model results, dimension curves, summary tables, and five figures.
- results/diagnostics/: supplemental diagnostic plots and selected summary tables.

Local caches, processed dataset copies, audit manifests, detailed logs, and identity-bearing submission files are excluded from this public repository.

## Installation and reproduction

Use Python 3.10 or later. From the repository root:

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python EE6222_Assignment_Standalone.py
~~~

On macOS or Linux, activate with source .venv/bin/activate. The first experiment run downloads Olivetti Faces if the local scikit-learn cache is missing. The script writes results under results/second_pass/ and audit output under audit/second_pass/; the cache is local under data/sklearn_cache/.

To rebuild the public report PDF:

~~~powershell
python build_submission_pdf.py
~~~

The builder uses ReportLab. pdfinfo from Poppler is optional and is used only to print the page count.

## Report

The public report is [EE6222_Assignment1_Report.pdf](EE6222_Assignment1_Report.pdf). It omits the matriculation number from both the displayed report and PDF metadata. The local submission copy contains the full course identity and is not included here.

## References and attribution

1. X. Jiang, “Linear Subspace Learning-Based Dimensionality Reduction,” IEEE Signal Processing Magazine, 28(2), 16-26, 2011. doi:10.1109/MSP.2010.939041.
2. P. N. Belhumeur, J. P. Hespanha, and D. J. Kriegman, “Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(7), 711-720, 1997. doi:10.1109/34.598228.
3. Scikit-learn, [fetch_olivetti_faces](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.fetch_olivetti_faces.html), [LinearDiscriminantAnalysis](https://scikit-learn.org/1.7/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html), and [cross-validation guide](https://scikit-learn.org/1.7/modules/cross_validation.html), version 1.7.
4. AT&T Laboratories Cambridge, [The Database of Faces](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html). Please credit AT&T Laboratories Cambridge.

No software license is included with this coursework repository. The dataset is not redistributed here.
