# Imaging Data Commons - Radiomics Reproducibility Use Case

*Curated by Dennis Bontempi and Andrey Fedorov*

This repository hosts a reproducibility study based on *Hosny et Al. - [Deep learning for lung cancer prognostication: A retrospective multi-cohort radiomics study](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002711)*, developed using the tools provided by the Imaging Data Commons and the Google Cloud Platform.

![Graphical Abstract](https://github.com/ImagingDataCommons/idc-radiomics-reproducibility/blob/main/assets/overview.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImagingDataCommons/idc-radiomics-reproducibility/blob/main/notebooks/pipeline_mwe.ipynb)

Clicking on the badge above will spawn a new Colab session, loading automatically the latest version of the demo notebook. Additional information about the Colab-GitHub integration can be found in [this Colab Notebook](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=WzIRIt9d2huC).

# Introduction

Description of the repository content and the series of analyses performed.

# Hosny et Al.

## Background

Describe why Hosny et Al. paper in an important milestone in the field.

![Model Architecture](https://github.com/ImagingDataCommons/idc-radiomics-reproducibility/blob/main/assets/model_architecture.png)

## Methods and Findings

The original paper performed an integrative analysis on 7 independent datasets across 5 institutions totaling 1,194 NSCLC patients (age median = 68.3 years [range 32.5–93.3], survival median = 1.7 years [range 0.0–11.7]). Using external validation in computed tomography (CT) data, Hosny et Al. identified prognostic signatures using a 3D convolutional neural network (CNN) for patients treated with radiotherapy (n = 771, age median = 68.0 years [range 32.5–93.3], survival median = 1.3 years [range 0.0–11.7]).

They then employed a transfer learning approach to achieve the same for surgery patients (n = 391, age median = 69.1 years [range 37.2–88.0], survival median = 3.1 years [range 0.0–8.8]). The paper finds that the CNN predictions were significantly associated with 2-year overall survival from the start of respective treatment for radiotherapy (area under the receiver operating characteristic curve [AUC] = 0.70 [95% CI 0.63–0.78], p < 0.001) and surgery (AUC = 0.71 [95% CI 0.60–0.82], p < 0.001) patients. The CNN was also able to significantly stratify patients into low and high mortality risk groups in both the radiotherapy (p < 0.001) and surgery (p = 0.03) datasets.

## Conclusions 

Hosny et Al. results provide evidence that deep learning networks may be used for mortality risk stratification based on standard-of-care CT images from NSCLC patients. The evidence presented in this paper motivated and still motivates the use of deep learning for such applications, and further research into better deciphering the clinical and biological basis of deep learning networks - as well as validation in prospective data.

# Replication Notes

The code in this repository provides the user with an example of how the tools provided by the Imaging Data Commons and the Google Cloud Platform can be used to reproduce an AI/ML end-to-end analysis on cohorts hosted by the IDC portal, and to describe what we identified as the best practices to do so.

## Deep Learning Framework Compatibility

Hosny et Al. model was developed using Keras 1.2.2 and an old version of Tensorflow, as stated by the authors (e.g., see [the docker config file in the model GitHub repository](https://github.com/modelhub-ai/deep-prognosis/blob/master/dockerfiles/keras:1.0.1)). Since early 2023, Google Colab instances are running TensorFlow 2.x.x only - and Hosny et Al. model is not compatible with neither TensorFlow 2.x.x nor the Tensorflow and Keras 1.x versions at `tf.compat.v1`. Therefore, pulling the model from the [project repository](https://github.com/modelhub-ai/deep-prognosis) will not work.

To enable the repeatability of this study, we converted the original model in the ONNX format. Additional details will be added here.

# Replication Results

When the same cohort as the authors is analysed, Hosny et Al. model shows a ROC AUC of 0.7, whereas the IDC replicated pipeline reaches a ROC AUC of 0.68.
