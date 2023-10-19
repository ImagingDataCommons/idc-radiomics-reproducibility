# Imaging Data Commons - Radiomics Reproducibility Use Case

[![DOI](https://zenodo.org/badge/615249629.svg)](https://zenodo.org/badge/latestdoi/615249629)

Transparent and Reproducible AI-based Medical Imaging Pipelines Using the Cloud.

If you use code or parts of this code in your work, please cite our publication:

> Dennis Bontempi, Leonard Nuernberg, Deepa Krishnaswamy, et Al. - Transparent and Reproducible AI-based Medical Imaging Pipelines Using the Cloud. https://doi.org/10.21203/rs.3.rs-3142996/v1

# Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [On the original study](#on-the-original-study)
- [Replication Notes](#replication-notes)
- [Using This Resource Locally](#using-this-resource-locally)
- [Acknowledgments](#acknowledgments)

# Overview

This repository hosts all the code for the replication of *Hosny et Al. - [Deep learning for lung cancer prognostication: A retrospective multi-cohort radiomics study](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002711)*, achieved using the tools provided by the Imaging Data Commons and the Google Cloud Platform. Our goal is to provide the user with an example of how the cloud (specifically, the tools provided by the Imaging Data Commons and the Google Cloud Platform) can be used to reproduce an AI/ML end-to-end analysis, and to describe what we identified as the best practices to do so.

![Graphical Abstract](https://github.com/ImagingDataCommons/idc-radiomics-reproducibility/blob/main/assets/overview.png)

# Repository Structure

This repository is organized as follows.

## Notebooks

In this folder, the user can find three different python notebooks developed using Colab.

The `processing_example` notebook provides the users with a detailed explanation of all the steps of the processing pipeline, together with the code to run such processing on one patient randomly selected from the validation dataset (NSCLC-Radiomics). This notebook serves as a minimal reproducible example, helps the user understand what are the steps involved in the pipeline, and elaborates on the best tools to run these steps

👇 To access this notebook, click on the badge below

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImagingDataCommons/idc-radiomics-reproducibility/blob/main/notebooks/processing_example.ipynb)

---

The `complete_inference` notebook is an extension of the `processing_example` notebook, and stores all of the code necessary to replicate the model validation in its entirety. This notebook was used to generate the results presented in the paper, visualized in the `results_comparison` notebook (see below), and stored in `data/nsclc-radiomics_reproducibility.csv` for ease of access. Indeed, it is worth to mention the complete replication of the study will take the user a few hours on Colab, due to the computationally intensive operations involved in the processing pipeline.

👇 To access this notebook, click on the badge below

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImagingDataCommons/idc-radiomics-reproducibility/blob/main/notebooks/complete_inference.ipynb)

---

The `results_comparison` notebook provides the users with all the code to compare the results from the original publication with those of our replication study. In this notebook we compute and visualize the Area Under the Receiver Operating Characteristic curve (AUROC), the Area Under the Precision Recall Curve (AUROC), the results of the survival analysis and the Kaplan-Meier analysis - as well as the tools we used to compare the AUROCs of the original study and our replication (two-sided Mann-Whitney U testing and DeLong testing for paired AUC curves). 

👇 To access this notebook, click on the badge below

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImagingDataCommons/idc-radiomics-reproducibility/blob/main/notebooks/results_comparison.ipynb)

## Source Code

The `src` folder stores all the code and files useful for data preparation and processing.

In particular, the `weights` folder stores the model files provided with the original publication and the model weights converted in the open neural network exchange (ONNX) format (see the [Deep Learning Framework Compatibility](#deep-learning-framework-compatibility) section of this README for additional details).

The `utils` folder contains all of the custom functions we used in the notebooks for data handling, processing and visualization purposes. We decided to outsourse these functions to these scripts to keep the notebooks more lightweight. Nevertheless, the code in these scripts is documented to our best and follows the principles of transparency and reproducibility of our work.

Finally, the `model` folder stores all of the code we used to convert the model shared with the original publication in the open neural network exchange (ONNX) format. We also provide additional information and documentation regarding this step in the README stored in the folder.

## Data

The `data` folder stores the CSV files shared with the original publication and generated in our replication study.

The `Radboud.csv` and `nsclc-radiomics_hosny_baseline.csv` were shared with the original publication, and store the radiomics features and the outputs of the deep learning model. We used the data stored in the first file to compute the median split threshold we used for the survival analysis in the `results_comparison` notebook, while the data in the second file were used to compare the results of the original study to those of our replication.

The `nsclc-radiomics_reproducibility.csv` stores the results of the `complete_inference` for ease of access (since the complete replication of the study will take the user a few hours on Colab, due to the computationally intensive operations involved in the processing pipeline).

Finally, the clinical data in `nsclc-radiomics_clinical.csv`, useful for the survival analysis and the Kaplan-Meier analysis computed in the `results_comparison` notebook, were obtained from the [TCIA page of the NSCLC-Radiomics dataset](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics). It is worth noting that the clinical data associated to cohorts hosted by the Imaging Data Commons can also be retrieved through the platform, following the tutorial found in [this notebook](https://github.com/ImagingDataCommons/IDC-Tutorials/blob/master/notebooks/clinical_data_intro.ipynb).

# On the Original Study

Hosny et Al. paper in an important milestone in the AI in medical imaging and oncology field, as it was one of the first papers comparing radiomics and deep learning for lung cancer prognostication in a (retrospective) multi-cohort setting.

![Model Architecture](https://github.com/ImagingDataCommons/idc-radiomics-reproducibility/blob/main/assets/model_architecture.png)

In the original work, the authors performed an integrative analysis on 7 independent datasets across 5 institutions totaling 1,194 NSCLC patients (age median = 68.3 years [range 32.5–93.3], survival median = 1.7 years [range 0.0–11.7]). Using external validation in computed tomography (CT) data, Hosny et Al. identified prognostic signatures using a 3D convolutional neural network (CNN) for patients treated with radiotherapy (n = 771, age median = 68.0 years [range 32.5–93.3], survival median = 1.3 years [range 0.0–11.7]).

They then employed a transfer learning approach to achieve the same for surgery patients (n = 391, age median = 69.1 years [range 37.2–88.0], survival median = 3.1 years [range 0.0–8.8]). The paper finds that the CNN predictions are significantly associated with 2-year overall survival from the start of respective treatment for radiotherapy (area under the receiver operating characteristic curve [AUC] = 0.70 [95% CI 0.63–0.78], p < 0.001) and surgery (AUC = 0.71 [95% CI 0.60–0.82], p < 0.001) patients. The authors show the deep learning model is able to significantly stratify patients into low and high mortality risk groups in both the radiotherapy (p < 0.001) and surgery (p = 0.03) datasets, outperforming a random forest classifier built using clinical parameters and engineered (radiomics) features.

Hosny et Al. was one of the first papers providing evidence that deep learning networks may be used for mortality risk stratification based on standard-of-care CT images from NSCLC patients. The evidence presented in this paper motivated and still motivates the use of deep learning for such applications, and further research into better deciphering the clinical and biological basis of deep learning networks - as well as validation in prospective data.

# Replication Notes

## Deep Learning Framework Compatibility

Hosny et Al. model was developed using Keras 1.2.2 and an old version of Tensorflow, as stated by the authors (e.g., see [the docker config file in the model GitHub repository](https://github.com/modelhub-ai/deep-prognosis/blob/master/dockerfiles/keras:1.0.1)). Since early 2023, Google Colab instances are running TensorFlow 2.x.x only - and Hosny et Al. model is not compatible with neither TensorFlow 2.x.x nor the Tensorflow and Keras 1.x versions at `tf.compat.v1`. Therefore, pulling the model from the [project repository](https://github.com/modelhub-ai/deep-prognosis) will not work.

To enable the repeatability of this study and promote forward compatibility, we converted the original model weights in the open neural network exchange format (ONNX). Complete instructions on how to replicate the conversion can be found in the [README file stored in the `src/model` folder](https://github.com/ImagingDataCommons/idc-radiomics-reproducibility/tree/main/src/model).

## Results

After testing with various statistical tools, we observed that the difference between the results produced by the pipeline originally published by Hosny et Al. and those generated in our replication are not statistically significant (`p>0.05` for the two-sided Mann-Whitney U test and the DeLong test for paired AUC curves). We can therefore conclude that the discrepancy between the two predictive models holds little to no significance or impact on the overall outcome or findings. These results also prove the model's robustness to variation in the input segmentation mask, as the original work claims.

Furthermore, we conducted a Kaplan-Meier analysis, as done in the original publication, to assess the stratification power of the AI pipeline. We found that both the original pipeline and the replicated pipeline can successfully stratify higher-risk patients from lower-risk patients (`p<0.001` and `p=0.023`, for the original and the replicated pipeline, respectively) when the risk-score threshold shared with the original publication is used to compute the split.

# Using This Resource Locally

## Local Runtime Using Docker

We strongly believe using the cloud is the most immediate, reliable, and easy solution - as it does not require any additional set-up or readily available hardware. However, should any user want to use the resources shared within this repository locally, we suggest they follow the instructions below.

The best way to reproduce the computational environment found on Google Colab (i.e., both the system and the Python dependencies) is containerization. Fortunately, since earlier this year, Google has made all the versions of the Docker containers upon which Colab is built publicly available [as part of this registry](https://console.cloud.google.com/artifacts/docker/colab-images/us/public/runtime). 

After [setting up Docker on your system](https://www.docker.com/get-started/), to pull the latest Docker Colab image, the user can run the following command:

```
docker pull us-docker.pkg.dev/colab-images/public/runtime
```

Please note this container will work independently from whether you have a working GPU installed or not - but if you want to use a GPU within the container, you should check out [this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to install the `nvidia-container-toolkit` alongside Docker on your machine.

Once the container is pulled, the user should follow the official [Colab local runtimes guide from Google](https://research.google.com/colaboratory/local-runtimes.html) to set up the local instance (by following Step 1 - Option 1). The following command

```
docker run --rm -it -p 127.0.0.1:9000:8080 us-docker.pkg.dev/colab-images/public/runtime
```

will run the previously downloaded container and ensure the local Colab runtime service will be available at port 9000 on your local machine. To enable GPU support (if the `nvidia-container-toolkit` was correctly installed), the user can run:

```
docker run --rm -it --gpus all -p 127.0.0.1:9000:8080 us-docker.pkg.dev/colab-images/public/runtime
```

or, if we want the container to access only a specific GPU (e.g., the GPU with id 0):

```
docker run --rm -it --gpus device=0 -p 127.0.0.1:9000:8080 us-docker.pkg.dev/colab-images/public/runtime
```

Docker also provides numerous useful options to limit the resources the container can access once running. If you want to learn more, please follow [the official guide](https://docs.docker.com/config/containers/resource_constraints/).

Once the container is up and running, the terminal will display something similar to the following:

```
[...]
To access the notebook, open this file in a browser:
file:///root/.local/share/jupyter/runtime/nbserver-172-open.html (type=jupyter)
Or copy and paste one of these URLs
http://127.0.0.1:9000/?token=c8f179317db76617ae37f47d5f53a3301ea2ce5e7a34f7d1
[...]
```

In order to connect Google Colab to the local instance, we will need to open the notebook we want to run (e.g., the [processing_example](https://colab.research.google.com/github/ImagingDataCommons/idc-radiomics-reproducibility/blob/main/notebooks/processing_example.ipynb) notebook we make available under notebooks) and follow these very simple steps (also highlighted in the official [Colab local runtimes guide](https://research.google.com/colaboratory/local-runtimes.html)):

> Step 2: Connect to the local runtime
In Colab, click the "Connect" button and select "Connect to local runtime...". Enter the URL from the previous step in the dialog that appears and click the "Connect" button. After this, you should now be connected to your local runtime.

<img width="454" alt="image" src="https://github.com/ImagingDataCommons/idc-radiomics-reproducibility/assets/31729248/30940347-2fb5-42ab-a453-86afb74d06e6">

<img width="632" alt="image" src="https://github.com/ImagingDataCommons/idc-radiomics-reproducibility/assets/31729248/1bf77c5e-3646-48eb-bdbe-1293e986bf60">

<img width="632" alt="image" src="https://github.com/ImagingDataCommons/idc-radiomics-reproducibility/assets/31729248/98716cc5-30c5-44af-bc19-dc81b139034c">

If everything is correctly setup, in the resources panel you should see something like the following:

<img width="284" alt="image" src="https://github.com/ImagingDataCommons/idc-radiomics-reproducibility/assets/31729248/b46810f6-4c13-4442-ac41-f55d9990005d">

If you see this message, you are ready to run our notebooks from a local Colab runtime!

## Local Runtime Without Using Docker

If, for some reason, the user does not want to install Docker on their machine, the official [Colab local runtimes guide from Google](https://research.google.com/colaboratory/local-runtimes.html) provides an alternative based on Jupyter to run a local runtime without it (see Step 1 - Option 2). However, in this case, the user must ensure all of the system and Python dependencies are installed (to mimic the Colab environment).

Please note that this might not be possible if the user runs a different OS or a different OS Version from what Colab is running (i.e., Ubuntu 22.04 LTS as of November 2023).

# Acknowledgments

The authors would like to thank [...]
