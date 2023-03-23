# Model Preparation

As the AI in medical imaging field evolves, it is only natural a good share of the published models will be based on older version of packages and AI frameworks and will face compatibility issues, sooner or later. This is because the hardware acceleration and AI libraries used to develop these pipelines are evolving at a fast pace, with new versions of the frameworks are being released frequently, and (with a few notable exceptions) the model repositories are not maintained for a long time.

Although it is likely that the weights and architecture file formats will become obsolete and not supported by the newer version of such frameworks in the future, the model files are still intact and working perfectly well. For reproducibility purposes, it is therefore recommended to update the models to the latest version of the frameworks to avoid any issues in the future - or convert these models to an open format such as [ONNX](https://onnx.ai/).

The Open Neural Network Exchange Format (ONNX) is a standard for exchanging deep learning models, which was born to bolster models portability and make deployment easier. It allows you to convert deep learning and machine learning models from different frameworks (such as TensorFlow, PyTorch, MATLAB, Caffe, and Keras) to a single format. It defines a common set of operators, common sets of building blocks of deep learning, and a common file format. This makes it easier to move models between different frameworks and hardware platforms. Finally, besides improving model interoperability, ONNX has various other benefits, among which a reported improvement in inference performance for a number of models.

<br>

---

As logged [in the project repository](https://github.com/modelhub-ai/deep-prognosis/blob/master/dockerfiles/keras%3A1.0.1), Hosny et Al. model was developed and trained using Keras 1.2.2. Since the end of November 2022, Google Colab has upgraded its default runtime to Python version 3.8. As widely reported, older version of Tensorflow (v1.x.x) will not be made available for newer python versions. It is therefore, at the moment, impossible to install TensorFlow 1.x.x in Google Colab instances - that should, instead, rely on `tensorflow.compat.v1` for inference and migration to the newer versions of Tensorflow.

Unfortunately, Hosny et Al. model is not compatible with `tensorflow.compat.v1`. Here, we therefore document how to convert this model (and any other model developed in Keras/TF1.x) to the open ONNX format - in order to enable inference using Google Colab and any other system that does not support older versions of Tensorflow.

## Keras to ONNX

Since, depending on the setup of your system, you might need to install some other system dependencies in this process, we advise to run these steps in a Docker container (e.g., one of the base Ubuntu images running `python<3.8`, such as `ubuntu:18.04` - or another image to your liking that runs `python<3.8`). Alternatively, you can use Conda.

To convert the model from an older version of TensorFlow and Keras we need access to those libraries in the first place. As Google Colab does not support these older versions of such packages, these steps cannot unfortunately be run in a Google Colab notebook. Howevery, we report them anyway for transparency and reproducibility purposes.

<br>

After connecting to the docker container, or creating a conda environment running `python<3.8`, we can set up the dependencies using the `keras2onnx_requirements.txt` file in the repository parent folder. 

If using Docker:
```
docker run --volume=$PATH_TO_REPO:/app --rm -it --entrypoint bash ubuntu:18.04

apt-get update && apt-get install -y python3 python3-pip

pip3 install protobuf==3.19.6

pip3 install -r /app/keras2onnx_requirements.txt
```

If using Conda, assuming the working directory is `idc-radiomics-reproducibility/src/model`:
```
conda create -n keras2onnx python=3.7

conda activate keras2onnx

pip install protobuf==3.19.6
pip install -r ../../keras2onnx_requirements.txt
```

To run the conversion script, after making sure the working directory (in either the docker container or the Conda environment) is the one where `convert_model.py` is found, simply run:

```
python3 convert_model.py
```