import json

import onnx
import keras
import keras2onnx

onnx_model_name = "../weights/model.onnx"

arch_json_path = "../weights/architecture.json"
weights_path = "../weights/weights.h5"

with open(arch_json_path, 'r') as json_file:
    model_json = json.load(json_file)

model = keras.models.model_from_config(model_json)
model.load_weights(weights_path)

onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, onnx_model_name)