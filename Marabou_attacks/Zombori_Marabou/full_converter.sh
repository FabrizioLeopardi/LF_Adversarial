cd raw_model
python3 generate_model.py
cd ..
python3 -m tf2onnx.convert --saved-model ./NN_model --output onnx_model_NN/model.onnx --opset 13
