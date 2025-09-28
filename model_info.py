from keras.models import load_model
#from keras.utils import plot_model

model = load_model("my_model.h5")
model.summary()

#Architecture of the model
#plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)

#layer details
#for layer in model.layers:
#   print(layer.name, layer.__class__.__name__, layer.output_shape, layer.count_params())
