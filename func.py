import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize


def adapt_img(image_loc, model_type): # Reshape the uploaded image in the correct size and preprocess it depending on the model it will be fed to.
    image = Image.open(image_loc)
    image = resize(np.array(image), (32, 32, 3))
    image = np.array(image * 255, dtype='int')
    if model_type == 'vgg16':
        return tf.keras.applications.vgg16.preprocess_input(image).reshape(1, 32, 32, 3)
    elif model_type == 'efficient':
        return tf.keras.applications.efficientnet.preprocess_input(image).reshape(1, 32, 32, 3)
    else:
        return 'error'


def prob(image_loc, model, model_type): # Output the probabilities of the image being a part of each of the 10 categories of the classifier.
    return model.predict(adapt_img(image_loc, model_type))[0]


def pred(image_loc, models): # Output the final prediction of the classifier and a graph with the probabilities for each categories.
    models_names = list(models.keys())
    class_names = {
        0: 'plane',
        1: 'car',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    preds = [prob(image_loc, models[m], m) for m in models_names]
    preds = np.stack(preds, axis=1)
    preds = np.sum(preds, axis=1) / len(models)
    class_pred = class_names[np.argmax(preds)]
    plot = create_plot(preds)
    plot.savefig("static/proba_plot.png")
    return class_pred


def create_plot(p): # Create the graph with the probabilities for each categories.
    catg = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
    plt.figure(facecolor='#808080')
    plt.rcParams["text.color"] = "white"
    plt.xticks(range(len(catg)), catg, rotation=0, color = "white")
    plt.gca().tick_params(axis='x', colors='white')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(catg)))
    plt.bar(range(len(catg)), p, color=colors)
    plt.gca().set_frame_on(False)
    plt.gca().set_yticklabels([])
    plt.gca().set_yticks([])
    plt.title("Probabilities for each category",y=1.2)
    for i, h in enumerate(p):
        plt.text(x=i - 0.5, y=h + 0.05, s=f"{h*100:.2f}%", fontsize=8)
    return plt


def create_vgg16(): # Build the first model composing the classifier with the weights and architecture files.
    with open("static/weights_vgg16/vgg16_architecture.json", "r") as f:
        model_architecture = f.read()
    model = tf.keras.models.model_from_json(model_architecture)
    weights = []
    num_splits = 8
    for i in range(num_splits):
        filename = "static//weights_vgg16/my_weights_split_{}.npy".format(i)
        split = np.load(filename, allow_pickle=True)
        weights.append(split)
    weights = list(np.concatenate(weights))
    model.set_weights(weights)
    return model


def create_eff(): # Build the second model composing the classifier with the weights and architecture files.
    with open("static/weights_eff/eff_architecture.json", "r") as f:
        model_architecture = f.read()
    model = tf.keras.models.model_from_json(model_architecture)
    weights = []
    num_splits = 7
    for i in range(num_splits):
        filename = "static//weights_eff/my_weights_split_{}.npy".format(i)
        split = np.load(filename, allow_pickle=True)
        weights.append(split)
    weights = list(np.concatenate(weights))
    model.set_weights(weights)
    return model

