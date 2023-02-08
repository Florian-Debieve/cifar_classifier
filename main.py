from flask import Flask, request, render_template
from func import pred, create_vgg16, create_eff


app = Flask(__name__, static_url_path='/static', static_folder='static')

# Dictionnary containing the 2 models composing the classifier.
models_ = {'vgg16': create_vgg16(),
           'efficient': create_eff()
           }


@app.route("/", methods=["GET", "POST"])
def classify_image():
    if request.method == "POST":
        image = request.files["image"]
        if image and image.filename:
            class_predicted = pred(image, models_)
            return render_template("results.html", prediction=class_predicted)
        else:
            return render_template("upload.html", message="Please select an image to upload.")
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
