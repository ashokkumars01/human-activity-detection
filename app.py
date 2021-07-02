from flask import Flask,render_template,request,url_for
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='MODEL_VGG16.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    img = load_img(img_path,target_size=(224,224))
    x = img_to_array(img)
    x = x/255.0
    x = np.expand_dims(x,axis=0)
    result=model.predict(x)
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        # get the file from post request
        f=request.files['file']

        # save the file to uploads folder
        basepath=os.path.dirname(os.path.realpath('__file__'))
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result =model_predict(file_path,model)
        #os.remove(file_path)

        #categories=['Cycling','Farming','Jumping','Sitting','Walking']

        # process your result for human
        pred_class = result.argmax()
        #output=categories[pred_class]
        #return output
        if pred_class == 0:
            return "Cycling"
        elif pred_class == 1:
            return "Farming"
        elif pred_class == 2:
            return "Jumping"
        elif pred_class == 3:
            return "Sitting"
        else:
            return "Walking"
    return None


if __name__ == '__main__':
    app.run(debug=True,port=5001)
