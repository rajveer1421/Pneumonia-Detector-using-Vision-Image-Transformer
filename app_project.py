from flask import Flask,render_template,request
from tensorflow.keras.models import load_model
from vision_transformer import VisionImageTransformer,TransformerBlock,MultiHeadSelfAttention
from preprocessing import process_image
app=Flask(__name__)
model=load_model('Pneumonia_Detector_Model_2.keras', custom_objects={
    "VisionImageTransformer": VisionImageTransformer,
    "TransformerBlock": TransformerBlock,
    "MultiHeadSelfAttention": MultiHeadSelfAttention
})
@app.route('/')
def home():
    return render_template('home1.html')
@app.route('/predict',methods=['POST'])
def predict_pneumonia():
    if 'file' not in request.files:
        return "problem loading the file"
    inp=request.files['file']
    if inp.filename==" ":
        print("No file selected")
    inp=process_image(inp)
    prediction=model.predict(inp)
    probability=prediction[0][0]
    if(probability>0.5):
        label="You have Pneumonia,Get Well Soon"
    else:
        label="Congratulations you don't have pneumonia and you are reports are normal"
    if probability<=0.35:
        confidence="We are highly sure in this prediction"
    if probability>=0.73:
        confidence="We are highly sure in this prediction"
    else:
        confidence="Low confidence in this prediction, consider confirming with a medical professional."
    return render_template('show.html',label=label,confidence=confidence)
if(__name__=='__main__'):
    app.run(debug=True)