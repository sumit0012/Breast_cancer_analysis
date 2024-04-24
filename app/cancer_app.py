import gradio as gr
from skops.io import load
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
pipe = load("Model/cancer_pipeline.skops", trusted=True)

def predict_cancer(texture_mean, symmetry_mean, texture_se, area_se, smoothness_se, concavity_se, symmetry_se, fractal_dimension_se, smoothness_worst):
    features = [[texture_mean, symmetry_mean, texture_se, area_se, smoothness_se, concavity_se, symmetry_se, fractal_dimension_se, smoothness_worst]]
    predicted_cancer = pipe.predict(features)[0]
    label = "Malignant" if predicted_cancer == "M" else "Benign"
    return label

inputs = [
    gr.inputs.Slider(minimum=0, maximum=40, step=0.1, label="Texture Mean"),
    gr.inputs.Slider(minimum=0, maximum=1, step=0.1, label="Symmetry Mean"),
    gr.inputs.Slider(minimum=0, maximum=4, step=1, label="Texture SE"),
    gr.inputs.Slider(minimum=0, maximum=190, step=10, label="Area SE"),
    gr.inputs.Slider(minimum=0, maximum=1, step=0.01, label="Smoothness SE"),
    gr.inputs.Slider(minimum=0, maximum=1, step=0.1, label="Concavity SE"),
    gr.inputs.Slider(minimum=0, maximum=1, step=1, label="Symmetry SE"),
    gr.inputs.Slider(minimum=0, maximum=2, step=1, label="Fractal Dimension SE"),
    gr.inputs.Slider(minimum=0, maximum=2, step=1, label="Smoothness Worst")
]

outputs = gr.outputs.Label(label="Predicted Result", type="auto")

examples = [
    [10.38, 0.2419, 0.9053, 153.40, 0.006399, 0.05373, 0.03003, 0.006193, 0.16220],
    [24.54, 0.1587, 1.4280, 19.15, 0.007189, 0.00000, 0.02676, 0.002783, 0.08996],
    [17.77, 0.1812, 0.7339, 74.08, 0.005225, 0.01860, 0.01389, 0.003532, 0.12380]
]

title = "Breast Cancer Prediction"
description = "Enter the details to predict breast cancer."
article = "This app predicts whether a tumor is malignant or benign based on its features."

gr.Interface(
    fn=predict_cancer,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme="huggingface",
    allow_flagging=False
).launch()
