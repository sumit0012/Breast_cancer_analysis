import gradio as gr
from skops.io import load
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the pre-trained model
pipe = load("Model/cancer_pipeline.skops", trusted=True)

def predict_cancer(texture_mean,symmetry_mean,texture_se,area_se,smoothness_se,concavity_se,symmetry_se,fractal_dimension_se,smoothness_worst):
    features = [[texture_mean,symmetry_mean,texture_se,area_se,smoothness_se,concavity_se,symmetry_se,fractal_dimension_se,smoothness_worst]]
    # Standardize features using the same scaler used during training
    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(features)
    predicted_cancer = pipe.predict(features)[0]

    label = "Malignant" if predicted_cancer == "M" else "Benign"
    # print(predicted_cancer)
    features[0] = [predicted_cancer] + features[0]
    
    temp_df = pd.read_csv("Data/data_cancer.csv")
    new_df = pd.DataFrame(features, columns=['diagnosis', 'texture_mean', 'symmetry_mean', 'texture_se', 'area_se', 'smoothness_se', 'concavity_se', 'symmetry_se', 'fractal_dimension_se', 'smoothness_worst'])
    temp_df = pd.concat([temp_df, new_df], ignore_index=True)
    temp_df = temp_df.drop_duplicates()
    temp_df.to_csv("Data/data_cancer.csv", header=True, index=False)

    return label
    # if predicted_cancer == 1 :

inputs = [
    gr.Slider(1, 40, step=0.1, label="texture_mean"),
    gr.Slider(0, 1, step=0.1, label="symmetry_mean"),
    gr.Slider(0, 4, step=1, label="texture_se"),
    gr.Slider(10, 190, step=10, label="area_se"),
    gr.Slider(0, 1, step=0.01, label="smoothness_se"),
    gr.Slider(0, 1, step=0.1, label="concavity_se"),
    gr.Slider(0, 1, step=1, label="symmetry_se"),
    gr.Slider(0, 2, step=1, label="fractal_dimension_se"),
    gr.Slider(0, 2, step=1, label="smoothness_worst")
]

outputs = gr.Label(num_top_classes=2)

examples = [
    [10.38,0.2419,0.9053,153.40,0.006399,0.05373,0.03003,0.006193,0.16220],
    [24.54,0.1587,1.4280,19.15,0.007189,0.00000,0.02676,0.002783,0.08996],
    [17.77,0.1812,0.7339,74.08,0.005225,0.01860,0.01389,0.003532,0.12380]
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
    theme=gr.themes.Soft(),
).launch(debug = True)
