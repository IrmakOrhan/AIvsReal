from fastai.vision.all import *
import gradio as gr

# Modelleri yükle
model_base = load_learner("final_model.pkl")
model_advanced = load_learner("final_model_advanced.pkl")
model_resnet18 = load_learner("resnet18-benchmark.pkl")

# Tahmin fonksiyonu
def predict_all_models(img):
    pred_base = model_base.predict(img)[2]
    pred_adv = model_advanced.predict(img)[2]
    pred_res18 = model_resnet18.predict(img)[2]

    vocab_base = model_base.dls.vocab
    vocab_adv = model_advanced.dls.vocab
    vocab_res18 = model_resnet18.dls.vocab

    result_base = {vocab_base[i]: float(pred_base[i]) for i in range(len(pred_base))}
    result_adv = {vocab_adv[i]: float(pred_adv[i]) for i in range(len(pred_adv))}
    result_res18 = {vocab_res18[i]: float(pred_res18[i]) for i in range(len(pred_res18))}

    return result_base, result_adv, result_res18

# Hugging Face Spaces otomatik olarak bu 'app' nesnesini çalıştırır
app = gr.Interface(
    fn=predict_all_models,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=2, label="Model ResNet34 (Base)"),
        gr.Label(num_top_classes=2, label="Model ResNet34 (Fine-Tuned)"),
        gr.Label(num_top_classes=2, label="Model ResNet18 (Benchmark)")
    ],
    title="AI vs Real Image Classifier",
    description="This app predicts whether an uploaded image is AI-generated or real using three different models."
)
