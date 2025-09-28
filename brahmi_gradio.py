import cv2
import numpy as np
import gradio as gr
from keras.preprocessing import image
import re
import tensorflow as tf
from PIL import Image
import os
import json

# -------------------- Processing Pipeline --------------------
def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = cv2.rotate(arr, angle)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    scores, angles = [], np.arange(-limit, limit + delta, delta)
    for angle in angles:
        _, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return best_angle, rotated

def preprocess_image(image):
    _, corrected_image = correct_skew(image)
    gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    filter1 = cv2.medianBlur(gray, 5)
    filter2 = cv2.GaussianBlur(filter1, (5, 5), 0)
    denoised_image = cv2.fastNlMeansDenoising(filter2, None, 17, 9, 17)
    _, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def segment_characters(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_image = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(letter_image, (x, y), (x + w, y + h), (0, 256, 0), 2)
    return letter_image, contours

def recognize_characters(image_folder):
    model = tf.keras.models.load_model("my_model.h5")
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    predicted_labels = []
    for img_name in os.listdir(image_folder):
        if img_name.startswith('.'): continue
        img_path = os.path.join(image_folder, img_name)
        img = image.load_img(img_path, target_size=(200, 200))
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        prediction = model.predict(X)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = [k for k, v in class_indices.items() if v == predicted_class_index][0]
        predicted_labels.append(predicted_class_name)
    reconstructed_text = ''.join(predicted_labels)
    return re.sub(r'\d+', '', reconstructed_text)

def process_pipeline(uploaded_image):
    image_np = np.array(uploaded_image.convert("RGB"))
    preprocessed_image = preprocess_image(image_np)
    segmented_characters, contours = segment_characters(preprocessed_image)
    temp_dir = "temp_segments"
    os.makedirs(temp_dir, exist_ok=True)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if h < 20 and w < 20: continue
        cropped_image = preprocessed_image[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(temp_dir, f"{i}.png"), cropped_image)
    recognized_text = recognize_characters(temp_dir)
    return (
        Image.fromarray(image_np),
        Image.fromarray(preprocessed_image),
        Image.fromarray(segmented_characters),
        "Recognized Text in Kannada:\n\n" + recognized_text
    )

# -------------------- UI Layout --------------------
custom_css = """
#app { padding: 24px !important; box-sizing: border-box; }  /* keep content off screen edges */  /* [web:2][web:15] */
.gradio-container { max-width: 100% !important; min-height: 100vh; background-color: #0f0f0f; color: #e0e0e0; }  /* [web:2] */

.app-row { gap: 24px; }  /* space between sidebar/results columns [web:4] */

#sidebar { background-color: #1c1c1c; padding: 24px; border: 1px solid #333; border-radius: 12px; }  /* no full-height lock [web:4] */
#results { padding: 24px; background-color: #141414; border: 1px solid #333; border-radius: 12px; overflow: auto; }  /* [web:4] */

.output-box { margin-bottom: 24px; border: 1px solid #333; border-radius: 12px; background: #1f1f1f; box-shadow: 0 2px 6px rgba(0,0,0,0.4); padding: 16px; }  /* [web:6] */

.gr-button { background-color: #2d6cdf; color: #fff; border-radius: 8px; border: none; font-weight: 600; padding: 10px 16px; transition: background 0.2s ease; }  /* [web:6] */
.gr-button:hover { background-color: #1e4ea8; }  /* [web:6] */
.gr-textbox textarea { background-color: #121212 !important; color: #e0e0e0 !important; border-radius: 8px; border: 1px solid #444; padding: 12px; font-size: 15px; line-height: 1.5; }  /* [web:6] */

#results .gr-image { margin-bottom: 20px; }  /* [web:13] */
#results .gr-textbox { margin-bottom: 20px; }  /* [web:10] */
"""
with gr.Blocks(css=custom_css, theme=gr.themes.Base(), elem_id="app") as demo:  # padded container [web:2][web:15]
    gr.HTML("""
        <div style='text-align:center; padding: 12px 0;'>
            <h1 style='color:#f5f5f5; font-size:28px; font-weight:700; margin-bottom:8px;'>Brahmi Script Translator</h1>
            <p style='color:#aaa; font-size:16px;'>Upload inscriptions, process, and view Kannada translations</p>
        </div>
    """)  # [web:9]
    with gr.Row(elem_classes=["app-row"]):  # gap between columns [web:4]
        with gr.Column(scale=3, elem_id="sidebar"):
            input_image = gr.Image(type="pil", label="Upload Brahmi Inscription", height="auto")  # [web:13]
            process_btn = gr.Button("Process Image", elem_classes="process-btn")  # [web:16]
        with gr.Column(scale=7, elem_id="results"):
            with gr.Column():
                output_uploaded = gr.Image(label="Uploaded Image", visible=False, elem_classes="output-box")  # [web:13]
                output_preprocessed = gr.Image(label="Preprocessed Image", visible=False, elem_classes="output-box")  # [web:13]
                output_segmented = gr.Image(label="Segmented Characters", visible=False, elem_classes="output-box")  # [web:13]
                output_text = gr.Textbox(label="Kannada Translation", visible=False, lines=12, elem_classes="output-box")  # [web:10]
    def reveal(*args):
        return [gr.update(visible=True)] * 4  # [web:18]
    process_btn.click(process_pipeline, inputs=[input_image], outputs=[output_uploaded, output_preprocessed, output_segmented, output_text]).then(
        reveal, inputs=[], outputs=[output_uploaded, output_preprocessed, output_segmented, output_text]
    )  # [web:18]

if __name__ == "__main__":
    demo.launch()
