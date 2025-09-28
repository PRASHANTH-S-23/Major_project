# Brahmi Script Translator

A deep learning-powered application for recognizing and translating ancient Brahmi script inscriptions to modern Kannada text. The project uses Convolutional Neural Networks (CNN) for character recognition and provides an intuitive web interface built with Gradio.

## Features

- **Image Preprocessing**: Automatic skew correction, noise reduction, and binarization
- **Character Segmentation**: Intelligent segmentation of individual characters from inscriptions
- **Deep Learning Recognition**: CNN-based character classification trained on Kannada dataset
- **Web Interface**: User-friendly Gradio interface for easy interaction
- **Real-time Processing**: Complete pipeline from image upload to text translation

## Project Structure

```
├── brahmi_gradio.py          # Main Gradio web interface
├── train_model.py            # CNN model training script
├── model_info.py             # Model architecture inspection
├── my_model.h5               # Trained CNN model
├── class_indices.json        # Character class mappings
├── tamil_class_indices.json  # Tamil character mappings
├── requirements.txt          # Python dependencies
├── test_samples/             # Sample images for testing
│   ├── handwritten_1.png
│   ├── script_1.png
│   └── ...
└── project_env/              # Virtual environment
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Major_Project
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv project_env
   # Windows
   project_env\Scripts\activate
   # Linux/Mac
   source project_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

Launch the Gradio web application:
```bash
python brahmi_gradio.py
```

The interface provides:
- Image upload functionality
- Real-time preprocessing visualization
- Character segmentation display
- Kannada text translation output

### Training Your Own Model

To train a new model with your dataset:

```bash
python train_model.py
```

**Dataset Structure Required:**
```
kannada_dataset/
├── train/
│   ├── character_1/
│   ├── character_2/
│   └── ...
├── val/
│   ├── character_1/
│   ├── character_2/
│   └── ...
└── test/
    ├── character_1/
    ├── character_2/
    └── ...
```

### Model Information

Inspect the trained model architecture:
```bash
python model_info.py
```

## Technical Details

### Model Architecture
- **Input**: 200x200x3 RGB images
- **Architecture**: Sequential CNN with multiple Conv2D and MaxPooling2D layers
- **Features**: Dropout regularization, data augmentation
- **Optimizer**: Adam with learning rate 0.001
- **Loss**: Categorical crossentropy

### Image Processing Pipeline
1. **Skew Correction**: Automatic rotation to correct document orientation
2. **Preprocessing**: Median blur, Gaussian blur, and denoising
3. **Binarization**: OTSU thresholding for optimal character separation
4. **Segmentation**: Contour-based character extraction
5. **Recognition**: CNN-based character classification

## Model Performance

The model is trained for 50 epochs with:
- **Batch Size**: 32
- **Image Augmentation**: Rotation, shifting, shearing, and zooming
- **Validation**: Real-time accuracy monitoring

## Dependencies

Key libraries used:
- TensorFlow/Keras - Deep learning framework
- OpenCV - Image processing
- Gradio - Web interface
- NumPy - Numerical computations
- Pillow - Image handling

## Testing

Use the sample images in [`test_samples/`](test_samples/) to test the application:
- Handwritten inscriptions
- Stone script samples
- Individual character tests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ancient script research community & https://www.aksharamukha.com/describe/Brahmi
- Kannada language preservation efforts
- Open source computer vision libraries

## Future Enhancements

- Support for multiple ancient scripts (Tamil, Telugu, etc.)
- Improved character recognition accuracy
- Mobile application development
- Historical context integration
- Batch processing capabilities