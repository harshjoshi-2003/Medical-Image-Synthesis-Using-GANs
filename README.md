# GAN for Pneumonia Detection

This project implements a Generative Adversarial Network (GAN) to generate images for pneumonia detection using the Chest X-ray dataset. The GAN consists of a generator and a discriminator model, trained to create synthetic X-ray images that can be used to identify pneumonia.

## Dataset

The dataset used for this project is the [Chest X-ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle. The dataset includes X-ray images categorized as either "NORMAL" or "PNEUMONIA".

## Project Structure

- **Data Preprocessing**: Images are preprocessed and normalized using `ImageDataGenerator`.
- **Generator Model**: Defines the architecture for generating synthetic images.
- **Discriminator Model**: Defines the architecture for distinguishing between real and fake images.
- **GAN Model**: Combines the generator and discriminator with custom training steps.
- **Training**: The GAN is trained using the Chest X-ray dataset, and the generator's progress is monitored through sample images saved after each epoch.

## Requirements

- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn

You can install the required packages using pip:

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn
```

## Usage

**1. Clone the repository**
```bash
git clone https://github.com/harshjoshi-2003/Medical-Image-Synthesis-Using-GANs.git
```

**2. Download the dataset :**
Download the dataset from Kaggle and place it in the input directory. Ensure the directory structure matches:
```bash
input/
└── chest-xray/
    ├── train/
    │   ├── PNEUMONIA/
    │   └── NORMAL/
    └── val/
        ├── PNEUMONIA/
        └── NORMAL/
```

**3. Run the Training**
```bash
python gan.py
```

## Callbacks
- Gan_Callback: Saves generated images and model checkpoints every 10 epochs. Images are saved as After_epochs_{epoch}.png.

## Results
After training, generated images will be saved periodically to monitor the progress of the GAN. These images will be stored in the current directory with filenames in the format After_epochs_{epoch}.png.

## Saving Models
The trained generator and discriminator models are saved to Google Drive for future use. Ensure you have access to Google Drive and mount it appropriately in your environment.

## Acknowledgements
- Kaggle Chest X-ray Pneumonia Dataset
- TensorFlow and Keras for deep learning functionalities
- OpenCV for image processing
