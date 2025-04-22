from pathlib import Path

DATA_PATH = Path("data/")
REAL_PATH = DATA_PATH / "real"
FAKE_PATH = DATA_PATH / "fake"
MODEL_SAVE_PATH = "models/GANalyzerModel.pth"
PROJECT_PATH = Path(__file__).resolve().parents[1]

BATCH_SIZE = 32
IMAGE_SIZE = 128
NUM_CLASSES = 2
EPOCHS = 10
LEARNING_RATE = 0.001
