from PIL import Image
from glob import glob
import os

def convertAndSave(filenames, new_directory):
    for path in filenames:
        img = Image.open(path).convert('LA')
        parts = str(path).split('/')
        filename = parts[-1]
        filename = filename.split('.')[0]
        img.save(new_directory + filename + ".png")

def main():
    cwd =  os.path.dirname(os.path.realpath(__file__))
    data_directory = cwd + "/data"

    test_normal = glob(data_directory + "/test/NORMAL/*")
    test_pneu = glob(data_directory + "/test/PNEUMONIA/*")

    train_normal = glob(data_directory + "/train/NORMAL/*")
    train_pneu = glob(data_directory + "/train/PNEUMONIA/*")

    val_normal = glob(data_directory + "/val/NORMAL/*")
    val_pneu = glob(data_directory + "/val/PNEUMONIA/*")


    convertAndSave(test_normal, cwd + "/gray_data/test/NORMAL/")
    convertAndSave(test_pneu, cwd + "/gray_data/test/PNEUMONIA/")
    convertAndSave(train_normal, cwd + "/gray_data/train/NORMAL/")
    convertAndSave(train_pneu, cwd + "/gray_data/train/PNEUMONIA/")
    convertAndSave(val_normal, "/gray_data/val/NORMAL/")
    convertAndSave(val_pneu, "/gray_data/val/PNEUMONIA/")

if __name__ == "__main__":
    main()