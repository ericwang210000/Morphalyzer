# GANalyzer
Detecting Synthetic Images Through File and Pixel Forensics

# Data Sources & Preprocessing
This project uses a balanced dataset of 20,000 images, comprising:
	•	10,000 AI-generated faces from the This Person Does Not Exist (TPDNE) dataset on Kaggle, curated by David Lorenzo.
	•	10,000 real face images from the CelebA dataset (Liu et al., 2015).

All images were preprocessed as follows:
	•	Resized to 128×128 pixels using the Pillow library.
	•	No padding was applied; aspect ratios were preserved through center-cropping when needed, followed by resizing.
	•	Images were loaded and converted to RGB format to ensure consistency across both domains.



Citation for CelebA

Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 3730–3738.