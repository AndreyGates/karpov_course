"""Image compression class using PCA"""
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

class ImagePCA:
    """
    This class compresses an image using PCA and reconstructs it upon calling.

    Attributes
    ----------
    img_path_ : str
        Stores the path of the image file.
        Load the jpg image of shape (H, W, 3)
    n_components_ : int
        Stores the number of principal components used for compression.
    mean_ : ndarray of the shape (W, 3)
        The mean value of the original image used for centering.
    Y_ : ndarray of the shape (3, H, n_components_)
        The compressed image data after applying PCA.
    Vt_ : ndarray of the shape (3, n_components_, W)
        The top n principal components from the SVD.
    """

    def __init__(self, img_path: str, n_components: int) -> None:
        self.img_path_ = img_path
        self.n_components_ = n_components

        self._compress()

    def _compress(self) -> None:
        """
        Compress the image using Singular Value Decomposition (SVD).

        The image is first converted to an array and centered. Then SVD is applied
        to compress the image data. This method sets the Y_ and Vt_ attributes.
        """
        with Image.open(self.img_path_) as img:
            img = np.array(img)

        # Calculate mean (feature-wise)
        self.mean_: np.ndarray = np.mean(img, axis=0)

        # Centering data
        img_centered = img - self.mean_
        # Moving RGB axis to the first position
        img_centered = np.moveaxis(img_centered, -1, 0)

        # SVD decomposition
        _, S, Vt = np.linalg.svd(img_centered, full_matrices=True)

        # Select the first n_components
        Vt_ = Vt[:, :self.n_components_, :]
        Y_ = np.matmul(img_centered, np.transpose(Vt_, (0, 2, 1)))

        self.Y_ = Y_
        self.Vt_ = Vt_#np.transpose(Vt_, (0, 2, 1))

    def __call__(self) -> PILImage:
        """
        Reconstruct and return the compressed image.

        When the instance is called, it reconstructs the image from the compressed
        data and returns it as a PIL Image object.

        Returns
        -------
        PILImage
            The reconstructed image after PCA compression.
        """
        img_ = np.matmul(self.Y_, self.Vt_)
        # move RGB axis back to the end
        img_ = np.moveaxis(img_, 0, -1)
        # reverse shifting (to return the initial RGB range)
        img_ = img_ + self.mean_
        img_ = np.clip(img_, 0, 255).astype(np.uint8)
        return Image.fromarray(img_)


if __name__ == "__main__":
    img_path = "src_junior/PCA/mug.jpg"
    n_components = 100

    img_pca = ImagePCA(img_path, n_components)
    img = img_pca()

    img.save("src_junior/PCA/reconstructed.jpg")
