# image-colorizer
An implementation that performs image colorization by denoising the Hue and Saturation channels in HSV colorspace


To run this code, you'll first need to install the required libraries using: pip install torch torchvision opencv-python
The code relies on OpenCV's visual display capabilities (cv2.imshow), so you'll need a graphical interface to see the results.
This is a demonstration script that shows how to build and train a basic UNet model using CIFAR-10 images. It processes the images by:

Converting them from RGB to HSV color space
Adding artificial noise to the hue and saturation channels
Training the model to remove this noise

Running this script as-is will train a basic UNet on CIFAR-10 color images. The script converts RGB images to HSV, adds Gaussian noise to the H and S channels, and trains the model to denoise these channels while preserving the V channel. This is a toy example primarily aimed at demonstrating the functionality and workflow; meaningful results may require extended training or fine-tuning.

While this example may not produce high-quality colorization results in just a few training cycles, it provides the foundational structure and concepts you'd need for a more sophisticated implementation.

## To-do:

HSV Conversion:

The rgb_to_hsv_torch and hsv_to_rgb_torch functions might not handle edge cases perfectly (e.g., when maxc is close to minc). Ensure the deltac > 1e-5 mask works as expected in all cases.
Consider adding comments about assumptions (e.g., expected input ranges) to make it clearer to users.
Noise Addition:

The noise added in add_noise_to_HS uses a normal distribution but doesn't include a random seed, which could make results non-deterministic across runs. Consider adding torch.manual_seed() for reproducibility.
Checkpoint Loading:

The checkpoint loading logic assumes the latest checkpoint is always valid. Adding a fallback for corrupt or incompatible checkpoint files might improve robustness.
Dataset Usage:

The CIFAR-10 dataset is downloaded to the default 'data' directory. It might be useful to make this path configurable.
CUDA Compatibility:

While CUDA is correctly utilized, consider adding a fallback message when CUDA is unavailable, e.g., "Falling back to CPU, training may be slow".
Preview Timing:

Popups for previews (cv2.imshow) can be blocking, depending on the cv2.waitKey() argument. This could disrupt the flow if run on servers without display support. Consider adding a headless mode for environments without GUI support.
Output Format:

The script uses cv2.imshow for previews but doesn't save examples for later reference. Consider saving previews to files, e.g., epoch_1_preview.png.
Epoch and Step Reporting:

For large datasets, printing logs every 100 steps might overwhelm the console. Consider adding an optional verbosity flag.
Documentation of Assumptions:

The note mentions that this is a toy example and may not converge well in a few epochs. Consider adding a section with tips for improving performance, such as using a larger dataset or adjusting hyperparameters.
Efficiency:

Batch-wise RGB-to-HSV and HSV-to-RGB conversions are not optimized for GPU usage, as they use element-wise operations. This may slow down training for larger datasets.
