# ACROBAT_submit
This project hosts the main codes for the ACROBAT challenge. We use a weakly supervised algorithm based on deep learning to achieve deformable registration. The pipeline is shown as follows:
* Image preprocess

>To locate tissues in slices, Unet was used to get the bounding boxes of tissues. For H&E WSIs, it is easy to extract tissue masks using color deconvolution. So we performed color deconvolution on H&E WSIs to extract the tissue bounding boxes. And then, they were input into Unet as labels to
 train the model. Finally, bounding boxes of IHC WSIs were generated from Unet. Images were cropped according to bounding boxes.
 
>We counted the rotation angle interval between each pair of images as labels, which are considered the weak supervision information of the algorithm. The angular difference between the images to be aligned was divided into four intervals, which are [0°,90°], [90°,180°], [180°,270°], [270°,360°], with corresponding labels of 0, 1, 2, 3, respectively. The source images were rotated according to the labels. For example, label 0 for 0 degrees, label 1 for  90 degrees, and so on. 

* Affine registration

>A model similar to [1] was used to perform affine registration. The network was pre-trained with the in-house pathological dataset, which includes 8000 pairs. The size of the input images was 512×512 pixels. The loss function was normalized cross-correlation (NCC).

* Deformable registration

>MaskFlownet pre-trained with the Sintel dataset was utilized for deformable registration. The size of the input images was 1024×1024 pixels. The loss functions were NCC and curvature loss.

## Installation of MaskFlownet
The correlation package must be installed first:
```
cd network/MaskFlownet-Pytorch/model/correlation_package
python setup.py install
```
## References

[1] https://github.com/pimed//ProsRegNet

[2] https://github.com/MWod/DeepHistReg/tree/563dd606899b58e9d220133938d25fd293da15d0

[3] https://github.com/cattaneod/MaskFlownet-Pytorch


 
