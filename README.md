# ACROBAT_submit
This project hosts the main codes for the ACROBAT challenge. We use a weakly supervised algorithm based on deep learning to achieve deformable registration. The pipeline of ia shown as follows:
* Image preprocess

>To avoid locate tissues in slices, Unet was used to get the bounding boxes of tissues. For H&E WSIs, it is easy to extract tissue masks using color deconvolution. So we performed color deconvolution on H&e WSIs to extract tissue bounding box. And then, they were input into Unet as labels to
 train the model. Finally, bounding boxes of IHC WSIs were generated from Unet. Images were croped accroding to bounding boxes.
 
>We count the rotation angle interval between each pair of images as labels, which are considered as the weak supervision information of the algorithm.The angular difference between the images to be aligned is divided into four intervals, which are [0°,90°], [90°,180°], [180°,270°], [270°,360°], with corresponding labels of 0, 1, 2, 3, respectively. Then sources images were rotated according to the labels. For eaxmple, label 0 for 0 degrees, label 1 for  90 degrees, and so on. 

* Affine registration

>A model similar to [1] was used to performed affine registration. The network was pre-trained with in-house pathologucal dataset, which includes 8000 pairs. The size of input images was 512×512 pixels. The loss function was normalized cross-correlation (NCC).

* Deformable registration

>MaskFlownet pre-trained with the FlyingChair dataset was used for deformable registration. The size of input images was 1024×1024 pixels. The loss function were NCC and the curvature loss.



 
