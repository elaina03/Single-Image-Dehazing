##  DenseFeaturesNet: a single image dehazing network with refined transmission estimation and local atmospheric light prediction

The proposed DenseFeaturesNet exploits the pretrained DenseNet-121 to generate favorable representative features of hazy image, applies two decoders for jointly forecasting the transmission map and global atmospheric light, and finally requires the dehazed image through a refinement module.
Inorder to strength the dehaze ability on remote scene, the method applies a specific synthesis procedure utilizing refined depth images, the WMSE derived from transmission map for loss computation, and a local estimation method for enhancing the applicability of dehazing in real life.

<p align='center'>
<img src="illustrates/network_architecture.png" height="515px" width='691px'>
</div>

### Prerequisites
* python 3
* PyTorch >= 1.0
* numpy
* matplotlib
* tensorboardX(optional)

### Datasets
The synthetic datasets are available from the download link(超連結)
#### Training:
Images synthesized utilizing NYU Depth V2 and OTS of [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/)
nyu_ots_haze_uniform_train2 : training dataset
nyu_ots_haze_uniform_val2 : valditation dataset

#### Testing:
12 benchmark images and SOTS of [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/)
real_haze: 12 compared real hazy images
SOTS_indoor: 500 synthetic indoor hazy images
SOTS_outdoor: 500 synthetic outdoor hazy images
nyu_ots_haze_uniform_test2 : testing dataset
real_haze_other: other real hazy images

### Procedure of Data Synthesis
<p align='center'>
<img src="illustrates/flowchart_of_synthesis.png" height="428px" width='615px'>
</div>

<p align='center'>
<img src="illustrates/refined_depth_map.png" height="388px" width='642px'>
</div>

<p align='center'>
<img src="illustrates/synthetic_hazy_image.png" height="517px" width='632px'>
</div>

### Usage

#### Training:

 ```shell
python train_final.py --fix_first_conv_blocks --bn_no_track_stats
 ```

#### Testing:
The pretrained model is available from [the download link](超連結)
*Put  the pretrained weight in the `current` folder.*
*.ipynb files in final dehazing result folders(global and local estimation method)*

<p align='center'>
<img src="illustrates/global_local_estimation.png" height="648px" width='638px'>
</div>

### Evaluation on Real Hazy Images
*The results of previous works are generated from the programs provided by the authors. For the paper without a given code, the program written by a third party will be executed for evaluation. Thus, the comparison results may different from the results directly produced by the authors.*

#### Result folders:
result0728_exp3_real_haze
result0803_exp3_real_haze_other
result0803_exp3_sots_indoor
result0803_exp3_sots_outdoor
#### Dehazed results for comparing with state-of-the-arts:
ref_papers_with_result ( folder contains 12 compared results)
[link](超連結)

<p align='center'>
<img src="illustrates/women.png" height="400px" width='631px'>
</div>

<p align='center'>
<img src="illustrates/yosemite2.png" height="578px" width='606px'>
</div>

<p align='center'>
<img src="illustrates/lviv.png" height="735px" width='451px'>
</div>

<p align='center'>
<img src="illustrates/manhattan2.png" height="735px" width='447px'>
</div>

<p align='center'>
<img src="illustrates/yosemite1.png" height="695px" width='449px'>
</div>

<p align='center'>
<img src="illustrates/landscape.png" height="745px" width='442px'>
</div>

### Evaluation on Synthetic Hazy Images

<p align='center'>
<img src="illustrates/sots_configurations.png" height="261px" width='421px'>
</div>

<p align='center'>
<img src="illustrates/sots_psnr_ssim.png" height="438px" width='349px'>
</div>

<p align='center'>
<img src="illustrates/sots_dehazed.png" height="422px" width='430px'>
</div>




