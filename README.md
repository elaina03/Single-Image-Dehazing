##  DenseFeaturesNet: a single image dehazing network with refined transmission estimation and local atmospheric light prediction

&4bsp; &nbsp;  The proposed DenseFeaturesNet exploits the pretrained DenseNet-121 to generate favorable representative features of hazy image, applies two decoders for jointly forecasting the transmission map and global atmospheric light, and finally requires the dehazed image through a refinement module.<br /><br />
&nbsp; &nbsp;  Inorder to strength the dehaze ability on remote scene, the method applies a specific synthesis procedure utilizing refined depth images, the WMSE derived from transmission map for loss computation, and a local estimation method for enhancing the applicability of dehazing in real life.

<p align='center'>
  <img src="illustrations/network_architecture.PNG" height='515' width='691'>
</p>


### Prerequisites
* python 3
* PyTorch >= 1.0
* numpy
* matplotlib
* tensorboardX(optional)

### Datasets
The synthetic datasets are available from the download link(超連結)
#### Training:
Images synthesized utilizing NYU Depth V2 and OTS of [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/)<br />
* nyu_ots_haze_uniform_train2 : training dataset
* nyu_ots_haze_uniform_val2 : valditation dataset

#### Testing:
12 benchmark images and SOTS of [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/)<br />
* real_haze: 12 compared real hazy images
* SOTS_indoor: 500 synthetic indoor hazy images
* SOTS_outdoor: 500 synthetic outdoor hazy images
* nyu_ots_haze_uniform_test2 : testing dataset
* real_haze_other: other real hazy images

### Procedure of Data Synthesis
<p align='center'>
  <img src="illustrations/flowchart_of_synthesis.PNG" height='428' width='615'>
</p>
 
<p align='center'>
  <img src="illustrations/refined_depth_map.PNG" height='388' width='642'>
</p>

<p align='center'>
  <img src="illustrations/synthetic_hazy_image.PNG" height='517' width='632'>
</p>

### Usage

#### Training:

 ```shell
python train_final.py --fix_first_conv_blocks --bn_no_track_stats
 ```

#### Testing:
* The pretrained model is available from [the download link](超連結)<br />
Put  the pretrained weight in the `current` folder.<br />
.ipynb files in final dehazing result folders(global and local estimation method)<br />

<p align='center'>
  <img src="illustrations/global_local_estimation.PNG" height='648' width='638'>
</p>

### Evaluation on Real Hazy Images
&nbsp; &nbsp;  The results of previous works are generated from the programs provided by the authors.<br />
For the paper without a given code, the program written by a third party will be executed for evaluation.<br />
Thus, the comparison results may different from the results directly produced by the authors.

#### Result folders:
* result0728_exp3_real_haze
* result0803_exp3_real_haze_other
* result0803_exp3_sots_indoor
* result0803_exp3_sots_outdoor
#### Dehazed results for comparing with state-of-the-arts:
* ref_papers_with_result ( folder contains 12 compared results)
[link](超連結)

<p align='center'>
  <img src="illustrations/women.PNG" height='400' width='631'>
</p>

<p align='center'>
  <img src="illustrations/yosemite2.PNG" height='578' width='606'>
</p>

<p align='center'>
  <img src="illustrations/lviv.PNG" height='735' width='451'>
</p>

<p align='center'>
  <img src="illustrations/manhattan2.PNG" height='735' width='447'>
</p>

<p align='center'>
  <img src="illustrations/yosemite1.PNG" height='695' width='449'>
</p>

<p align='center'>
  <img src="illustrations/landscape.PNG" height='745' width='442'>
</p>

### Evaluation on Synthetic Hazy Images

<p align='center'>
  <img src="illustrations/sots_configurations.PNG" height='261' width='421'>
</p>

<p align='center'>
  <img src="illustrations/sots_psnr_ssim.PNG" height='438' width='349'>
</p>

<p align='center'>
  <img src="illustrations/sots_dehazed.PNG" height='422' width='430'>
</p>




