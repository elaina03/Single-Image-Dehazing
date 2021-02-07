DenseFeaturesNet: a single image dehazing network with refined transmission estimation and local atmospheric light prediction

code:
train_final.py
new_model_final.py
loss_final.py
data_final.py
utils.py

model weight for final dehazing results:
exp3_epoch92.pth
exp3_epoch92_separate.pth

train:
python train_final.py --fix_first_conv_blocks --bn_no_track_stats
test:
.ipynb files in final dehazing result folders

exp training dataset:
nyu_ots_haze_uniform_train2 : training dataset
nyu_ots_haze_uniform_val2 : valditation dataset

exp testing datasets:
real_haze: 12 compared real hazy images (in thesis)
SOTS_indoor: 500 synthetic indoor hazy images (in thesis)
SOTS_outdoor: 500 synthetic outdoor hazy images (in thesis)
nyu_ots_haze_uniform_test2 : testing dataset
real_haze_other: other real hazy images

final dehazing result folders:
result0728_exp3_real_haze
result0803_exp3_real_haze_other
result0803_exp3_sots_indoor
result0803_exp3_sots_outdoor

dehazed results for comparing with state-of-the-arts: (in thesis)
ref_papers_with_result ( folder contains 12 compared results)