# Intallation
Install the dependencies with Anaconda and activate the environment with:

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    conda create --name UniDB python=3.9
    conda activate UniDB
    pip install -r requirements.txt

# UniDB++

## Test
1. Prepare datasets.
2. Download pretrained checkpoints [here]
3. Modify options, including dataroot_GT, dataroot_LQ and pretrain_model_G.
4. Choose a model to sample (Default: UniDB (Euler)): test function in `codes/models/denoising_model.py`.
5. In test.yml, setting corresponding parameters:
    + `solver_step` (for UniDB++)
        + possible values: 5, 10, 20, 25, 50, 100;
        + As for Euler methods, we fixed 100 steps.
    + `method` (for UniDB & UniDB++):
        + possible values: euler, noise-solver-1, noise-solver-2, data-solver-1, data-solver-2;
    + `solver_type` (for UniDB & UniDB++):
        + possible values: sde, mean-ode;
    + `gamma` (for UniDB & UniDB++):
        + possible values: 1e6, 1e7, 1e8,...;
        + gamma should be consistent with the pre-trained pth.
6. `python test.py -opt=options/test.yml`

The Test results will be saved in `\results`.

## Train
1. Prepare datasets.
2. Modify options, including dataroot_GT, dataroot_LQ.
3. `python train.py -opt=options/train.yml` for single GPU.<br> `python -m torch.distributed.launch --nproc_per_node=2 --master_port=1111 train.py -opt=options/train.yml --launcher pytorch` for multi GPUs. *Attention: see [Important Option Details](#important-option-details)*.
4. For the DIV2K dataset, your GPU memory needs to be greater than 34GB. 
5. You can modify the parameter of gamma in UniDB/utils/sde_utils.py to balance the control term and the terminal penalty term in the stochastic optimal control, so that the image can achieve better quality.


The Training log will be saved in `\experiments`.

Other tasks can also be written in imitation.

## Important Option Details
* `dataroot_GT`: Ground Truth (High-Quality) data path.
* `dataroot_LQ`: Low-Quality data path.
* `pretrain_model_G`: Pretraind model path.
* `GT_size, LQ_size`: Size of the data cropped during training.
* `niter`: Total training iterations.
* `val_freq`: Frequency of validation during training.
* `save_checkpoint_freq`: Frequency of saving checkpoint during training.
* `gpu_ids`: In multi-GPU training, GPU ids are separated by commas in multi-gpu training.
* `batch_size`: In multi-GPU training, must satisfy relation: *batch_size/num_gpu>1*.

## FID
We provid a brief guidelines for commputing FID of two set of images:

1. Install FID library: `pip install pytorch-fid`.
2. Commpute FID: `python -m pytorch_fid GT_images_file_path generated_images_file_path --batch-size 1`<br>if all the images are the same size, you can remove `--batch-size 1` to accelerate commputing.
