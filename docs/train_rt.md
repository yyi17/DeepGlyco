# DeepGlyco Tutorial: Training New Models for iRT
Training new models for glycopeptide iRT prediction.

## Prerequisites
### System Requirements
In this tutorial, model training has been tested on a workstation with Intel Core i9-12900K CPU, 64 GB RAM, and Microsoft Windows 10 Version 22H2 (OS Build 19045.2604) operating system. A NVIDIA GeForce RTX 3090 GPU is needed, with CUDA Version 11.6.

### Software Dependency
The following software and packages are required:
- Python (version 3.9.16, [Anaconda](https://www.anaconda.com/) distribution is recommended)
- [PyTorch](https://pytorch.org/) (version 1.12.1)
- [DGL](https://www.dgl.ai/) (version 1.0.1)
- numpy (version 1.23.5)
- pandas (version 1.5.2)
- scipy (version 1.10.1)
- scikit-learn (version 1.2.1)
- statsmodels (version 0.13.2)
- h5py (version 3.7.0)
- pymzml (version 2.5.2)
- matplotlib (version 3.6.2)
- tensorboard (version 2.10.0)

Later versions may be compatible, but have not been tested.

## Tutorial Data
### Starting Materials
Starting materials of this tutorial are available at ProteomeXchange and iProX with identifier [`PXD045248`](http://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD045248) or [`IPX0007075004`](https://www.iprox.cn/page/project.html?id=IPX0007075004).


A StrucGP search result file:
- MouseFiveTissues_result.xlsx

Peptide retention time extracted from Byonic search results:
- MouseFiveTissues_Byonic.zip
- PXD031032_MouseFiveTissues_Byonic_peptides.rt.csv

A pretrained model for peptide iRT prediction:
- PXD004452_peprt_model.pt


## Train a iRT model
### Prepare Training Data
Create a directory for the project, and place the files in the folder.

Open Anaconda PowerShell Prompt, set the project folder as working directory, and set path to the scripts as global parameter.
``` powershell
cd "Path_to_project_data"
$script_path = "Path_to_DeepGlyco\src"
```

#### Retention Time Extraction
Create a subdirectory named `gprt` in the project directory. Extract experimental retention time values from the StrucGP result.
``` powershell
mkdir "gprt" -f | Out-Null

python "$script_path\build_gprtlib.py" `
--in "mzML\MouseFiveTissues_result.xlsx" `
--out "gprt\PXD025859_MouseFiveTissues_StrucGP.rtlib.h5"
```
A subdirectory named `gprt` is created in the project directory. The extracted retention time values are imported into a HDF5 file (`*.rtlib.h5`).

#### iRT Calibration
Predict iRT of the peptides identified in the same experiment.
``` powershell
python "$script_path\predict_rt.py" `
--no-validate `
--model "PXD004452_peprt_model.pt" `
--in "PXD031032_MouseFiveTissues_Byonic_peptides.rt.csv" `
--out "PXD031032_MouseFiveTissues_Byonic_peptides_prediction.rt.csv"
```

These peptides are used as anchors for iRT calibration of glycopeptides. Combine replicate iRT entries into consensus entries (one entry per glycopeptide).
``` powershell
python "$script_path\build_consensus_gprtib.py" `
--in "gprt\PXD025859_MouseFiveTissues_StrucGP.rtlib.h5" `
--reference "PXD031032_MouseFiveTissues_Byonic_peptides_prediction.rt.csv" `
--out "gprt\PXD025859_MouseFiveTissues_StrucGP_consensus.rtlib.h5"
```
The consensus library is saved in a HDF5 file (`*_consensus.rtlib.h5`), which will be used as training data.

### Model Training
Create a subdirectory named `training` in the project directory, and place the pretrained model in the project folder:
- training\PXD004452_peprt_model.pt

Start model training.
``` powershell
python "$script_path\train_gprt.py" `
--in "gprt\PXD025859_MouseFiveTissues_StrucGP_consensus.rtlib.h5" `
--wkdir "training\deepgprt\PXD025859_MouseFiveTissues" `
--pretrained "training\PXD004452_peprt_model.pt"
```
The trained model checkpoints are saved in Python binary files (`epoch_*.pt`) in the path `training\deepgprt\PXD025859_MouseFiveTissues\checkpoints`.

Open another Anaconda PowerShell Prompt, start TensorBoard to monitor the training process.
``` powershell
tensorboard --logdir=training\deepgprt\PXD025859_MouseFiveTissues
```
Select the best model based on the metrics on training and validation set.

### Model Testing
Test the trained model.
``` powershell
python "$script_path\predict_gprt.py" `
--validate `
--pretrained "training\deepgprt\PXD025859_MouseFiveTissues\checkpoints\epoch_265.pt" `
--in "gprt\PXD025859_MouseFiveTissues_StrucGP_consensus.rtlib.h5" `
--out "training\deepgprt\PXD025859_MouseFiveTissues\PXD025859_MouseFiveTissues_StrucGP_consensus_prediction.rt.csv"
```
Note that `epoch_265.pt` should be replaced with the model selected for testing. The `--in` argument should be changed when testing on other data (`*.rtlib.h5`).

The predicted and target data are saved as a CSV file(`*_prediction.rt.csv`).

### Model finetuning
When finetuning a pretrained model with other data, change the config and epoch number.
``` powershell
python "$script_path\train_gprt.py" `
--in "gprt\FinetuningData_consensus.rtlib.h5" `
--wkdir "training\deepgprt\FinetuningData" `
--pretrained "training\deepgprt\PXD025859_MouseFiveTissues\checkpoints\epoch_265.pt" `
--config "$script_path\deepglyco\deeplib\gpep\rt\gprtmodel_finetune.yaml" `
--epochs 50
```
