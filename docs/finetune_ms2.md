# DeepGlyco Tutorial: Finetuning Models with New Data
Finetuning new models for glycopeptide MS/MS prediction using users' data.

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
Starting materials of this tutorial are available at ProteomeXchange and iProX with identifier [`PXD045248`](http://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD045248) or [`IPX0007075001`](https://www.iprox.cn/page/project.html?id=IPX0007075001).

MS/MS data in mzML format (6 files):
- 480_20210524_WZY_IGP_HILIC_DDACancer_*.mzML.gz
- 480_20210714_WZY_NIGP_HILIC_DDA_Cancer_*.mzML.gz

StrucGP search result files:
- 480_20210524_WZY_IGP_HILIC_DDACancer_*_result_StrucGP.xlsx
- 480_20210714_WZY_NIGP_HILIC_DDA_Cancer_*_result_StrucGP.xlsx

A pretrained model for glycopeptide MS/MS prediction:
- PXD026649_PXD030804_HumanSperm_gpms2_model.pt


## Finetune a MS/MS model
### Prepare Training Data
Create a directory for the project.

Create a subdirectory named `mzML` in the project directory, and place the mzML files and StrucGP reports in the subfolder:
- mzML\*.mzML.gz
- mzML\*_result.xlsx

Open Anaconda PowerShell Prompt, set the project folder as working directory, and set path to the scripts as global parameter.
``` powershell
cd "Path_to_project_data"
$script_path = "Path_to_DeepGlyco\src"
```

#### Spectrum Annotation
Create a subdirectory named `gpms2` in the project directory. Extract and annotate identified MS/MS spectra from the StrucGP result.
``` powershell
mkdir "gpms2" -f | Out-Null

ls mzML *_result.xlsx -Name | % {
    python src\build_gpspeclib.py `
    --in "mzML\$_" `
    --spectra "mzML\$($_.Replace('_StrucGP_result.xlsx', '.mzML'))" `
    --out gpms2\PXD031025_HumanSerum_StrucGP.speclib.h5
}
```
A subdirectory named `gpms2` is created in the project directory. The extracted and annotated spectra are imported into a HDF5 file (`*.speclib.h5`).

#### Building Consensus Spectra
Combine replicate spectra into consensus spectra (one spectrum per glycopeptide precursor).
``` powershell
python "$script_path\build_consensus_gpspeclib.py" `
--in "gpms2\PXD031025_HumanSerum_StrucGP.speclib.h5" `
--out "gpms2\PXD031025_HumanSerum_StrucGP_consensus.speclib.h5"
```
The consensus spectra are saved in a HDF5 file (`*_consensus.speclib.h5`), which will be used as training data.

The dataset contains ~3000 glycopeptide precursors. Users should check whether the data size is enough for model finetuning when using their own datasets.

### Model finetuning
#### Testing the Pretrained Model
Before starting finetuning, users can test the pretrained model.
``` powershell
python "$script_path\validate_gpms2.py" `
--model "PXD026649_PXD030804_HumanSperm_gpms2_model.pt" `
--in "gpms2\PXD031025_HumanSerum_StrucGP_consensus.speclib.h5" `
--out "gpms2\PXD031025_HumanSerum_StrucGP_consensus_prediction_pretrained.ms2score.csv"
```
The performance of the pretrained model on the dataset (spectral similarites of predicted and target data) are saved as a CSV file(`*_prediction_pretrained.ms2score.csv`) and visualized in a SVG file.

#### Model finetuning
When finetuning a pretrained model with other data, specify the config and epoch number parameters.
``` powershell
python "$script_path\train_gpms2.py" `
--in "gpms2\PXD031025_HumanSerum_StrucGP_consensus.speclib.h5" `
--wkdir "training\deepgpms2\PXD031025_HumanSerum" `
--pretrained "PXD026649_PXD030804_HumanSperm_gpms2_model.pt" `
--config "$script_path\deepglyco\deeplib\gpep\ms2\gpms2model_finetune.yaml" `
--epochs 20
```
Finetuning the model on the dataset with 20 epochs will take ~15 min.

#### Model selection
The trained model checkpoints will be saved in Python binary files (`epoch_*.pt`) in the path `training\deepgpms2\PXD031025_HumanSerum\checkpoints`.

Optionally, TensorBoard can be used to monitor the training process. Open another Anaconda PowerShell Prompt, start TensorBoard.
``` powershell
tensorboard --logdir=training\deepgpms2\PXD025859_MouseFiveTissues
```
Select the best model based on the metrics (epoch with the minimal sa_total) on validation set.

#### Testing the Finetuned Model
Test the finetuned model.
``` powershell
python "$script_path\validate_gpms2.py" `
--model "training\deepgpms2\PXD031025_HumanSerum\checkpoints\epoch_X.pt" `
--in "gpms2\PXD031025_HumanSerum_StrucGP_consensus.speclib.h5" `
--out "gpms2\PXD031025_HumanSerum_StrucGP_consensus_prediction_finetuned.ms2score.csv"
```
Note that `epoch_X.pt` should be replaced with the model selected for testing.

The performance of the pretrained model on the dataset (spectral similarites of predicted and target data) are saved as a CSV file(`*_prediction_finetuned.ms2score.csv`) and visualized in a SVG file.

Ideally, the median spectral angle loss (SA) will be improved from >0.3 to ~0.2 after finetuning, corresponding to dot product (DP) spectral similarity from ~0.8 to >0.9.
