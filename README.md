# Gradual Domain Adaptation via Normalizing Flows
Codes for the paper "Gradual Domain Adaptation via Normalizing Flows".

## Requirements
Please check the file named gdacnf_env.yml.  
This file is to create the execution environment.  
>conda env create -n your-env-name -f gdacnf_env.yml

## Usage
Our experiments consist of three steps as follows.
1. Fit UMAP
2. Train CNF
3. Apply Gradual Domain Adaptation  

For starting experiments from 1st step, please download the datasets from the links listed in [the Datasets section](#Datasets).  
After downloading the datasets, run FitUMAP.py to obtain preprocessed datasets.  
Since downloading takes a very long time, it is recommended to use preprocessed datasets in this supplementary material.  
  
Running `runCNF.sh` conducts the CNF training.  
We conduct the experiments on AWS EC2 g4dn.8xlarge instance and the training of CNF take about one day per dataset.  
Since we publish the results of CNF training as .tar files in this supplementary material, this step can be skipped.  

Running `runGST.sh` conducts application of gradual domain adaptation.  
The results of baseline methods are obtained by running `runBaseline.sh` script.  

Lastly, we can use `makeFigure.py` to parse the experimental results and obtain the figure shown in our papers.  

<a id="Datasets"></a>
## Datasets
Portraits  
https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0

SHIFT15M  
https://github.com/st-tech/zozo-shift15m  

RxRx1  
We use WILDS to load pre-processed dataset.  
https://wilds.stanford.edu/datasets/  

Tox21  
We use MoleculeNet to load pre-processed dataset.  
https://moleculenet.org/
