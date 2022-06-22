# About
Codes for the paper "Gradual Domain Adaptation via Normalizing Flows".

# Requirements
Please check the file gdacnf_env.txt.

# Dataset
We can download the Portraits dataset from the following link.
https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0

# Usage
We can execute the experiments with the preprocessed datasets (.pkl files).  
The details of preprocessing are described in our paper and notebook OptUMAP.ipynb.  
The following command executes the training of CNFs. The training of each dataset takes about 24 hours on AWS EC2 g4dn.8xlarge instance.

> python mainCNF.py mnist  
  
> python mainCNF.py portraits

We can execute the experiments of gradual domain adaptation when the training of CNFs has been finished.
> python mainGST.py mnist

> python mainGST.py portraits