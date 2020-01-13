# Getting Started using conda

# Making a virtual environment
conda create -n genoML python=3.7

# Activating and changing directories to environment
conda activate genoML

# Installing from a requirements file using pip
pip install .

# Saving out environment requirements to a .txt file
#pip freeze > requirements.txt

# Running the munging script
python GenoMLMunging.py --prefix ./output/test_discrete_geno \
--geno examples/training \
--pheno examples/training_pheno.csv

# Running the munging script with VIF filtering 
python GenoMLMunging.py --prefix ./output/test_discrete_geno \
--geno examples/training \
--pheno examples/training_pheno.csv \
--vif 5 \
--iter 1

# Running the training script
python GenoMLDiscreteSupervised.py --prefix ./examples/test_discrete_geno

# Running the tuning script
python GenoMLDiscreteSupervised.py --prefix ./examples/test_discrete_geno --max-tune 10 --n-cv 3

# Removing a conda virtualenv
conda remove --name genoML --all


###
# Using pip to save out environment
#pip freeze > requirements.txt

# Installing from a requirements file using conda
#while read requirement; do conda install --yes $requirement; done < requirements.txt

# Saving out environment requirements to a .txt file
#conda list > requirements.txt

# Exporting the environment to a YAML
#conda env export > environment.yml




### INTERMEDIATE CHECKS FOR MUNGING SCRIPT
# Check impute flag 
python GenoMLMunging.py --prefix ./output/test_discrete_geno \
--geno examples/training \
--impute mean \
--pheno examples/training_pheno.csv

# Check the addit flag and the Z-scale 
python GenoMLMunging.py --prefix ./output/test_discrete_geno_addit \
--geno examples/training \
--impute mean \
--addit examples/training_addit.csv \
--pheno examples/training_pheno.csv

# Check the VIF 
    # FIXME: Still not working!! 
python GenoMLMunging.py --prefix ./output/test_discrete_geno_addit_vif \
--geno examples/training \
--impute mean \
--vif 5 \
--iter 2 \
--addit examples/training_addit.csv \
--pheno examples/training_pheno.csv
