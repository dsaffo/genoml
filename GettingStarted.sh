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
python GenoMLMunging.py --prefix ./output/test_discrete_geno --geno examples/training --pheno examples/training_pheno.csv
    # FIXME: Need to download PLINK and export to path prior to running this to make the .raw file

# Running the training script
python GenoMLDiscreteSupervised.py --prefix ./examples/test_discrete_geno

# Running the tuning script
python GenoMLDiscreteSupervised.py --prefix ./examples/test_discrete_geno --max-tune 10 --n-cv 3

# Removing a conda virtualenv
conda remove --name genoML --all


###
# Installing from a requirements file using conda
#while read requirement; do conda install --yes $requirement; done < requirements.txt

# Saving out environment requirements to a .txt file
#conda list > requirements.txt