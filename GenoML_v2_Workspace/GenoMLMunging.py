# Import the necessary packages
import os
import sys
import argparse
import math
import time
import h5py
import joblib
import subprocess
import numpy as np
import pandas as pd

# Additional packages for VIF calculation
import random
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from genoml.preprocessing import utils, munging

if __name__ == "__main__":
    # Create the arguments
    parser = argparse.ArgumentParser(
        description="Arguments for building a training dataset for GenoML.")
    parser.add_argument("--prefix", type=str, default="GenoML_data",
                        help="Prefix for your training data build.")
    parser.add_argument("--geno", type=str, default="nope",
                        help="Genotype: (string file path). Path to PLINK format genotype file, everything before the *.bed/bim/fam [default: nope].")
    parser.add_argument("--addit", type=str, default="nope",
                        help="Additional: (string file path). Path to CSV format feature file [default: nope].")
    parser.add_argument("--pheno", type=str, default="lost",
                        help="Phenotype: (string file path). Path to CSV phenotype file [default: lost].")
    parser.add_argument("--gwas", type=str, default="nope",
                        help="GWAS summary stats: (string file path). Path to CSV format external GWAS summary statistics containing at least the columns SNP and P in the header [default: nope].")
    parser.add_argument("--p", type=float, default=0.001,
                        help="P threshold for GWAS: (some value between 0-1). P value to filter your SNP data on [default: 0.001].")
    parser.add_argument("--vif", type=int, default=0,
                        help="Variance Inflation Factor (VIF): (integer). This is the VIF threshold for pruning non-genotype features. We recommend a value of 5-10. The default of 0 means no VIF filtering will be done. [default: 0].")
    parser.add_argument("--iter", type=int, default=0,
                        help="Iterator: (integer). How many iterations of VIF pruning of features do you want to run. To save time VIF is run in randomly assorted chunks of 1000 features per iteration. The default of 0 means only one pass through the data. [default: 0].")
    parser.add_argument("--impute", type=str, default="median",
                        help="Imputation: (mean, median). Governs secondary imputation and data transformation [default: median].")

    # Process the arguments
    args = parser.parse_args()
    run_prefix = args.prefix

    utils.print_config(args)
    pheno_path, addit_path, gwas_path, geno_path, pheno_df, addit_df, gwas_df = utils.parse_args(args)

    munger = munging(pheno_path, addit_path, gwas_path, geno_path, pheno_df, addit_df, gwas_df, run_prefix, args)

    munger.plinker()

   

    # etc etc
