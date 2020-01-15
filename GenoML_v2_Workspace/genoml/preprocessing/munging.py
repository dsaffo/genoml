# Import the necessary packages 
import subprocess
import pandas as pd

# Importing GenoML specific tools 
from genoml.preprocessing import utils

class munging:
    def __init__(self, pheno_path, addit_path, gwas_path, geno_path, pheno_df, addit_df, gwas_df, run_prefix, impute_type, vif_threshold, iteration, args):
        self.pheno_path = pheno_path
        self.addit_path = addit_path
        self.gwas_path = gwas_path
        self.geno_path = geno_path
        self.pheno_df = pheno_df
        self.addit_df = addit_df
        self.gwas_df = gwas_df
        self.run_prefix = run_prefix
        self.impute_type = impute_type
        self.vif_threshold = vif_threshold
        self.iteration = iteration
        self.args = args

    def plink_inputs(self):
        # Initializing some variables 
        impute_type = self.impute_type
        addit_df = self.addit_df
        pheno_df = self.pheno_df
        
        outfile_h5 = self.run_prefix + ".dataForML.h5"
        pheno_df.to_hdf(outfile_h5, key='pheno', mode = 'w')

        # Set the bashes
        bash1a = "plink --bfile " + self.geno_path + " --indep-pairwise 1000 50 0.05"
        bash1b = "plink --bfile " + self.geno_path + " --extract " + self.run_prefix + ".p_threshold_variants.tab" + " --indep-pairwise 1000 50 0.05"
        bash2 = "plink --bfile " + self.geno_path + " --extract plink.prune.in --make-bed --out temp_genos"
        bash3 = "plink --bfile temp_genos --recode A --out " + self.run_prefix
        bash4 = "cut -f 2,5 temp_genos.bim > " + self.run_prefix + ".variants_and_alleles.tab"
        bash5 = "rm temp_genos.*"
        bash6 = "rm " + self.run_prefix + ".raw"
        bash7 = "rm plink.log"
        bash8 = "rm plink.prune.*"
        bash9 = "rm " + self.run_prefix + ".log"

        # Set the bash command groups
        cmds_a = [bash1a, bash2, bash3, bash4, bash5, bash7, bash8, bash9]
        cmds_b = [bash1b, bash2, bash3, bash4, bash5, bash7, bash8, bash9]

        if (self.gwas_path != "nope") & (self.geno_path != "nope"):
            p_thresh = self.args.p
            gwas_df_reduced = self.gwas_df[['SNP','p']]
            snps_to_keep = gwas_df_reduced.loc[(gwas_df_reduced['p'] <= p_thresh)]
            outfile = self.run_prefix + ".p_threshold_variants.tab"
            snps_to_keep.to_csv(outfile, index=False, sep = "\t")
            print(f"Your candidate variant list prior to pruning is right here: {outfile}.")

        if (self.gwas_path == "nope") & (self.geno_path != "nope"):
            print(f"A list of pruned variants and the allele being counted in the dosages (usually the minor allele) can be found here: {self.run_prefix}.variants_and_alleles.tab")
            for cmd in cmds_a:
                subprocess.run(cmd, shell=True)

        if (self.gwas_path != "nope") & (self.geno_path != "nope"):
            print(f"A list of pruned variants and the allele being counted in the dosages (usually the minor allele) can be found here: {self.run_prefix}.variants_and_alleles.tab")
            for cmd in cmds_b:
                subprocess.run(cmd, shell=True)

        if (self.geno_path != "nope"):
            raw_path = self.run_prefix + ".raw"
            raw_df = pd.read_csv(raw_path, engine = 'c', sep = " ")
            raw_df.drop(columns=['FID','MAT','PAT','SEX','PHENOTYPE'], inplace=True)
            raw_df.rename(columns={'IID':'ID'}, inplace=True)
            subprocess.run(bash6, shell=True)

    # Checking the impute flag and execute
        # Currently only supports mean and median 

        impute_list = ["mean", "median"]

        if impute_type not in impute_list:
            return "The 2 types of imputation currently supported are 'mean' and 'median'"
        elif impute_type.lower() == "mean":
            raw_df = raw_df.fillna(raw_df.mean())
        elif impute_type.lower() == "median":
            raw_df = raw_df.fillna(raw_df.median())
        print("")
        print(f"You have just imputed your genotype features, covering up NAs with the column {impute_type} so that analyses don't crash due to missing data.")
        print("Now your genotype features might look a little better (showing the first few lines of the left-most and right-most columns)...")
        print("#"*70)
        print(raw_df.describe())
        print("#"*70)
        print("")

    # Checking the imputation of non-genotype features 

        if (self.addit_path != "nope"):
            if impute_type not in impute_list:
                return "The 2 types of imputation currently supported are 'mean' and 'median'"
            elif impute_type.lower() == "mean":
                addit_df = addit_df.fillna(addit_df.mean())
            elif impute_type.lower() == "median":
                addit_df = addit_df.fillna(addit_df.median())
            print("")
            print(f"You have just imputed your non-genotype features, covering up NAs with the column {impute_type} so that analyses don't crash due to missing data.")
            print("Now your non-genotype features might look a little better (showing the first few lines of the left-most and right-most columns)...")
            print("#"*70)
            print(addit_df.describe())
            print("#"*70)
            print("")

            # Remove the ID column 
            cols = list(addit_df.columns)
            cols.remove('ID')
            addit_df[cols]

            # Z-scale the features 
            for col in cols:
                if (addit_df[col].min() != 0) & (addit_df[col].max() != 1):
                    addit_df[col] = (addit_df[col] - addit_df[col].mean())/addit_df[col].std(ddof=0)

            print("")
            print("You have just Z-scaled your non-genotype features, putting everything on a numeric scale similar to genotypes.")
            print("Now your non-genotype features might look a little closer to zero (showing the first few lines of the left-most and right-most columns)...")
            print("#"*70)
            print(addit_df.describe())
            print("#"*70)

    # Saving out the proper HDF5 file 
        if (self.geno_path != "nope"):
            merged = raw_df.to_hdf(outfile_h5, key='geno')

        if (self.addit_path != "nope"):
            merged = addit_df.to_hdf(outfile_h5, key='addit')

        if (self.geno_path != "nope") & (self.addit_path != "nope"):
            pheno = pd.read_hdf(outfile_h5, key = "pheno")
            geno = pd.read_hdf(outfile_h5, key = "geno")
            addit = pd.read_hdf(outfile_h5, key = "addit")
            temp = pd.merge(pheno, addit, on='ID', how='inner')
            merged = pd.merge(temp, geno, on='ID', how='inner')
          
        if (self.geno_path != "nope") & (self.addit_path == "nope"):
            pheno = pd.read_hdf(outfile_h5, key = "pheno")
            geno = pd.read_hdf(outfile_h5, key = "geno")
            merged = pd.merge(pheno, geno, on='ID', how='inner')

        if (self.geno_path == "nope") & (self.addit_path != "nope"):
            pheno = pd.read_hdf(outfile_h5, key = "pheno")
            addit = pd.read_hdf(outfile_h5, key = "addit")
            merged = pd.merge(pheno, addit, on='ID', how='inner')

            self.merged = merged  

        return pheno_path, addit_path, gwas_path, geno_path, pheno_df, addit_df, gwas_df, impute_type, merged

    def vif_calculation(self):
    # Initializing some variables 
        iteration = self.iteration
        vif_threshold = self.vif_threshold
        merged = self.plink_inputs

        # No VIF filtering should happen if the user specifies 0 for the VIF threshold/no. of iterations 
            # Merged is not a suitable dataframe? will need to unpack the HDF5 prior to VIF 
        if (iteration==0) or (vif_threshold==0):
            merged.to_hdf(outfile_h5, key="dataForML")
        else:
            df = merged 

            # Save out the IDs to be used later 
            # Save out the phenotypes to be used later

            #IDs = discrete_df['ID']
            #PHENO = discrete_df['PHENO']

            print("Stripping erroneous space, dropping non-numeric columns...") 
            df.columns = df.columns.str.strip()

            print("Drop any rows where at least one element is missing...")
            # Convert any infinite values to NaN prior to dropping NAs
            df.replace([np.inf, -np.inf], np.nan)
            df.dropna(how='any', inplace=True)

            print("Keeping only numerical columns...")
            int_cols = \
                df = df._get_numeric_data()

            print("Checking datatypes...")
            data_type = df.dtypes

            # Subset df to include only relevant numerical types
            int_cols = df.select_dtypes(include=["int", "int16", "int32", "int64", "float",
                                                    "float16", "float32", "float64"]).shape[1]

            print("Sampling 100 rows at random to reduce memory overhead...")
            cleaned_df = df.sample(n=100).copy().reset_index()
            cleaned_df.drop(columns=["index"], inplace=True)

            print("Dropping columns that are not SNPs...")
            cleaned_df.drop(columns=['PHENO'], axis=1, inplace=True) 
            print("Dropped!")

            print("Cleaned!")

            print("")
            print("Shuffling columns...")
            col_names_list = cleaned_df.columns.values.tolist()
            col_names_shuffle = random.sample(col_names_list, len(col_names_list))
            cleaned_df = cleaned_df[col_names_shuffle]
            print("Shuffled!")

            print("Generating chunked, randomized dataframes...")
            chunked_list = [col_names_shuffle[i * chunk_size:(i + 1) * chunk_size] for i in range((len(col_names_shuffle) + chunk_size - 1) // chunk_size)] 
            df_list = []
            for each_list in chunked_list: 
                temp_df = cleaned_df[each_list].astype(float)
                df_list.append(temp_df.copy())

            no_chunks = len(df_list)
            print(f"The number of dataframes you have moving forward is {no_chunks}")
            print("Complete!")
        
            return df_list 


#TODO: Add VIF class 

############################################
### Experimental ###########################
############################################
    

