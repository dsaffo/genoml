import subprocess
import pandas as pd

class munging:
    def __init__(self, pheno_path, addit_path, gwas_path, geno_path, pheno_df, addit_df, gwas_df, run_prefix, args):
        self.pheno_path = pheno_path
        self.addit_path = addit_path
        self.gwas_path = gwas_path
        self.geno_path = geno_path
        self.pheno_df = pheno_df
        self.addit_df = addit_df
        self.gwas_df = gwas_df
        self.run_prefix = run_prefix
        self.args = args

    def plink_inputs(self):
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
            print("Your candidate variant list prior to pruning is right here", outfile, ".")

        if (self.gwas_path == "nope") & (self.geno_path != "nope"):
            print("A list of pruned variants and the allele being counted in the dosages (usually the minor allele) can be found here ", self.run_prefix + ".variants_and_alleles.tab.")
            for cmd in cmds_a:
                subprocess.run(cmd, shell=True)

        if (self.gwas_path != "nope") & (self.geno_path != "nope"):
            print("A list of pruned variants and the allele being counted in the dosages (usually the minor allele) can be found here", self.run_prefix + ".variants_and_alleles.tab.")
            for cmd in cmds_b:
                subprocess.run(cmd, shell=True)

        if (self.geno_path != "nope"):
            raw_path = self.run_prefix + ".raw"
            raw_df = pd.read_csv(raw_path, engine = 'c', sep = " ")
            raw_df.drop(columns=['FID','MAT','PAT','SEX','PHENOTYPE'], inplace=True)
            raw_df.rename(columns={'IID':'ID'}, inplace=True)
            subprocess.run(bash6, shell=True)

#class vif:
# Have separate functions for all these things 


# parser.add_argument("--vif", type=int, default=0,
# help="Variance Inflation Factor (VIF): (integer).
# This is the VIF threshold for pruning non-genotype features.
# We recommend a value of 5-10. 
# The default of 0 means no VIF filtering will be done. [default: 0].")
      