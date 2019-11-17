# Mike's unit testing 

## Munging. from ~/Desktop/GenoMlv2/sandbox. Note, still needs VIF. Also ```export PATH=$PATH:~/Desktop/GenoMLv2/sandbox/```
python  prune_filter_merge_export_fortrainingdataset.py --prefix test_discrete_geno --pheno /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/training_pheno.csv --geno /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/training
python  prune_filter_merge_export_fortrainingdataset.py --prefix test_continuous_geno --pheno /Users/nallsm/Desktop/GenoMLv2/example_data/continuous/training_pheno.csv --geno /Users/nallsm/Desktop/GenoMLv2/example_data/continuous/training
python  prune_filter_merge_export_fortrainingdataset.py --prefix test_discrete_geno_addit --pheno /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/training_pheno.csv --geno /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/training --addit /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/training_addit.csv
python  prune_filter_merge_export_fortrainingdataset.py --prefix test_continuous_geno_addit --pheno /Users/nallsm/Desktop/GenoMLv2/example_data/continuous/training_pheno.csv --geno /Users/nallsm/Desktop/GenoMLv2/example_data/continuous/training --addit /Users/nallsm/Desktop/GenoMLv2/example_data/continuous/training_addit.csv
python  prune_filter_merge_export_fortrainingdataset.py --prefix test_discrete_addit --pheno /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/training_pheno.csv --addit /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/training_addit.csv
python  prune_filter_merge_export_fortrainingdataset.py --prefix test_continuous_addit --pheno /Users/nallsm/Desktop/GenoMLv2/example_data/continuous/training_pheno.csv --addit /Users/nallsm/Desktop/GenoMLv2/example_data/continuous/training_addit.csv
python  prune_filter_merge_export_fortrainingdataset.py --prefix test_discrete_geno_gwas --pheno /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/training_pheno.csv --geno /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/training --gwas /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/example_GWAS.csv
python  prune_filter_merge_export_fortrainingdataset.py --prefix test_continuous_geno_gwas --pheno /Users/nallsm/Desktop/GenoMLv2/example_data/continuous/training_pheno.csv --geno /Users/nallsm/Desktop/GenoMLv2/example_data/continuous/training --gwas /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/example_GWAS.csv
python  prune_filter_merge_export_fortrainingdataset.py --prefix test_discrete_geno_gwas_p --pheno /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/training_pheno.csv --geno /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/training --gwas /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/example_GWAS.csv --p 0.01
python  prune_filter_merge_export_fortrainingdataset.py --prefix test_continuous_geno_gwas_p --pheno /Users/nallsm/Desktop/GenoMLv2/example_data/continuous/training_pheno.csv --geno /Users/nallsm/Desktop/GenoMLv2/example_data/continuous/training --gwas /Users/nallsm/Desktop/GenoMLv2/example_data/discrete/example_GWAS.csv --p 0.01

## Test training
python train_discrete.py --prefix test_discrete_geno_addit
python train_continuous.py --prefix test_continuous_geno_addit
python train_discrete.py --prefix test_discrete_geno_addit --rank-features run
python train_continuous.py --prefix test_continuous_geno_addit --rank-features run

## Test tuning
python tune_discrete.py --prefix test_discrete_geno_addit
python tune_continuous.py --prefix test_continuous_geno_addit
python tune_discrete.py --prefix test_discrete_geno_addit --max-tune 10
python tune_continuous.py --prefix test_continuous_geno_addit --max-tune 10
python tune_discrete.py --prefix test_discrete_geno_addit --max-tune 10 --n-cv 3
python tune_continuous.py --prefix test_continuous_geno_addit --max-tune 10 --n-cv 3