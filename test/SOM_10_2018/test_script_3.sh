bsub "python combined_model_features.py -d 1X -m 234-13-gaussian,234-245-gaussian -gc 1,3-2,4,5 > result/combined_sum_234_gaussian_13-245.txt 2>&1"
bsub "python combined_model_features.py -d 2X -m 134-13-gaussian,134-245-gaussian -gc 1,3-2,4,5 > result/combined_sum_134_gaussian_13-245.txt 2>&1"
bsub "python combined_model_features.py -d 3X -m 124-13-gaussian,124-245-gaussian -gc 1,3-2,4,5 > result/combined_sum_124_gaussian_13-245.txt 2>&1"
bsub "python combined_model_features.py -d 4X -m 123-13-gaussian,123-245-gaussian -gc 1,3-2,4,5 > result/combined_sum_123_gaussian_13-245.txt 2>&1"

# bsub "python combined_model_features.py -d 1X -m 234-123-gaussian,234-45-gaussian -gc 1,2,3-4,5 > result/combined_sum_234_gaussian_123-45.txt 2>&1"
# bsub "python combined_model_features.py -d 2X -m 134-123-gaussian,134-45-gaussian -gc 1,2,3-4,5 > result/combined_sum_134_gaussian_123-45.txt 2>&1"
# bsub "python combined_model_features.py -d 3X -m 124-123-gaussian,124-45-gaussian -gc 1,2,3-4,5 > result/combined_sum_124_gaussian_123-45.txt 2>&1"
# bsub "python combined_model_features.py -d 4X -m 123-123-gaussian,123-45-gaussian -gc 1,2,3-4,5 > result/combined_sum_123_gaussian_123-45.txt 2>&1"