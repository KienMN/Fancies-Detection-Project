bsub "python combined_model_features.py -d 1X -m 234-12-gaussian,234-345-gaussian -gc 1,2-3,4,5 > result/combined_sum_234_gaussian_12-345.txt 2>&1"
bsub "python combined_model_features.py -d 2X -m 134-12-gaussian,134-345-gaussian -gc 1,2-3,4,5 > result/combined_sum_134_gaussian_12-345.txt 2>&1"
bsub "python combined_model_features.py -d 3X -m 124-12-gaussian,124-345-gaussian -gc 1,2-3,4,5 > result/combined_sum_124_gaussian_12-345.txt 2>&1"
bsub "python combined_model_features.py -d 4X -m 123-12-gaussian,123-345-gaussian -gc 1,2-3,4,5 > result/combined_sum_123_gaussian_12-345.txt 2>&1"

bsub "python combined_model_features.py -d 1X -m 234-123-gaussian,234-45-gaussian -gc 1,2,3-4,5 > result/combined_sum_234_gaussian_123-45.txt 2>&1"
bsub "python combined_model_features.py -d 2X -m 134-123-gaussian,134-45-gaussian -gc 1,2,3-4,5 > result/combined_sum_134_gaussian_123-45.txt 2>&1"
bsub "python combined_model_features.py -d 3X -m 124-123-gaussian,124-45-gaussian -gc 1,2,3-4,5 > result/combined_sum_124_gaussian_123-45.txt 2>&1"
bsub "python combined_model_features.py -d 4X -m 123-123-gaussian,123-45-gaussian -gc 1,2,3-4,5 > result/combined_sum_123_gaussian_123-45.txt 2>&1"