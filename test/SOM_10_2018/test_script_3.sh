bsub "python combined_model_features.py -d 1X -m 234-123,234-234,234-345 -gc 1,2,3-2,3,4-3,4,5 > result/combined_234_123-234-345.txt 2>&1"
bsub "python combined_model_features.py -d 2X -m 134-123,134-234,134-345 -gc 1,2,3-2,3,4-3,4,5 > result/combined_134_123-234-345.txt 2>&1"
bsub "python combined_model_features.py -d 3X -m 124-123,124-234,124-345 -gc 1,2,3-2,3,4-3,4,5 > result/combined_124_123-234-345.txt 2>&1"
bsub "python combined_model_features.py -d 4X -m 123-123,123-234,123-345 -gc 1,2,3-2,3,4-3,4,5 > result/combined_123_123-234-345.txt 2>&1"