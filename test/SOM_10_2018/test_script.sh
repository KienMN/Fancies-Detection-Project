bsub "python model-features.py -tr 1X,2X,3X -te 4X -m all-feature-123 -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 4 -n bubble -c 1,2,3,4,5 > result/all_feature_123_bubble.txt 2>&1"
bsub "python model-features.py -tr 1X,2X,4X -te 3X -m all-feature-124 -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 4 -n bubble -c 1,2,3,4,5 > result/all_feature_124_bubble.txt 2>&1"
bsub "python model-features.py -tr 1X,3X,4X -te 2X -m all-feature-134 -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 4 -n bubble -c 1,2,3,4,5 > result/all_feature_134_bubble.txt 2>&1"
bsub "python model-features.py -tr 2X,3X,4X -te 1X -m all-feature-234 -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 4 -n bubble -c 1,2,3,4,5 > result/all_feature_234_bubble.txt 2>&1"

bsub "python model-features.py -tr 1X,2X,3X -te 4X -m all-feature-123 -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4,5 > result/all_feature_123_gaussian.txt 2>&1"
bsub "python model-features.py -tr 1X,2X,4X -te 3X -m all-feature-124 -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4,5 > result/all_feature_124_gaussian.txt 2>&1"
bsub "python model-features.py -tr 1X,3X,4X -te 2X -m all-feature-134 -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4,5 > result/all_feature_134_gaussian.txt 2>&1"
bsub "python model-features.py -tr 2X,3X,4X -te 1X -m all-feature-234 -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4,5 > result/all_feature_234_gaussian.txt 2>&1"