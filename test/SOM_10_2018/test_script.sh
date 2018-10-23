bsub "python model-features.py -tr 1X,2X,3X -te 4X -m all-feature-123_bubble -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 5 -n bubble -c 1,2,3,4,5 > result/all_feature_123_4040_bubble.txt 2>&1"
bsub "python model-features.py -tr 1X,2X,4X -te 3X -m all-feature-124_bubble -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 5 -n bubble -c 1,2,3,4,5 > result/all_feature_124_4040_bubble.txt 2>&1"
bsub "python model-features.py -tr 1X,3X,4X -te 2X -m all-feature-134_bubble -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 5 -n bubble -c 1,2,3,4,5 > result/all_feature_134_4040_bubble.txt 2>&1"
bsub "python model-features.py -tr 2X,3X,4X -te 1X -m all-feature-234_bubble -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 5 -n bubble -c 1,2,3,4,5 > result/all_feature_234_4040_bubble.txt 2>&1"

# bsub "python model-features.py -tr 1X,2X,3X -te 4X -m all-feature-123_gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4,5 > result/all_feature_123_gaussian.txt 2>&1"
# bsub "python model-features.py -tr 1X,2X,4X -te 3X -m all-feature-124_gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4,5 > result/all_feature_124_gaussian.txt 2>&1"
# bsub "python model-features.py -tr 1X,3X,4X -te 2X -m all-feature-134_gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4,5 > result/all_feature_134_gaussian.txt 2>&1"
# bsub "python model-features.py -tr 2X,3X,4X -te 1X -m all-feature-234_gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4,5 > result/all_feature_234_gaussian.txt 2>&1"