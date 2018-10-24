bsub "python model-features.py -tr 1X,2X,3X -te 4X -m 123-12-gaussian -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2 > result/123_12_gaussian.txt 2>&1"
bsub "python model-features.py -tr 1X,2X,3X -te 4X -m 123-45-gaussian -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 4,5 > result/123_45_gaussian.txt 2>&1"

bsub "python model-features.py -tr 1X,2X,4X -te 3X -m 124-12-gaussian -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2 > result/124_12_gaussian.txt 2>&1"
bsub "python model-features.py -tr 1X,2X,4X -te 3X -m 124-45-gaussian -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 4,5 > result/124_45_gaussian.txt 2>&1"

bsub "python model-features.py -tr 1X,3X,4X -te 2X -m 134-12-gaussian -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2 > result/134_12_gaussian.txt 2>&1"
bsub "python model-features.py -tr 1X,3X,4X -te 2X -m 134-45-gaussian -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 4,5 > result/134_45_gaussian.txt 2>&1"

bsub "python model-features.py -tr 2X,3X,4X -te 1X -m 234-12-gaussian -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2 > result/234_12_gaussian.txt 2>&1"
bsub "python model-features.py -tr 2X,3X,4X -te 1X -m 234-45-gaussian -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 4,5 > result/234_45_gaussian.txt 2>&1"