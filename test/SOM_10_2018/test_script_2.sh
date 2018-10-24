bsub "python model-features.py -tr 1X,2X,3X -te 4X -m 123-13-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,3 > result/123_13_gaussian.txt 2>&1"
bsub "python model-features.py -tr 1X,2X,3X -te 4X -m 123-245-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 2,4,5 > result/123_245_gaussian.txt 2>&1"

bsub "python model-features.py -tr 1X,2X,4X -te 3X -m 124-13-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,3 > result/124_13_gaussian.txt 2>&1"
bsub "python model-features.py -tr 1X,2X,4X -te 3X -m 124-245-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 2,4,5 > result/124_245_gaussian.txt 2>&1"

bsub "python model-features.py -tr 1X,3X,4X -te 2X -m 134-13-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,3 > result/134_13_gaussian.txt 2>&1"
bsub "python model-features.py -tr 1X,3X,4X -te 2X -m 134-245-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 2,4,5 > result/134_245_gaussian.txt 2>&1"

bsub "python model-features.py -tr 2X,3X,4X -te 1X -m 234-13-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,3 > result/234_13_gaussian.txt 2>&1"
bsub "python model-features.py -tr 2X,3X,4X -te 1X -m 234-245-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 2,4,5 > result/234_245_gaussian.txt 2>&1"