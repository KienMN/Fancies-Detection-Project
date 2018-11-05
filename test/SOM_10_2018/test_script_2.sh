# bsub "python3 model-features.py -tr 1X,2X,3X -te 4X -m 123-1234-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4 > result/123_1234_gaussian.txt 2>&1"
# bsub "python3 model-features.py -tr 1X,2X,3X -te 4X -m 123-1235-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,5 > result/123_1235_gaussian.txt 2>&1"
# bsub "python3 model-features.py -tr 1X,2X,3X -te 4X -m 123-1245-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,4,5 > result/123_1245_gaussian.txt 2>&1"
# bsub "python3 model-features.py -tr 1X,2X,3X -te 4X -m 123-1345-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,3,4,5 > result/123_1345_gaussian.txt 2>&1"
# bsub "python3 model-features.py -tr 1X,2X,3X -te 4X -m 123-2345-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 2,3,4,5 > result/123_2345_gaussian.txt 2>&1"

# bsub "python3 model-features.py -tr 1X,2X,4X -te 3X -m 124-13-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,3 > result/124_13_gaussian.txt 2>&1"
# bsub "python3 model-features.py -tr 1X,2X,4X -te 3X -m 124-245-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 2,4,5 > result/124_245_gaussian.txt 2>&1"

# bsub "python3 model-features.py -tr 1X,3X,4X -te 2X -m 134-13-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,3 > result/134_13_gaussian.txt 2>&1"
# bsub "python3 model-features.py -tr 1X,3X,4X -te 2X -m 134-245-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 2,4,5 > result/134_245_gaussian.txt 2>&1"

# bsub "python3 model-features.py -tr 2X,3X,4X -te 1X -m 234-1234-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4 > result/234_1234_gaussian.txt 2>&1"
# bsub "python3 model-features.py -tr 2X,3X,4X -te 1X -m 234-1235-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,5 > result/234_1235_gaussian.txt 2>&1"
# bsub "python3 model-features.py -tr 2X,3X,4X -te 1X -m 234-1245-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,4,5 > result/234_1245_gaussian.txt 2>&1"
# bsub "python3 model-features.py -tr 2X,3X,4X -te 1X -m 234-1345-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,3,4,5 > result/234_1345_gaussian.txt 2>&1"
# bsub "python3 model-features.py -tr 2X,3X,4X -te 1X -m 234-2345-gaussian -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 2,3,4,5 > result/234_2345_gaussian.txt 2>&1"

bsub "python3 model-features.py -tr 2X,3X,4X -te 1X -m 234-1345-distance-1 -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,3,4,5 > result/234_1345_distance-1.txt 2>&1"
bsub "python3 model-features.py -tr 2X,3X,4X -te 1X -m 234-1345-distance-2 -s 30 -fi 10000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,3,4,5 > result/234_1345_distance-2.txt 2>&1"
bsub "python3 model-features.py -tr 2X,3X,4X -te 1X -m 234-1345-distance-3 -s 30 -fi 30000 -si 10000 -lr 0.75 -sig 3 -n gaussian -c 1,3,4,5 > result/234_1345_distance-3.txt 2>&1"

bsub "python3 model-features.py -tr 2X,3X,4X -te 1X -m 234-all-distance-1 -s 30 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4,5 > result/234_all_distance-1.txt 2>&1"
bsub "python3 model-features.py -tr 2X,3X,4X -te 1X -m 234-all-distance-2 -s 30 -fi 10000 -si 30000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4,5 > result/234_all_distance-2.txt 2>&1"
bsub "python3 model-features.py -tr 2X,3X,4X -te 1X -m 234-all-distance-3 -s 30 -fi 30000 -si 10000 -lr 0.75 -sig 3 -n gaussian -c 1,2,3,4,5 > result/234_all_distance-3.txt 2>&1"