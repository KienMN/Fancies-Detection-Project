bsub "python3 random_maps.py -tr 1X,2X,3X -te 4X -m 123-1345 -n 50 -s 5 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,3,4,5 > result/123_1345_random_map_50.txt 2>&1"
bsub "python3 random_maps.py -tr 1X,2X,3X -te 4X -m 123-1345 -n 75 -s 4 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,2,3,4,5 > result/123_all_random_map_75.txt 2>&1"
bsub "python3 random_maps.py -tr 1X,2X,3X -te 4X -m 123-1345 -n 75 -s 4 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,3,4,5 > result/123_1345_random_map_75.txt 2>&1"

bsub "python3 random_maps.py -tr 2X,3X,4X -te 1X -m 123-1345 -n 50 -s 5 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,3,4,5 > result/234_1345_random_map_50.txt 2>&1"
bsub "python3 random_maps.py -tr 2X,3X,4X -te 1X -m 123-1345 -n 50 -s 5 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,2,3,4,5 > result/234_all_random_map_50.txt 2>&1"
bsub "python3 random_maps.py -tr 2X,3X,4X -te 1X -m 123-1345 -n 75 -s 4 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,2,3,4,5 > result/234_all_random_map_75.txt 2>&1"
bsub "python3 random_maps.py -tr 2X,3X,4X -te 1X -m 123-1345 -n 75 -s 4 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,3,4,5 > result/234_1345_random_map_75.txt 2>&1"