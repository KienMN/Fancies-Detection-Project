bsub "python3 random_maps.py -tr 1X,2X,3X -te 4X -m 123-1345 -n 250 -s 3 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,2,3,4,5 > result/123_all_random_map_150_3.txt 2>&1"
bsub "python3 random_maps.py -tr 1X,2X,3X -te 4X -m 123-1345 -n 250 -s 3 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,3,4,5 > result/123_1345_random_map_150_3.txt 2>&1"
bsub "python3 random_maps.py -tr 1X,2X,3X -te 4X -m 123-1345 -n 300 -s 3 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,2,3,4,5 > result/123_all_random_map_200_3.txt 2>&1"
bsub "python3 random_maps.py -tr 1X,2X,3X -te 4X -m 123-1345 -n 300 -s 3 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,3,4,5 > result/123_1345_random_map_200_3.txt 2>&1"

bsub "python3 random_maps.py -tr 2X,3X,4X -te 1X -m 123-1345 -n 250 -s 3 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,3,4,5 > result/234_1345_random_map_150_3.txt 2>&1"
bsub "python3 random_maps.py -tr 2X,3X,4X -te 1X -m 123-1345 -n 250 -s 3 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,2,3,4,5 > result/234_all_random_map_150_3.txt 2>&1"
bsub "python3 random_maps.py -tr 2X,3X,4X -te 1X -m 123-1345 -n 300 -s 3 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,2,3,4,5 > result/234_all_random_map_200_3.txt 2>&1"
bsub "python3 random_maps.py -tr 2X,3X,4X -te 1X -m 123-1345 -n 300 -s 3 -fi 20000 -si 20000 -lr 0.75 -sig 3 -c 1,3,4,5 > result/234_1345_random_map_200_3.txt 2>&1"