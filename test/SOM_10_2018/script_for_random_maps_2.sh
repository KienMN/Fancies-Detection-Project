bsub "python3 random_maps.py -tr 1X,2X,3X -te 4X -m 123-1345 -n 150 -s 5 -fi 100 -si 30000 -lr 0.75 -sig 3 -c 1,2,3,4,5 -fa 0,1,2-1,2,3-2,3,4 -mm 50 > result/123_0,1,2-1,2,3-2,3,4_random_map_150_5.txt 2>&1"
# bsub "python3 random_maps.py -tr 1X,2X,3X -te 4X -m 123-1345 -n 300 -s 3 -fi 100 -si 30000 -lr 0.75 -sig 2 -c 1,2,3,4,5 -fa 0,1,2-1,2,3-2,3,4 -mm 100 > result/123_0,1,2-1,2,3-2,3,4_random_map_300_3.txt 2>&1"

# bsub "python3 random_maps.py -tr 2X,3X,4X -te 1X -m 123-1345 -n 150 -s 5 -fi 100 -si 30000 -lr 0.75 -sig 3 -c 1,2,3,4,5 -fa 0,1,2-1,2,3-2,3,4 -mm 50 > result/234_0,1,2-1,2,3-2,3,4_random_map_150_5.txt 2>&1"
# bsub "python3 random_maps.py -tr 2X,3X,4X -te 1X -m 123-1345 -n 300 -s 3 -fi 100 -si 30000 -lr 0.75 -sig 2 -c 1,2,3,4,5 -fa 0,1,2-1,2,3-2,3,4 -mm 100 > result/234_0,1,2-1,2,3-2,3,4_random_map_300_3.txt 2>&1"