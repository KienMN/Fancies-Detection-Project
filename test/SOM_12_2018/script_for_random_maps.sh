# bsub "python3 random_maps.py -m 123-1345 -n 50 -s 5 -fi 10000 -fe 4000 -si 20000 -se 4000 -lr 0.75 -sig 3 -c 1,2,3,4,5,6 > result/all_random_map_50_5_0,75_3.txt 2>&1"
# bsub "python3 random_maps.py -m 123-1345 -n 100 -s 5 -fi 10000 -fe 4000 -si 20000 -se 4000 -lr 0.75 -sig 3 -c 1,2,3,4,5,6 > result/all_random_map_100_5_0,75_3.txt 2>&1"
# bsub "python3 random_maps.py -m 123-1345 -n 50 -s 4 -fi 10000 -fe 4000 -si 20000 -se 4000 -lr 0.75 -sig 3 -c 1,2,3,4,5,6 > result/all_random_map_50_4_0,75_3.txt 2>&1"
# bsub "python3 random_maps.py -m 123-1345 -n 100 -s 4 -fi 10000 -fe 4000 -si 20000 -se 4000 -lr 0.75 -sig 3 -c 1,2,3,4,5,6 > result/all_random_map_100_4_0,75_3.txt 2>&1"

bsub "python3 random_maps.py -m 123-1345 -n 50 -s 3 -fi 5000 -fe 2500 -si 20000 -se 5000 -lr 0.75 -sig 3 -c 1,2,3,4,5,6 > result/all_random_map_50_3_0,75_3.txt 2>&1"
bsub "python3 random_maps.py -m 123-1345 -n 75 -s 3 -fi 5000 -fe 2500 -si 20000 -se 5000 -lr 0.75 -sig 3 -c 1,2,3,4,5,6 > result/all_random_map_75_3_0,75_3.txt 2>&1"
bsub "python3 random_maps.py -m 123-1345 -n 100 -s 3 -fi 5000 -fe 2500 -si 20000 -se 5000 -lr 0.75 -sig 3 -c 1,2,3,4,5,6 > result/all_random_map_100_3_0,75_3.txt 2>&1"