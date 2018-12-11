# bsub "python3 random_maps.py -m 123-1345 -n 50 -s 5 -fi 10000 -fe 4000 -si 20000 -se 4000 -lr 0.75 -sig 3 -c 1,2,3,4,5,6 > result/all_random_map_50_5_0,75_3.txt 2>&1"
# bsub "python3 random_maps.py -m 123-1345 -n 100 -s 5 -fi 10000 -fe 4000 -si 20000 -se 4000 -lr 0.75 -sig 3 -c 1,2,3,4,5,6 > result/all_random_map_100_5_0,75_3.txt 2>&1"
# bsub "python3 random_maps.py -m 123-1345 -n 50 -s 4 -fi 10000 -fe 4000 -si 20000 -se 4000 -lr 0.75 -sig 3 -c 1,2,3,4,5,6 > result/all_random_map_50_4_0,75_3.txt 2>&1"
# bsub "python3 random_maps.py -m 123-1345 -n 100 -s 4 -fi 10000 -fe 4000 -si 20000 -se 4000 -lr 0.75 -sig 3 -c 1,2,3,4,5,6 > result/all_random_map_100_4_0,75_3.txt 2>&1"

bsub "python3 random_maps.py -m 123-1345 -n 100 -s 4 -fi 5000 -fe 2500 -si 20000 -se 5000 -lr 0.5 -sig 2 -c 2,3,4,5,6 > result/23456_random_map_100_4_0,5_2.txt 2>&1"
bsub "python3 random_maps.py -m 123-1345 -n 100 -s 4 -fi 5000 -fe 2500 -si 20000 -se 5000 -lr 0.5 -sig 2 -c 1,3,4,5,6 > result/13456_random_map_100_4_0,5_2.txt 2>&1"
bsub "python3 random_maps.py -m 123-1345 -n 100 -s 4 -fi 5000 -fe 2500 -si 20000 -se 5000 -lr 0.5 -sig 2 -c 1,2,4,5,6 > result/12456_random_map_100_4_0,5_2.txt 2>&1"
bsub "python3 random_maps.py -m 123-1345 -n 100 -s 4 -fi 5000 -fe 2500 -si 20000 -se 5000 -lr 0.5 -sig 2 -c 1,2,3,5,6 > result/12356_random_map_100_4_0,5_2.txt 2>&1"
bsub "python3 random_maps.py -m 123-1345 -n 100 -s 4 -fi 5000 -fe 2500 -si 20000 -se 5000 -lr 0.5 -sig 2 -c 1,2,3,4,6 > result/12346_random_map_100_4_0,5_2.txt 2>&1"
bsub "python3 random_maps.py -m 123-1345 -n 100 -s 4 -fi 5000 -fe 2500 -si 20000 -se 5000 -lr 0.5 -sig 2 -c 1,2,3,4,5 > result/12345_random_map_100_4_0,5_2.txt 2>&1"