bsub "python3 lvq_two_phases.py -s 10 -fi 5000 -fe 1000 -si 20000 -se 1000 -lr 0.5 -sig 3 -ne bubble > result/lvq_1.txt 2>&1"
bsub "python3 lvq_two_phases.py -s 10 -fi 5000 -fe 1000 -si 20000 -se 1000 -lr 0.5 -sig 3 -ne gaussian > result/lvq_2.txt 2>&1"

bsub "python3 lvq_two_phases.py -s 15 -fi 5000 -fe 1000 -si 20000 -se 1000 -lr 0.5 -sig 3 -ne bubble > result/lvq_3.txt 2>&1"
bsub "python3 lvq_two_phases.py -s 15 -fi 5000 -fe 1000 -si 20000 -se 1000 -lr 0.5 -sig 3 -ne gaussian > result/lvq_4.txt 2>&1"

bsub "python3 lvq_two_phases.py -s 15 -fi 5000 -fe 1000 -si 20000 -se 1000 -lr 0.25 -sig 3 -ne bubble > result/lvq_5.txt 2>&1"
bsub "python3 lvq_two_phases.py -s 15 -fi 5000 -fe 1000 -si 20000 -se 1000 -lr 0.25 -sig 3 -ne gaussian > result/lvq_6.txt 2>&1"

bsub "python3 lvq_two_phases.py -s 20 -fi 5000 -fe 1000 -si 20000 -se 1000 -lr 0.5 -sig 3 -ne bubble > result/lvq_7.txt 2>&1"
bsub "python3 lvq_two_phases.py -s 20 -fi 5000 -fe 1000 -si 20000 -se 1000 -lr 0.5 -sig 3 -ne gaussian > result/lvq_8.txt 2>&1"