bsub "python model-features.py -tr 1X,2X,3X -te 4X -m 123-123 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 1,2,3 > result/123_123_bubble.txt 2>&1"
bsub "python model-features.py -tr 1X,2X,3X -te 4X -m 123-234 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 2,3,4 > result/123_234_bubble.txt 2>&1"
bsub "python model-features.py -tr 1X,2X,3X -te 4X -m 123-345 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 3,4,5 > result/123_345_bubble.txt 2>&1"

bsub "python model-features.py -tr 1X,2X,4X -te 3X -m 124-123 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 1,2,3 > result/124_123_bubble.txt 2>&1"
bsub "python model-features.py -tr 1X,2X,4X -te 3X -m 124-234 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 2,3,4 > result/124_234_bubble.txt 2>&1"
bsub "python model-features.py -tr 1X,2X,4X -te 3X -m 124-345 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 3,4,5 > result/124_345_bubble.txt 2>&1"

bsub "python model-features.py -tr 1X,3X,4X -te 2X -m 134-123 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 1,2,3 > result/134_123_bubble.txt 2>&1"
bsub "python model-features.py -tr 1X,3X,4X -te 2X -m 134-234 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 2,3,4 > result/134_234_bubble.txt 2>&1"
bsub "python model-features.py -tr 1X,3X,4X -te 2X -m 134-345 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 3,4,5 > result/134_345_bubble.txt 2>&1"

bsub "python model-features.py -tr 2X,3X,4X -te 1X -m 234-123 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 1,2,3 > result/234_123_bubble.txt 2>&1"
bsub "python model-features.py -tr 2X,3X,4X -te 1X -m 234-234 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 2,3,4 > result/234_234_bubble.txt 2>&1"
bsub "python model-features.py -tr 2X,3X,4X -te 1X -m 234-345 -s 40 -fi 30000 -si 30000 -lr 0.75 -sig 3 -n bubble -c 3,4,5 > result/234_345_bubble.txt 2>&1"