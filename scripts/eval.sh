CONFIG=$1
CHECKPOINT=$2

python tools/train.py -c $CONFIG --load $CHECKPOINT --evaluate-only