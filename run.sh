DATA_DIR="/home/cxu-serve/p61/RWAVS/scene_4_19/release"
LOG_DIR="./logs"
EXP_NAME="reproduce"
METHOD="anerf"
for i in {1..13}
do
    echo $i
    python main.py --data-root ${DATA_DIR}/$i/ --log-dir ${LOG_DIR}/$i/audio_output/ --output-dir $METHOD/$EXP_NAME/ --conv --lr 5e-4 --max-epoch 100
done

python eval.py --log-dir ${LOG_DIR}/ --output-dir $METHOD/$EXP_NAME/