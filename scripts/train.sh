DT=`date "+%Y-%m-%d_%H:%M:%S"`
EXP_PTH=$1
RUN=$2-${DT}
GPU=$3
EXP=${EXP_PTH#exp*/}
EXP=${EXP%.*}
export CUDA_VISIBLE_DEVICES=$GPU

LOG_DIR=logs/$EXP/

if [ ! -d "$LOG_DIR" ]; then
  echo "Creating directory $LOG_DIR"
  mkdir -p $LOG_DIR
fi

CMD="python train.py --config ${EXP_PTH} --run ${RUN}"
LOG_FILE="$LOG_DIR/${RUN}.log"
nohup $CMD > $LOG_FILE 2>&1 &

sleep 1

tail -f $LOG_FILE