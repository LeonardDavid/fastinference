#/bin/bash

set -e

FIOPTIONS=""

for i in "$@"
do
case $i in
    --outpath=*)
    OUTPATH="${i#*=}"
    shift # past argument=value
    ;;
    --type=*)
    TYPE="${i#*=}"
    shift # past argument=value
    ;;
    --modelpath=*)
    MODELPATH="${i#*=}"
    shift # past argument=value
    ;;
    --modelname=*)
    MODELNAME="${i#*=}"
    shift # past argument=value
    ;;
    --nestimators=*)
    NESTIMATORS="${i#*=}"
    shift # past argument=value
    ;;
    --nclasses=*)
    NCLASSES="${i#*=}"
    shift # past argument=value
    ;;
    --nfeatures=*)
    NFEATURES="${i#*=}"
    shift # past argument=value
    ;;
    --difficulty=*)
    DIFFICULTY="${i#*=}"
    shift # past argument=value
    ;;
    --nexamples=*)
    NEXAMPLES="${i#*=}"
    shift # past argument=value
    ;;
    --binarize=*)
    BINARIZE="${i#*=}"
    shift # past argument=value
    ;;
    --implementation=*)
    FIOPTIONS="${FIOPTIONS} --implementation ${i#*=}"
    IMPLEM="${i#*=}"
    shift # past argument=value
    ;;
    --baseimplementation=*)
    FIOPTIONS="${FIOPTIONS} --baseimplementation ${i#*=}"
    shift # past argument=value
    ;;
    --optimize=*)
    FIOPTIONS="${FIOPTIONS} --optimize ${i#*=}"
    shift # past argument=value
    ;;
    --baseoptimize=*)
    FIOPTIONS="${FIOPTIONS} --baseoptimize ${i#*=}"
    shift # past argument=value
    ;;
    --generate_data=*)
    GENERATE_DATA="${i#*=}"
    shift # past argument=value
    ;;
    --dataset=*)
    DATASET="${i#*=}"
    shift # past argument=value
    ;;
    --train_data=*)
    TRAIN_DATA="${i#*=}"
    shift # past argument=value
    ;;
    --batch_size=*)
    BATCH_SIZE="${i#*=}"
    shift # past argument=value
    ;;
    *)
    # unknown option
    ;;
esac
done

OUTPATH=$OUTPATH/$MODELNAME
GENERATE_DATASET="generate_${DATASET}"
TRAIN_DATASET="train_${DATASET}_${TYPE}"

if [ "$GENERATE_DATA" == "true" ]; then
    mkdir -p $OUTPATH
    if [ "$TYPE" = "cnn" ]; then
        if [ "$BINARIZE" == "on" ]; then
            python3 tests/data/$GENERATE_DATASET.py --out $OUTPATH
        else
            python3 tests/data/$GENERATE_DATASET.py --out $OUTPATH --float
        fi
    else
        if [ "$BINARIZE" == "on" ]; then
            python3 tests/data/generate_data.py --out $OUTPATH --nclasses $NCLASSES --nfeatures $NFEATURES --difficulty $DIFFICULTY --nexamples $NEXAMPLES
        else
            python3 tests/data/generate_data.py --out $OUTPATH --nclasses $NCLASSES --nfeatures $NFEATURES --difficulty $DIFFICULTY --nexamples $NEXAMPLES --float
        fi
    fi

    if [ "$TYPE" = "mlp" ] || [ "$TYPE" = "cnn" ]; then
        if [ "$BINARIZE" == "on" ]; then
            python3 tests/$TRAIN_DATASET.py --training $OUTPATH/training.csv --testing $OUTPATH/testing.csv --out $OUTPATH --name $MODELNAME --binarize
        else
            python3 tests/$TRAIN_DATASET.py --training $OUTPATH/training.csv --testing $OUTPATH/testing.csv --out $OUTPATH --name $MODELNAME 
        fi
        ENDING="onnx"
    else
        python3 tests/train_$TYPE.py --training $OUTPATH/training.csv --testing $OUTPATH/testing.csv --out $OUTPATH --name $MODELNAME  --nestimators $NESTIMATORS 
        ENDING="json"
    fi
else
    if [ "$TYPE" = "mlp" ] || [ "$TYPE" = "cnn" ]; then
        if [ "$BINARIZE" == "on" ]; then
            python3 tests/$TRAIN_DATASET.py --training $OUTPATH/training.csv --testing $OUTPATH/testing.csv --out $OUTPATH --name $MODELNAME --binarize
        else
            python3 tests/$TRAIN_DATASET.py --training $OUTPATH/training.csv --testing $OUTPATH/testing.csv --out $OUTPATH --name $MODELNAME 
        fi
        ENDING="onnx"
    else
        python3 tests/train_$TYPE.py --training $OUTPATH/training.csv --testing $OUTPATH/testing.csv --out $OUTPATH --name $MODELNAME  --nestimators $NESTIMATORS 
        ENDING="json"
    fi

    if [ "$TYPE" = "mlp" ] || [ "$TYPE" = "cnn" ]; then
        ENDING="onnx"
    else
        ENDING="json"
    fi
fi

ENDING="onnx"
if [ "$BINARIZE" == "on" ]; then
    FEATURE_TYPE="int"
    LABEL_TYPE="int"
else
    FEATURE_TYPE="double"
    LABEL_TYPE="double"
fi

python3 fastinference/main.py --model $OUTPATH/$MODELNAME.$ENDING --feature_type $FEATURE_TYPE --label_type $LABEL_TYPE --out_path $OUTPATH --out_name "model" $FIOPTIONS --implementation.batch_size $BATCH_SIZE --implementation.dataset $DATASET
python3 ./tests/data/convert_data.py --file $OUTPATH/testing.csv --out $OUTPATH/testing.h --dtype $FEATURE_TYPE --ltype "unsigned int"

IMPLE=(${IMPLEM//./ })
Q="\""
IMPL="$Q${IMPLE[1]}$Q" # echoes "xyz" in order to send it as string to CMake -> c++ code

cp ./tests/main.cpp $OUTPATH
cp ./tests/CMakeLists.txt $OUTPATH
cd $OUTPATH
cmake . -DMODELNAME=$MODELNAME -DFEATURE_TYPE=$FEATURE_TYPE -DLABEL_TYPE=$LABEL_TYPE -DBATCH_SIZE=$BATCH_SIZE -DIMPL=${IMPL}
make
./testCode testing.csv 1
