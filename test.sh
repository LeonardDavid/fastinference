# Define some constants
OUTPATH="/tmp/fastinference"
MODELNAME="RandomForestClassifier"
FEATURE_TYPE="int"

# Generate some artificial data with 5 classes, 20 features and 10000 data points.
python3 tests/data/generate_data.py --out $OUTPATH --nclasses 5 --nfeatures 20 --difficulty 0.5 --nexamples 10000

# Train a RF with 25 trees on the generated data
python3 tests/train_$TYPE.py --training $OUTPATH/training.csv --testing $OUTPATH/testing.csv --out $OUTPATH --name $MODELNAME  --nestimators 25

# Perform the actual optimization + code generation
python3 fastinference/main.py --model $OUTPATH/$MODELNAME.json --feature_type $FEATURE_TYPE --out_path $OUTPATH --out_name "model" --implementation cpp --baseimplementation cpp.ifelse --baseoptimize swap

# Prepare the C++ files for compilation
python3 ./tests/data/convert_data.py --file $OUTPATH/testing.csv --out $OUTPATH/testing.h --dtype $FEATURE_TYPE --ltype "unsigned int"
cp ./tests/main.cpp $OUTPATH
cp ./tests/CMakeLists.txt $OUTPATH
cd $OUTPATH

# Compile the code
cmake . -DMODELNAME=$MODELNAME -DFEATURE_TYPE=$FEATURE_TYPE
make

# Run the code
./testCode