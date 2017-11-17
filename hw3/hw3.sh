python3 "cnn.py" "$1" "model/model.h5"
python3 "semi.py" "$1" "test.csv" "model/model.h5"
