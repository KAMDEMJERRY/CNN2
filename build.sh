sudo apt update
sudo apt install libeigen3-dev
sudo apt install libopencv-dev
sudo apt install libgtest-dev
sudo apt install libboost-all-dev

rm -r build/
mkdir build
cd build
cmake ..
make
cd src/
ls
./CNN