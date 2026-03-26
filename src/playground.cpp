// #include "Tensor.hpp"
// #include "SparseTensor.hpp"
// #include "SparseConvLayer3D.hpp"
// #include <iostream>
// #include <fstream>



// int main(int argc, char* argv[]) {



//     return 0;
// }

#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <array>
#include <vector>
#include <fstream>


template<typename A>
class T{
    std::array<A, 10> arr;
public:    
    T(std::initializer_list<A> a){
        std::copy(a.begin(), a.begin() + 10, arr.begin());
    }

    A& operator[](int idx){
        return arr[idx];
    }
};

int main() {
    std::ofstream of{"file.txt", std::ios::out};
    T<int> t{1,2,3};
    T<int> & rt = t;
    t[0] = 4;
    std::cout << t[0];
    of << "{" << t[0] <<"}";
    
    return 0;

}
