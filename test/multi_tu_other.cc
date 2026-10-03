#include "ceras.hpp"
double other_tu() {return ceras::sum(ceras::ones<double>({2}));}

double other_random() {return ceras::random<double>({1})[0];}
