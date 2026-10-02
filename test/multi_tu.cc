#include "ceras.hpp"
double other_tu();
double other_random();
int main() {
    ceras::seed_random(42); auto expected=ceras::random<double>({1})[0];
    ceras::seed_random(42); if (other_random()!=expected) return 2;
    auto a=ceras::ones<double>({2});
    return a[0]==1 && other_tu()==2 ? 0 : 1;
}
