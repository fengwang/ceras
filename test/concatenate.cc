#include "../include/ceras.hpp"
#include "../include/utils/debug.hpp"
#include "../include/utils/color.hpp"
#include <cmath>
#include <iostream>

void test_1()
{
    std::cout << color::rize( "test_1", "Red" ) << std::endl;
    auto a = ceras::linspace<double>( 1.0, 16.0, 16 );
    a.reshape( {1, 4, 4, 1} );
    std::cout << "a created with:\n" << ceras::squeeze(a) << std::endl;

    auto b = ceras::linspace<double>( -16.0, -1.0, 16 );
    b.reshape( {1, 4, 4, 1} );
    std::cout << "b created with:\n" << ceras::squeeze(b) << std::endl;

    auto va = ceras::variable{ a };
    auto vb = ceras::variable{ b };
    //auto cab = ceras::concatenate( va, vb )();
    auto cab = ceras::concatenate()( va, vb );

    //ceras::session<ceras::tensor<double>> s;
    auto& s = ceras::get_default_session<ceras::tensor<double>>();
    auto ans = s.run( cab );
    cab.reset_states();
    std::cout << "after concate(a,b)():\n" << ceras::squeeze(ans) << std::endl;

    auto grad = ceras::linspace<double>( 1.0, 32.0, 32 );
    grad.reshape( {1, 4, 4, 2} );
    std::cout << "gradient generated as:\n" << ceras::squeeze(grad) << std::endl;
    cab.backward( grad );

    auto new_ga = (va.state_->gradient_);
    std::cout << "propageated gradient at a:\n" << ceras::squeeze(new_ga) << std::endl;

    auto new_gb = (vb.state_->gradient_);
    std::cout << "propageated gradient ab b:\n" << ceras::squeeze(new_gb) << std::endl;
}

int main()
{
    test_1();

    return 0;
}

