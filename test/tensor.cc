#include "../include/tensor.hpp"
#include <iostream>

int main()
{
    ceras::tensor<double> A{ {2, 2}, {1.0, 2.0, 3.0, 1.0} };
    std::cout << "A = \n" << A << std::endl;
    std::cout << "A*A = \n" << A*A << std::endl;

    ceras::tensor<double> B{ {2, 1}, {-1.0, 1.0} };
    std::cout << "B = \n" << B << std::endl;
    std::cout << "A*B = \n" << A*B << std::endl;
    std::cout << "A*A*B = \n" << A*A*B << std::endl;

    {
        ceras::tensor<double> A{ {2, 2} };
    }

    {
        auto all_one = ceras::ones<double>( {2, 2} );
        std::cout << "ones( {2,2} ):\n" << all_one << std::endl;
        std::cout << "2*ones:\n" << 2.0 * all_one << std::endl;
        std::cout << "ones*2:\n" << all_one*2.0 << std::endl;
    }

    {
        auto x = ceras::randn<double>( {2, 2} );
        std::cout << "randn( {2,2} ):\n" << x << std::endl;
    }

    {
        auto x = ceras::randn<double>( {2,} );
        auto y = ceras::randn<double>( {2,} );
        //auto y = ceras::ones<double>( {2,} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "y:\n" << y << std::endl;
        std::cout << "x+y:\n" << x+y << std::endl;
        std::cout << "x-y:\n" << x-y << std::endl;
    }

    {
        auto x = ceras::randn<double>( {2,2} );
        auto y = ceras::randn<double>( {2,2} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "y:\n" << y << std::endl;
        std::cout << "concatenate(x, y, 0):\n" << ceras::concatenate(x, y, 0) << std::endl;
        std::cout << "concatenate(x, y, 1):\n" << ceras::concatenate(x, y, 1) << std::endl;
        std::cout << "concatenate(x, y, -1):\n" << ceras::concatenate(x, y, -1) << std::endl;
    }

    {
        auto x = ceras::randn<double>( {1,2} );
        auto y = ceras::randn<double>( {2,2} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "y:\n" << y << std::endl;
        std::cout << "concatenate(x, y, 0):\n" << ceras::concatenate(x, y, 0) << std::endl;
    }

    {
        auto x = ceras::randn<double>( {2, 1} );
        auto y = ceras::randn<double>( {2,2} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "y:\n" << y << std::endl;
        std::cout << "concatenate(x, y, 1):\n" << ceras::concatenate(x, y, 1) << std::endl;
    }

    if (0) //runtime error expected
    {
        auto x = ceras::randn<double>( {2, 1} );
        auto y = ceras::randn<double>( {2,2} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "y:\n" << y << std::endl;
        std::cout << "concatenate(x, y, 0):\n" << ceras::concatenate(x, y, 0) << std::endl;
    }


    {
        auto x = ceras::random<double>( {2, 2} );
        std::cout << "randnom( {2,2} ):\n" << x << std::endl;
    }

    {
        auto x = ceras::random<double>( {2, 3} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "repmat(x, 1, 2):\n" << ceras::repmat(x, 1, 2) << std::endl;
        std::cout << "repmat(x, 2, 1):\n" << ceras::repmat(x, 2, 1) << std::endl;
        std::cout << "repmat(x, 2, 2):\n" << ceras::repmat(x, 2, 2) << std::endl;
    }
    {
        std::cout << "Random matrix multiplication tests.\n";
        unsigned long N = 10;
        for ( auto l = 1UL; l != N; ++l )
            for ( auto m = 1UL; m != N; ++m )
                for ( auto n = 1UL; n != N; ++n )
                {
                    auto x = ceras::random<double>( {l, m} ) * ceras::random<double>( {m,n}  );
                    std::cout << x[0] << "\t";
                }
        std::cout << "\n";
    }

    {
        auto x = ceras::random<double>( {2, 3} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "-x:\n" << -x << std::endl;
        std::cout << "-x+x:\n" << -x+x << std::endl;
    }

    {
        auto x = ceras::random<double>( {2, 3} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "x.ndim():\n" << x.ndim() << std::endl;
    }

    {
        auto x = ceras::random<double>( {2, 3} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "x.reset():\n" << x.reset() << std::endl;
    }

    {
        auto x = ceras::random<double>( {2, 3} );
        std::cout << "zeros_like(x):\n" << ceras::zeros_like(x) << std::endl;
        std::cout << "zeros([2, 3]):\n" << ceras::zeros<double>(x.shape()) << std::endl;
    }

    {
        auto x = ceras::random<double>( {2, 3} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "x-1:\n" << x-1.0 << std::endl;
        std::cout << "1-x:\n" << 1.0-x << std::endl;
    }

    // test gemm
    {
        auto x = ceras::tensor<double>{ {2, 2}, {0.0, 1.1, 2.3, 4.7} };
        std::cout << "x:\n" << x << std::endl;

        auto y = ceras::tensor<double>{ {2, 2}, {0.5, 1.3, 2.1, 4.5} };
        std::cout << "y:\n" << y << std::endl;

        auto a = ceras::tensor<double>{ {2, 2} };

        ceras::gemm( x.data(), true, y.data(), true, 2, 2, 2, a.data() );
        std::cout << "x' y' = \n" <<  a << std::endl;

        ceras::gemm( x.data(), true, y.data(), false, 2, 2, 2, a.data() );
        std::cout << "x' y = \n" <<  a << std::endl;

        ceras::gemm( x.data(), false, y.data(), true, 2, 2, 2, a.data() );
        std::cout << "x y' = \n" <<  a << std::endl;

        ceras::gemm( x.data(), false, y.data(), false, 2, 2, 2, a.data() );
        std::cout << "x y = \n" <<  a << std::endl;
    }


    {
        auto x = ceras::linspace( 0.0, 24.0, 24, false );
        x.reshape( {2, 3, 4} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "max(x):\n" << max(x) << std::endl;
        std::cout << "max(x, 0):\n" << max(x, 0) << std::endl;
        std::cout << "max(x, 1):\n" << max(x, 1) << std::endl;
        std::cout << "max(x, 2):\n" << max(x, 2) << std::endl;
        std::cout << "max(x, -1):\n" << max(x, -1) << std::endl;
        std::cout << "max(x, 0, true):\n" << max(x, 0, true) << std::endl;
        std::cout << "max(x, 1, true):\n" << max(x, 1, true) << std::endl;
        std::cout << "max(x, 2, true):\n" << max(x, 2, true) << std::endl;
        std::cout << "max(x, -1, true):\n" << max(x, -1, true) << std::endl;
    }

    {
        auto x = ceras::linspace( 0.0, 24.0, 24, false );
        x.reshape( {2, 3, 4} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "min(x):\n" << min(x) << std::endl;
        std::cout << "min(x, 0):\n" << min(x, 0) << std::endl;
        std::cout << "min(x, 1):\n" << min(x, 1) << std::endl;
        std::cout << "min(x, 2):\n" << min(x, 2) << std::endl;
        std::cout << "min(x, -1):\n" << min(x, -1) << std::endl;
        std::cout << "min(x, 0, true):\n" << min(x, 0, true) << std::endl;
        std::cout << "min(x, 1, true):\n" << min(x, 1, true) << std::endl;
        std::cout << "min(x, 2, true):\n" << min(x, 2, true) << std::endl;
        std::cout << "min(x, -1, true):\n" << min(x, -1, true) << std::endl;
    }

    {
        auto x = ceras::linspace( 0.0, 24.0, 24, false );
        x.reshape( {2, 3, 4} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "mean(x):\n" << mean(x) << std::endl;
        std::cout << "mean(x, 0):\n" << mean(x, 0) << std::endl;
        std::cout << "mean(x, 1):\n" << mean(x, 1) << std::endl;
        std::cout << "mean(x, 2):\n" << mean(x, 2) << std::endl;
        std::cout << "mean(x, -1):\n" << mean(x, -1) << std::endl;
        std::cout << "mean(x, 0, true):\n" << mean(x, 0, true) << std::endl;
        std::cout << "mean(x, 1, true):\n" << mean(x, 1, true) << std::endl;
        std::cout << "mean(x, 2, true):\n" << mean(x, 2, true) << std::endl;
        std::cout << "mean(x, -1, true):\n" << mean(x, -1, true) << std::endl;
    }

    {
        auto x = ceras::linspace( 0.0, 24.0, 24, false );
        x.reshape( {2, 3, 4} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "sum(x):\n" << sum(x) << std::endl;
        std::cout << "sum(x, 0):\n" << sum(x, 0) << std::endl;
        std::cout << "sum(x, 1):\n" << sum(x, 1) << std::endl;
        std::cout << "sum(x, 2):\n" << sum(x, 2) << std::endl;
        std::cout << "sum(x, -1):\n" << sum(x, -1) << std::endl;
        std::cout << "sum(x, 0, true):\n" << sum(x, 0, true) << std::endl;
        std::cout << "sum(x, 1, true):\n" << sum(x, 1, true) << std::endl;
        std::cout << "sum(x, 2, true):\n" << sum(x, 2, true) << std::endl;
        std::cout << "sum(x, -1, true):\n" << sum(x, -1, true) << std::endl;
    }
    {
        auto x = ceras::linspace( 0.0, 24.0, 24, false );
        x.reshape( {6, 4} );
        auto y = ceras::copy( x );
        std::cout << "y:\n" << y << std::endl;
    }

    {
        auto x = ceras::as_tensor( 1.0e-3 );
        std::cout << "x:\n" << x << std::endl;
    }

    {
        auto x = ceras::random<double>( {3, 3}, 0.0, 10.0 );
        std::cout << "x:\n" << x << std::endl;
        auto sx = ceras::softmax( x );
        std::cout << "softmax x:\n" << sx << std::endl;
    }

    {
        auto x = ceras::linspace( 0.0, 6.0, 6, false );
        x.reshape( {1, 6} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "sum(x):\n" << sum(x) << std::endl;
        std::cout << "sum(x, 0):\n" << sum(x, 0) << std::endl;
        std::cout << "sum(x, 1):\n" << sum(x, 1) << std::endl;
        std::cout << "sum(x, -1):\n" << sum(x, -1) << std::endl;
        std::cout << "sum(x, 0, true):\n" << sum(x, 0, true) << std::endl;
        std::cout << "sum(x, 1, true):\n" << sum(x, 1, true) << std::endl;
        std::cout << "sum(x, -1, true):\n" << sum(x, -1, true) << std::endl;
    }

    {
        auto x = ceras::linspace( 0.0, 6.0, 6, false );
        x.reshape( { 6, 1} );
        std::cout << "x:\n" << x << std::endl;
        std::cout << "sum(x):\n" << sum(x) << std::endl;
        std::cout << "sum(x, 0):\n" << sum(x, 0) << std::endl;
        std::cout << "sum(x, 1):\n" << sum(x, 1) << std::endl;
        std::cout << "sum(x, -1):\n" << sum(x, -1) << std::endl;
        std::cout << "sum(x, 0, true):\n" << sum(x, 0, true) << std::endl;
        std::cout << "sum(x, 1, true):\n" << sum(x, 1, true) << std::endl;
        std::cout << "sum(x, -1, true):\n" << sum(x, -1, true) << std::endl;
    }
    {
        auto x = ceras::random<double>( {1, 2, 3, 4}, 0.0, 10.0 );
        auto v4 = ceras::view_4d{ x.data(), 1, 2, 3, 4 };
        std::cout << v4[0][0][0][0] << std::endl;
        std::cout << v4[0][1][2][3] << std::endl;
        auto v3 = ceras::view_3d{ x.data(), 2, 3, 4 };
        std::cout << v3[0][0][0] << std::endl;
        std::cout << v3[1][2][3] << std::endl;

        v3[1][2][3] = 1.0;
        std::cout << v4[0][1][2][3] << std::endl;
    }

    return 0;
}


