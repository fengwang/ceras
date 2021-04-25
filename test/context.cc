#include "../include/ceras.hpp"

using namespace ceras;


template< Expression Ex >
auto test_context( Ex const& ex ) noexcept
{
	std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
    // this operator will sum up all the inputs
    // if run N times with `ones({n, n})`, this operator will produce `N* ones({n,n})`
    //
	return make_unary_operator
	(
		[forward_cache]<Tensor Tsor>( Tsor const& tsor ) noexcept
		{
			typedef typename Tsor::value_type value_type;

			Tsor& ans = context_cast<Tsor>( forward_cache );
			ans.resize( tsor.shape() );
            ans += tsor;
            return ans;
		},
		[]<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
		{
            return zeros_like( grad );
		}
	)( ex );
}

int main()
{
    auto input = Input(); // ( 4, 4 )
    auto output = test_context( input );
    auto m = model{ input, output };

    auto one = ones<float>( {3, 3} );
    for ( auto idx : range( 4 ) )
        std::cout << "the " << idx+1 << "th run gives result:\n" << m.predict( one ) << std::endl;


    auto M = m;
    std::cout << "After copying, the model gives result on the first run:\n" << M.predict( one ) << std::endl;


}




