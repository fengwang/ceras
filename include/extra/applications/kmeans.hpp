#ifndef RTWQPTYASMYDTYHVSHPTUECQOBHWFWBXEJKPVSAJMXDYAOFICKLURFYSPNFAWGAJABATYSQKE
#define RTWQPTYASMYDTYHVSHPTUECQOBHWFWBXEJKPVSAJMXDYAOFICKLURFYSPNFAWGAJABATYSQKE

#include "../../tensor.hpp"
#include "../../operation.hpp"
#include "../../utils/better_assert.hpp"
#include "../../utils/range.hpp"

namespace ceras::applications
{

    template< Tensor Tsor >
    struct kmeans
    {
        typedef typename Tsor::value_type value_type;

        Tsor samples_; // of shape (n_samples, n_features)
        unsigned long n_clusters_;
        unsigned long iterations_;
        unsigned long n_samples_;
        unsigned long n_features_;
        Tsor centers_; // of shape(n_clusters_, n_features_)
        Tsor sample_parameters_; // of shape(n_features_, 2)

        kmeans( Tsor const& samples, unsigned long n_clusters, unsigned long iterations ) noexcept : samples_{ samples }, n_clusters_{ n_clusters }, iterations_{ iterations }
        {
            better_assert( samples_.shape().size() == 2, "kmeans: expecting input samples has 2 dimensions, but got ", samples_.shape().size() );

            n_samples_ = *(samples_.shape().begin());
            n_features_ = *(samples_.shape().rbegin());
            centers_ = random( {n_clusters_, n_features_} );
            sample_parameters_.resize( {n_features_, 2} ); // collecting <min, max> for each feature
            {
                view_2d vp{ sample_parameters_.data(), n_features_, 2 };
                view_2d v2{ samples_.data(), n_samples_, n_features_ };
                for ( auto idx : range( n_features_ ) )
                {
                    vp[idx][0] = *std::min_element( v2.col_begin(idx), v2.col_end(idx) );
                    vp[idx][1] = *std::max_element( v2.col_begin(idx), v2.col_end(idx) );

                    for_each( v2.col_begin(idx), v2.col_end(idx), [mn=vp[idx][0], mx=vp[idx][1]]( value_type& x ){ x = ( x - mn ) / ( mx - mn + eps ); } ); // normalize features in samples_ to range[0, 1]
                }
            }

            fit();
        }

        Tsor predict( Tsor const& input )
        {
            // TODO:
        }

        private:

        void fit()
        {
            // create model
            auto input = place_holder<Tsor>{}; // to be bind to samples_, but maybe in batch. of shape (n_samples, n_features,)
            auto l0 = reshape( {1, n_features_} )( input ); // shape -> ( n_samples, 1, n_features )
            auto l2 = repeat( n_clusters_, 1 )( l0 ); // shape -> ( n_samples, n_clusters, n_features )
            auto c = variance{ centers_ }; // directly binding centers to a variable, shape -> ( n_clusters, n_features )
            auto df = l2 - c; // shape -> ( n_samples, n_clusters, n_features )
            auto df2 = hadamard_product( df, df ); // shape -> (n_samples, n_clusters, n_features)
            auto df2x = reduce_sum( -1 )( df2 ); // shape -> (n_samples, n_clusters )
            auto df2xm = reduce_min( -1 )( df2x ); // shape -> (n_samples,)

            auto target = place_holder<Tsor>{};

            auto& s = get_default_session<tensor_type>();




        }
    };

}//namespace ceras::applications

#endif//RTWQPTYASMYDTYHVSHPTUECQOBHWFWBXEJKPVSAJMXDYAOFICKLURFYSPNFAWGAJABATYSQKE

