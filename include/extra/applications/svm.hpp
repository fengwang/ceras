#ifndef RTWQPTYASMYDTYHVSHPTUECQOBHWFWBXEJKPVSAJMXDYAOFICKLURFYSPNFAWGAJABATYSQKE
#define RTWQPTYASMYDTYHVSHPTUECQOBHWFWBXEJKPVSAJMXDYAOFICKLURFYSPNFAWGAJABATYSQKE

#include "../../ceras.hpp"
#include "../../utils/string.hpp"

namespace ceras::applications
{

#if 0

    example usage:

    svc<"linear", float> c{ parameters... };
    auto m = c.fit( x, y );
    auto ans = m.predict( xx );

#endif


    template<Tensor Tsor >
    auto svm_linear_fit( Tsor const& x, Tsor const& y, unsigned long epoch, typename Tsor::value_type c=1 )
    {
        typedef Tsor::value_type value_type;

        unsigned long const batch_size = *(x.shape().begin());
        unsigned long const dim = *(x.shape().rbegin());

        // define a model
        auto input = place_holder<Tsor>{};
        auto w = variable<Tsor>{ glorot_uniform( {dim,} ) };
        auto b = variable<Tsor>{ zeros( {1,} ) };
        auto output = input * w + b;
        auto y_ = place_holder<Tsor>{};
        auto sm_ loss = sum_reduce( relu( value{value_type{1}} - hadamard_product( y_, output ) ) * value{value_type{c+c}} );
        auto regulation_loss = sum_reduce( square( w ) );
        auto loss = sm_loss + regulation_loss;

        // define optimizer
        auto& s = get_default_session<Tsor>();
        s.bind( input, x );
        s.bind( y_, y );
        value_type learning_rate = 1.0e-2;
        auto optimizer = gradient_descent{ loss, batch_size, learning_rate };

        for ( [[maybe_unused]] auto e : range( epoch ) )
        {
            s.run( loss );
            s.run( optimizer );
        }

        auto m = model{ input, output };
        return m;
    }

    ///
    /// auto scv = svm<"linear">{};
    ///
    template<string name> // name -- linear, .....
    struct svm
    {
        ///
        /// return a trained model
        ///
        template< Tensor Tsor >
        auto fit( Tsor const& x, Tsor const& y, unsigned long epoch )
        {
            typedef Tsor::value_type value_type;

            auto m = make_svm_model<name>( x.shape(), y.shape() );

            unsigned long const batch_size = *(s.shape().begin());
            auto cm = m.compiple( Hinge(), SGD(batch_size, value_type{1.0e-2}) );
            cm.fit( x, y, batch_siz, epoch );
            return cm;
        }
    };// struct svm

    ///
    ///
    ///








    ///
    /// Linear SVM.
    ///
    /// @tparam Tsor A tensor type indicating the data type used for training and prediction.
    ///
    template<Tensor Tsor>
    struct linear_svm
    {
        typedef typename Tsor::value_type value_type;

        Tsor x_; ///< 2D tensor. The first dimension is for the numbers of instances, and the second dimension is for the numbers of features
        Tsor y_; //< 2D tensor. The first dimension is for the numbers of instances, and the second dimension is for the labels. The label is either -1 or 1
        unsigned long epochs_; ///< Train epoches..
        value_type c_; ///< Optimization constraint.

        ///
        /// @brief constructor
        ///
        svm( Tsor const& x, Tsor const& y, unsigned long epochs = 100, value_type c=1.0 ) : x_{ x }, y_{ y }, epochs_{ epochs }, c_{ c } {}

        value_type train()
        {
            unsigned long const features = *(x_.shape().rbegin());

            // creating model
            auto x = place_holder<Tsor>{};
            auto w = variable{ glorot_uniform<value_type>({features, 1}) };
            auto b = variable{ zeros<value_type>( {1,} ) };
            auto _y = x * w + b;

            // creating loss
            auto y = place_holder<Tsor>{};
            auto hinge_loss = sum_reduce( maximum( value{value_type{0.0}}, value{value_type{1.0}} - hadamard_product(_y, y) ) );
            auto regulation_loss = sum_reduce( hadamard_product( w, w ) );
            auto loss = regulation_loss + value{value_type{c_}} * hinge_loss;

            // creating optimizer
            unsigned long const batch_size = *(x_.shape().begin());
            value_type learning_rate = 0.01;
            auto optimizer = gradient_descent{ loss, batch_size, learning_rate };

            // traing the model
            session<Tsor> s;
            s.bind( x, x_ );
            s.bind( y, y_ );
            for ( [[maybe_unused]]auto e : range( epoch ) )
            {
                s.run( _y ); // forward pass
                s.run( optimizer ); // backward pass
            }

            // get loss
            auto actual_loss = s.run( loss );
            // build model
            auto svm_model = model{ x, _y };

            return std::make_tuple( svm_model, *(actual_loss.begin()) );
        }
    };

}//namespace ceras::applications

#endif//RTWQPTYASMYDTYHVSHPTUECQOBHWFWBXEJKPVSAJMXDYAOFICKLURFYSPNFAWGAJABATYSQKE

