#ifndef DGIDKOSTMBYAWNFFKALGHEJNQNOVEFFCRCJMOVVQQSNSJQPJMMTFBIVANGSXAWWUKGMEDNOAM
#define DGIDKOSTMBYAWNFFKALGHEJNQNOVEFFCRCJMOVVQQSNSJQPJMMTFBIVANGSXAWWUKGMEDNOAM

#include "../includes.hpp"
#include "../place_holder.hpp"

#include "./initializer.hpp"
#include "./constraint.hpp"
#include "./regularizer.hpp"
#include "./activation.hpp"
#include "./keras_utils.hpp"

#include "../utils/debug.hpp"
#include "../utils/enable_shared.hpp"

namespace Keras
{

    // for layers
    // - record information when calling constructor
    // - buildup computation graph when calling ()
    // - each layer has a _config layer for interfacing

    namespace keras_details
    {
        template< typename Concrete_Layer >
        struct layer_default
        {
            bool trainable_ = true; // always true unless training a GAN
            std::string name_;

            std::vector<unsigned long> input_shape_;
            std::vector<unsigned long> output_shape_;

            std::vector<std::shared_ptr<ceras::variable<ceras::tensor<float>>>> weights_;
            std::vector<std::shared_ptr<ceras::variable<ceras::tensor<float>>>> trainable_weights_;
        };

        struct Input : layer_default<Input>//, ceras::enable_shared<Input>
        {
            ceras::place_holder<ceras::tensor<float>> place_holder_; // place_holder is copiable

            Input( std::vector<unsigned long>const& shape ) noexcept
            {
                (*this).name_ = "input";
                (*this).input_shape_ = shape;
                (*this).output_shape_ = shape;
                place_holder_ = std::make_shared<ceras::place_holder<ceras::tensor<float>>>( (*this).input_shape_ );
            }

            auto operator()() noexcept // return an instance of an expression
            {
                return place_holder_;
            }
        };//Input

    }//keras_details

    // auto inp = input();
    // auto inp = input( {28*28,} );
    // auto inp = input( {28, 28, 1} );
    //
    //using Input = keras_details::Input;


    // 'cause of bug 99118 at gcc <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=99118>
    struct Input : keras_details::Input
    {
        Input( std::vector<unsigned long> const& shape ) noexcept : keras_details::Input{ shape } {}
    };

    namespace keras_details
    {
        template<typename Config, typename Upcoming_Layer>
        struct dense : layer_default<dense<Config, Upcoming_Layer>>, ceras::enable_shared<dense<Config, Upcoming_Layer>>
        {
            Config config_;
            Upcoming_Layer upcoming_layer_;

            typedef ceras::variable<ceras::tensor<float>> variable_type;
            typedef std::shared_ptr< variable_type > variable_pointer;

            variable_pointer weight_;
            variable_pointer bias_;

            dense( Config const& config, Upcoming_Layer const& upcoming_layer ) noexcept : config_{ config }, upcoming_layer_{ upcoming_layer }
            {
                (*this).input_shape_ = upcoming_layer.output_shape_;
                better_assert( (*this).input_shape_.size() == 1, "Dense layer expects an input layer of size 1." );

                //unsigned long const input_dim = ((*this).input_shape_)[0];
                unsigned long const output_dim = config_.filters_;

                ((*this).output_shape_).resize( 1 );
                ((*this).output_shape_)[0] = output_dim;
            }

            auto operator()() noexcept // operation happens here, return instance of an expression
            {
                unsigned long const input_dim = ((*this).input_shape_)[0];
                unsigned long const output_dim = config_.filters_;

                // create weight
                {
                    typedef typename Config::kernel_initializer_type kernel_initializer_type;
                    weight_ = std::make_shared<variable_type>( kernel_initializer_type{}( {input_dim, output_dim} ) );
                    (*this).weights_.push_back( weight_ );
                    (*this).trainable_weights_.push_back( weight_ );
                }

                //if constexpr( config_.use_bias_ )
                if constexpr( Config::use_bias_type::data_ )
                {
                    typedef typename Config::bias_initializer_type bias_initializer_type;
                    bias_ = std::make_shared< variable_type >( bias_initializer_type{}( {output_dim,} ) );
                    (*this).weights_.push_back( bias_ );
                    (*this).trainable_weights_.push_back( bias_ );
                }

                typedef typename Config::activation_type activation_type;
                auto prev_expression = upcoming_layer_();

                if constexpr ( std::is_same_v<decltype(prev_expression), ceras::place_holder<ceras::tensor<float>>> )
                    ceras::debug_print( "Keras: dense layer generating expression, with prev place holder id ", prev_expression.id_ );

                if constexpr( Config::use_bias_type::data_ )
                    return activation_type{}( prev_expression * (*weight_) + (*bias_) );
                else
                    return activation_type{}(  prev_expression * (*weight_) );
            }
        };

        template< unsigned long Filters, typename Activation=activation<"None">, typename Use_Bias=use_bias<true>,
                  typename Kernel_Initializer=initializer<"glorot_uniform">, typename Bias_Initializer=initializer<"zeros">,
                  typename Kernel_Regularizer=regularizer<"None">, typename Bias_Regularizer=regularizer<"None">,
                  typename Activity_Regularizer=regularizer<"None">, typename Kernel_Constraint=constraint<"None">,
                  typename Bias_Constraint=constraint<"None">
            >
        struct Dense_Config
        {
            typedef Activation              activation_type;
            typedef Use_Bias                use_bias_type;
            typedef Kernel_Initializer      kernel_initializer_type;
            typedef Bias_Initializer        bias_initializer_type;
            typedef Kernel_Regularizer      kernel_regularizer_type;
            typedef Bias_Regularizer        bias_regularizer_type;
            typedef Activity_Regularizer    activity_regularizer_type;
            typedef Kernel_Constraint       kernel_constraint_type;
            typedef Bias_Constraint         bias_constraint_type;

            static constexpr unsigned long filters_ = Filters;
            //static constexpr bool use_bias_ = Use_Bias::data_;

            template<typename Upcoming_Layer>
            auto operator()( Upcoming_Layer const& upcoming_layer ) const noexcept
            {
                return dense{ *this, upcoming_layer };
            }
        };

    }//keras_details

    // auto inp = input{ {28*28,} };
    // auto l1 = Dense<64>{}( inp );
    //template< unsigned long Filters, typename ... Args >
    //using Dense = keras_details::Dense_Config<Filters, Args...>;

    template< unsigned long Filters, typename ... Args >
    struct Dense : keras_details::Dense_Config<Filters, Args...> {};


#if 0

    namespace keras_details
    {
        // auto x = Reshape<target_shape<28, 28, 1>>()( input );
        //
        template< typename Config, typename Upcoming_Layer >
        struct reshape : layer_default<reshape<Config, Upcoming_Layer>>
        {
            Config config_;
            Upcoming_Layer upcoming_layer_;

            reshape( Config const& reshape_config, Upcoming_Layer const& upcoming_layer ) noexcept : config_{ reshape_config }, upcoming_layer_{ upcoming_layer }
            {
                output_shape_ = Config::target_shape_type::data_;
            }

            auto operator()() const noexcept
            {
                return ceras::reshape( output_shape_ )( upcoming_layer_() );
            }
        };

        template< typename Target_Shape >
        struct Reshape_Config
        {
            typedef Target_Shape target_shape_type;

            template< typename Upcoming_Layer >
            auto operator ()( Upcoming_Layer const& upcoming_layer ) const noexcept
            {
                return reshape{*this, upcoming_layer};
            }
        };
    }//keras_details

    template< typename Target_Shape >
    using Reshape = keras_details::Reshape_Config<Target_Shape>;


    namespace keras_details
    {
        // TODO: remove std::optional wrapper here
    template< typename Config, typename Upcoming_Layer >
    struct conv2d : layer_default<conv2d<Config, Upcoming_Layer>>
    {
        Config config_;
        Upcoming_Layer upcoming_layer_;

        // to calculate
        std::vector<std::optional<unsigned long>> output_shape_;
        std::vector<std::optional<unsigned long>> input_shape_;

        unsigned long row_kernel_;
        unsigned long col_kernel_;
        unsigned long row_stride_;
        unsigned long col_stride_;
        unsigned long row_dilation_;
        unsigned long col_dilation_;
        unsigned long row_input_;
        unsigned long col_input_;
        unsigned long input_filters_;
        unsigned long output_filters_;
        std::string padding_;

        auto operator()() const noexcept
        {
            auto conv_kernel = ceras::variable{ ceras::randn<float>({output_filters, row_kernel_, col_kernel_, input_filters_}, 0.0, 10.0f/(std::sqrt(output_filters_*row_kernel_*col_kernel_*input_filters_))) }; //TODO: apply kernel initializer
            auto output_operation = ceras::conv2d(row_input_, col_input_, row_stride_, col_stride_, row_dilation_, col_dilation_, padding_)(upcoming_layer_(), conv_kernel);

            if constexpr( std::is_same_v<Config::use_bias_type, use_bias<true>> )
                return output_operation + ceras::variable{ceras::zeros<float>( {output_filters_,} )};
            else
                return output_operation;
        }

        conv2d( Config const& conv2d_config, Upcoming_Layer const& upcoming_layer ) : config_{conv2d_config}, upcoming_layer_{upcoming_layer}
        {
            input_shape_ = upcoming_layer.output_shape_;
            better_assert( input_shape_.size() == 3, "Expecting a 3D input, but got ", input_shape_.size() );

            std::tie( row_input_, col_input_ input_filters_ ) = std::make_tuple( input_shape_[0].value_or(0), input_shape_[1].value_or(0), input_shape_[3].value() );
            std::tie( row_kernel_, col_kernel_ ) = Config::kernel_type::data_;
            std::tie( row_stride_, col_stride_ ) = Config::strides_type::data_;
            std::tie( row_dilation_, col_dilation_ ) = Config::dilation_rate_type::data_;
            output_filters_ = Config::filters_type::data_;
            padding_ = Config::padding_type::data_;

            calculate_output_shape();
        }

        void calculate_output_shape()
        {
            // the last dim is the filters in config
            unsigned long const row_input = input_shape_[0].value_or(0);
            unsigned long const col_input = input_shape_[1].value_or(0);

            auto const [row_kernel, col_kernel] = Config::kernel_type::data_;
            auto const [row_stride, col_stride] = Config::strides_type::data_;
            auto const [row_dilation, col_dilation] = Config::dilation_rate_type::data_;

            unsigned long row_padding = 0;
            unsigned long col_padding = 0;
            if costexpr( std::is_same_v<Config::padding_type,padding<"same">> )
            {
                unsigned long const row_padding_total = (row_kernel + (row_kernel - 1) * (row_dilation - 1) - row_stride);
                unsigned long const col_padding_total = (col_kernel + (col_kernel - 1) * (col_dilation - 1) - col_stride);
                row_padding = row_padding_total >> 1;
                col_padding = col_padding_total >> 1;
            }
            unsigned long const row_output = ( row_input + 2 * row_padding - ( row_dilation * (row_kernel - 1) + 1 ) ) / row_stride + 1;
            unsigned long const col_output = ( col_input + 2 * row_padding - ( col_dilation * (col_kernel - 1) + 1 ) ) / col_stride + 1;

            //deal with nullopt cases
            output_shape_.resize( 3 );
            if ( input_shape_[0] ) output_shape_[0] = row_output;
            if ( input_shape_[1] ) output_shape_[1] = col_output;
            output_shape_[2] = Config::filters_type::data_;
        }
    };

    template< typename Filters=filters<1>, typename Kernel_Size=kernel_size<3, 3>, typename Strides=strides<1,1>, typename Padding=padding("valid"), /* data_format is not supported */
              typename Dilation_Rate=dilation_rate<1,1>, typename Activation=activation<"None">, typename Use_Bias=use_bias<true>,
              typename Kernel_Initializer=kernel_initializer<"glorot_uniform">, typename Bias_Initializer=bias_initializer<"zeros">,
              typename kernel_Regularizer=kernel_regularizer<"None">, typename Bias_Regularizer=bias_regularizer<"None">,
              typename Activity_Regularizer=activity_regularizer<"None">, typename Kernel_Constraint=kernel_constraint<"None">,
              typename Bias_Constraint=bias_constraint<"None">
            >
    struct Conv2D_Config
    {
        typedef Filters filters_type;
        typedef Kernel_Size kernel_size_type;
        typedef Strides strides_type;
        typedef Padding padding_type;
        typedef Dilation_Rate dilation_rate_type;
        typedef Activation activation_type;
        typedef Use_Bias use_bias_type;
        typedef Kernel_Initializer kernel_initializer_type;
        typedef Bias_Initializer bias_initializer_type;
        typedef Kernel_Regularizer kernel_regularizer_type;
        typedef Bias_Regularizer bias_regularizer_type;
        typedef Activity_Regularizer activity_regularizer_type;
        typedef Kernel_Constraint kernel_constraint_type;
        typedef Bias_Constraint bias_constraint_type;

        template< typename Upcoming_Layer >
        auto operator ()( Input_layer const& upcoming_layer ) const noexcept
        {
            return conv2d{*this, upcoming_layer};
        }
    };
    }//keras_details

    template< typename ... Args >
    using Conv2D = keras_details::Conv2D_Config<Args...>; // to make 'Conv2D<...>{}( upcoming_layer );' work


#endif




}//namespace Keras

#endif//DGIDKOSTMBYAWNFFKALGHEJNQNOVEFFCRCJMOVVQQSNSJQPJMMTFBIVANGSXAWWUKGMEDNOAM

