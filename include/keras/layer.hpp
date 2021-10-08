//
// trying to duplicate keras interfaces, following the same style ...
//
#ifndef BIYVCSVIYREATABTGLTJXSOTHRCQIOUAUKNTNMSACVTQWWTIDVLGGTFAMNIIFJOXOHFDCEAWI
#define BIYVCSVIYREATABTGLTJXSOTHRCQIOUAUKNTNMSACVTQWWTIDVLGGTFAMNIIFJOXOHFDCEAWI

#include "../includes.hpp"
#include "../operation.hpp"
#include "../activation.hpp"
#include "../tensor.hpp"
#include "../place_holder.hpp"
#include "../session.hpp"
#include "../utils/debug.hpp"
#include "../utils/fmt.hpp"

#include "./field.hpp"

namespace ceras::keras
{

    static constexpr unsigned long None = std::numeric_limits<unsigned long>::max();

    ///
    /// @brief Extract the current layer (the leading layer).
    ///
    /// \code{.cpp}
    /// auto input = Input().shape( {128, 128, 3} )();
    /// auto l1 = Conv2D().filters(16).kernel_size({3, 3}).padding("valid")( input );
    /// auto conv2d_layer = layer( l1 );
    /// auto const& input_shape = conv2d_layer.input_shape(); // access inner properities of a layer, the input shape
    /// auto const& output_shape = conv2d_layer.compute_output_shape(); // calculate the output shape
    /// auto const& gamma = conv2d_layer.gamm_; // access the gamma weight
    /// auto const& beta = conv2d_layer.beta_; // access the beta weight
    /// \endcode
    ///
    template< typename... Layers >
    auto layer( std::tuple<Layers...> const& lt ) noexcept
    {
        return *(std::get<0>(lt));
    }

    // construct a computation graph for the accumulated layers
    template< typename... Layers >
    auto construct_computation_graph( std::tuple<Layers...> const& lt ) noexcept
    {
        constexpr unsigned long dim = sizeof...(Layers);

        auto const& leading_layer = *(std::get<0>(lt));

        if constexpr( dim == 1 ) // input layer
            return leading_layer();
        else if constexpr( dim == 2 ) // unary operator
            return leading_layer( construct_computation_graph( std::get<1>(lt) ) );
        else if constexpr( dim == 3 ) // binary operator
            return leading_layer( construct_computation_graph( std::get<1>(lt) ), construct_computation_graph( std::get<2>(lt) ) );
        else
            better_assert( false, "Error: should never reach here." );
    }

    ///
    /// CRTP layer base, providing several interfaces such as input_shape, output_shape through config method
    ///
    /// \code{.cpp}
    /// x_layer : Layer { /*...*/ };
    /// x_layer x;
    /// auto const& input_shape = x.config().input_shape();
    /// auto const& output_shape = x.config().output_shape();
    /// auto const& name = x.config().name();
    /// //auto const& properity_xxxx = x.config().xxxx(); // were there such a properity
    /// \endcode
    ///
    template< typename Concrete_Layer >
    struct Layer
    {
        auto const& config() const noexcept
        {
            return static_cast<Concrete_Layer const&>(*this).config_;
        }

        auto& config() noexcept
        {
            return static_cast<Concrete_Layer&>(*this).config_;
        }
    };


    struct InputLayer;

    struct InputConfig :
         enable_batch_size<InputConfig, None>,
         enable_input_shape<InputConfig, None>,
         enable_name<InputConfig, "Input">,
         enable_output_shape<InputConfig, None>,
         enable_shape<InputConfig, None>,
         enable_uses_learning_phase<InputConfig, false>,
         enable_trainable<InputConfig, false>
    {
        auto operator()() const noexcept
        {
            auto const& input_shape = shape();
            auto const& output_shape = shape();
            auto const& updated_config = InputConfig{*this}.input_shape(input_shape).output_shape(output_shape);
            return std::make_tuple( std::make_shared<InputLayer>(updated_config) );
        }
    };

    ///
    /// @brief Input Layer
    ///
    /// \code{.cpp}
    /// auto input_a = Input().shape( {12} ).name("input_a")();
    /// auto input_b = Input().name("input_b").shape( {24, 34, 3} )();
    /// auto input_c = Input().shape( { 134, 3} )();
    /// \endcode
    ///
    using Input = InputConfig;

    struct InputLayer : Layer< InputLayer >
    {
        InputConfig config_;

        place_holder<tensor<float>> expression_;

        InputLayer( InputConfig const& config ) noexcept : config_{config}, expression_{ config.shape() } { }

        auto operator()() const noexcept
        {
            return expression_;
        }
    };



    struct DenseLayer;

    struct DenseConfig :
        enable_bias_constraint<DenseConfig, "None">,
        enable_bias_initializer<DenseConfig, "zeros">,
        enable_bias_regularizer_l1<DenseConfig, "0.0">,
        enable_bias_regularizer_l2<DenseConfig, "0.0">,
        enable_input_shape<DenseConfig, None>,
        enable_kernel_constraint<DenseConfig, "None">,
        enable_kernel_initializer<DenseConfig, "glorot_uniform">,
        enable_kernel_regularizer_l1<DenseConfig, "0.0">,
        enable_kernel_regularizer_l2<DenseConfig, "0.0">,
        enable_name<DenseConfig, "Dense">,
        enable_output_shape<DenseConfig, None>,
        enable_trainable<DenseConfig, true>,
        enable_units<DenseConfig, None>,
        enable_use_bias<DenseConfig, true>,
        enable_uses_learning_phase<DenseConfig, false>
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            auto const& input_shape = (*prev_layer).config().output_shape();
            auto const& output_shape = std::vector<unsigned long>{ {(*this).units(),} };
            auto const& config = DenseConfig{*this}.input_shape( input_shape ).output_shape( output_shape );

            better_assert( input_shape.size() == 1, fmt::format("Dense layer expects incoming 1D layer, but got {}", input_shape) );
            better_assert( *(input_shape.rbegin()) != None, fmt::format("Dense layer expects an exact dimension, but got {}", input_shape) );

            return std::make_tuple( std::make_shared<DenseLayer>( config ), lt );
        }
    }; // struct DenseConfig

    ///
    /// @brief A normal Dense Layer
    ///
    /// \code{.cpp}
    /// auto input = Input().shape( {12} )();
    //  auto l1 = Dense().name("first_layer").units(127)( input );
    /// auto l2 = Dense().units(129).name("second_layer")( l1 );
    /// \endcode
    ///
    using Dense = DenseConfig;

    struct DenseLayer : Layer< DenseLayer >
    {
        DenseConfig config_;

        variable<tensor<float>> w_;
        variable<tensor<float>> b_;

        DenseLayer( DenseConfig const& config ) noexcept : config_{ config }
        {
            w_ = variable<tensor<float>>( glorot_uniform<float>({(config.input_shape())[0], config.units()}), config.kernel_regularizer_l1(), config.kernel_regularizer_l2() );
            b_ = variable<tensor<float>>( zeros<float>({1, config.units()}), config.bias_regularizer_l1(), config.bias_regularizer_l2(), config.use_bias()/*trainable*/ );
        }

        template< Expression Ex>
        auto operator()( Ex const& ex ) const noexcept
        {
            return ex * w_ + b_;
        }
    }; // struct DenseLayer



    struct ReLULayer;

    struct ReLUConfig :
        enable_name<ReLUConfig, "ReLU">,
        enable_input_shape<ReLUConfig, None>,
        enable_output_shape<ReLUConfig, None>
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            auto const& input_shape = (*prev_layer).config().output_shape();
            auto const& output_shape = input_shape;
            auto const& config = ReLUConfig{*this}.input_shape( input_shape ).output_shape( output_shape );
            return std::make_tuple( std::make_shared<ReLULayer>( config ), lt );
        }
    };

    ///
    /// @brief ReLU layer.
    ///
    /// \code{.cpp}
    /// auto input = Input().shape( {12, 34} )();
    /// auto l1 = ReLU().name("relu")( input );
    /// \endcode
    ///
    using ReLU = ReLUConfig;

    struct ReLULayer : Layer< ReLULayer >
    {
        ReLUConfig config_;
        ReLULayer( ReLUConfig const& config ) noexcept : config_(config) {}

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return relu( ex );
        }
    };


    struct LeakyReLULayer;

    struct LeakyReLUConfig :
        enable_input_shape<LeakyReLUConfig>,
        enable_output_shape<LeakyReLUConfig>,
        enable_name<LeakyReLUConfig, "LeakyReLU">,
        enable_alpha<LeakyReLUConfig, "0.3">
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            auto const& shape =  (*prev_layer).config().output_shape();
            auto const& updated_config = LeakyReLUConfig{*this}.input_shape( shape ).output_shape( shape );

            return std::make_tuple( std::make_shared<LeakyReLULayer>( updated_config ), lt );
        }
    };

    ///
    /// @brief LeakyReLU layer.
    ///
    /// \code{.cpp}
    /// auto input = Input().shape( {12, 3} )();
    /// auto l1 = LeakyReLU().factor(0.1f).( input );
    /// \endcode
    ///
    using LeakyReLU = LeakyReLUConfig;
    using ELU = LeakyReLUConfig;

    struct LeakyReLULayer : Layer<LeakyReLULayer>
    {
        LeakyReLUConfig config_;
        LeakyReLULayer( LeakyReLUConfig const& config ) noexcept : config_{ config } {}

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return leaky_relu(config_.alpha())( ex );
        }
    };



    struct DropoutLayer;

    struct DropoutConfig :
        enable_input_shape<DropoutConfig>,
        enable_output_shape<DropoutConfig>,
        enable_noise_shape<DropoutConfig, None>,
        enable_seed<DropoutConfig, None>,
        enable_name<DropoutConfig, "Dropout">,
        enable_rate<DropoutConfig, "0.3">
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            auto const& shape =  (*prev_layer).config().output_shape();
            auto const& updated_config = DropoutConfig{*this}.input_shape( shape ).output_shape( shape );
            return std::make_tuple( std::make_shared<DropoutLayer>( updated_config ), lt );
        }
    };

    ///
    /// @brief Dropout layer.
    ///
    /// \code{.cpp}
    /// auto input = Input().shape( {12, 3} )();
    /// auto l1 = Dropout().rate(0.1f).( input );
    /// \endcode
    ///
    using Dropout = DropoutConfig;

    struct DropoutLayer : Layer<DropoutLayer>
    {
        DropoutConfig config_;
        DropoutLayer( DropoutConfig const& config ) noexcept : config_{ config } {}

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return drop_out(config_.rate())( ex );
        }
    };


    struct ReshapeLayer;

    struct ReshapeConfig :
        enable_name<ReshapeConfig, "Reshape" >,
        enable_input_shape< ReshapeConfig, None >,
        enable_output_shape< ReshapeConfig, None >,
        enable_target_shape< ReshapeConfig, None >
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& config = ReshapeConfig{ *this }.input_shape( (*(std::get<0>(lt))).config().output_shape() ).output_shape( (*this).target_shape() );
            return std::make_tuple( std::make_shared<ReshapeLayer>(config), lt );
        }
    };

    ///
    /// @brief Reshape layer.
    ///
    /// \code{.cpp}
    /// auto input = Input().shape( {12, 3} )();
    /// auto l1 = Reshape().target_shape({4, 9})( input );
    /// \endcode
    ///
    using Reshape = ReshapeConfig;

    struct ReshapeLayer : Layer<ReshapeLayer>
    {
        ReshapeConfig config_;
        ReshapeLayer( ReshapeConfig const& config ) noexcept : config_{config} {}

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return reshape(config_.target_shape())( ex );
        }
    };


    struct MaxPooling2DLayer;

    struct MaxPooling2DConfig:
        enable_name<MaxPooling2DConfig, "MaxPooling2D">,
        enable_input_shape<MaxPooling2DConfig, None>,
        enable_output_shape<MaxPooling2DConfig, None>,
        enable_pool_size<MaxPooling2DConfig, 2>,
        enable_strides<MaxPooling2DConfig, 1>,
        enable_padding<MaxPooling2DConfig, "None">
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = *(std::get<0>(lt));
            unsigned long const stride = *((*this).pool_size().begin());
            std::vector<unsigned long> o_shape = prev_layer.config().output_shape(); //
            better_assert(o_shape.size()==3, fmt::format("Expecting 3D output, but got {}", o_shape.size()));
            o_shape[0] /= stride;
            o_shape[1] /= stride;

            auto const& config = MaxPooling2DConfig{*this}.input_shape( prev_layer.config().output_shape() ).output_shape( o_shape );
            return std::make_tuple( std::make_shared<MaxPooling2DLayer>( config ), lt );
        }

    };

    ///
    /// @brief MaxPooling2D layer.
    ///
    /// \code{.cpp}
    /// auto input = Input()( {12, 12, 3} );
    /// auto l1 = MaxPooling2D().pool_size({3,})( input );
    /// \endcode
    ///
    using MaxPooling2D = MaxPooling2DConfig;

    struct MaxPooling2DLayer : Layer<MaxPooling2DLayer>
    {
        MaxPooling2DConfig config_;
        MaxPooling2DLayer( MaxPooling2DConfig config ) noexcept : config_(config) {}

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return max_pooling_2d(*(config_.pool_size().begin()))( ex );
        }
    };


    struct AveragePooling2DLayer;

    struct AveragePooling2DConfig:
        enable_name<AveragePooling2DConfig, "AveragePooling2D">,
        enable_input_shape<AveragePooling2DConfig, None>,
        enable_output_shape<AveragePooling2DConfig, None>,
        enable_pool_size<AveragePooling2DConfig, 2>,
        enable_strides<AveragePooling2DConfig, 1>,
        enable_padding<AveragePooling2DConfig, "None">
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = *(std::get<0>(lt));
            unsigned long const stride = *((*this).pool_size().begin());
            std::vector<unsigned long> o_shape = prev_layer.config().output_shape(); //
            better_assert(o_shape.size()==3, fmt::format("Expecting 3D output, but got {}", o_shape.size()));
            o_shape[0] /= stride;
            o_shape[1] /= stride;

            auto const& config = AveragePooling2DConfig{*this}.input_shape( prev_layer.config().output_shape() ).output_shape( o_shape );
            return std::make_tuple( std::make_shared<AveragePooling2DLayer>( config ), lt );
        }

    };

    ///
    /// @brief AveragePooling2D layer.
    ///
    /// \code{.cpp}
    /// auto input = Input()( {12, 12, 3} );
    /// auto l1 = AveragePooling2D().pool_size({3,})( input );
    /// \endcode
    ///
    using AveragePooling2D = AveragePooling2DConfig;

    struct AveragePooling2DLayer : Layer<AveragePooling2DLayer>
    {
        AveragePooling2DConfig config_;
        AveragePooling2DLayer( AveragePooling2DConfig config ) noexcept : config_(config) {}

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return average_pooling_2d(*(config_.pool_size().begin()))( ex );
        }
    };



    struct UpSampling2DLayer;

    struct UpSampling2DConfig:
        enable_name<UpSampling2DConfig, "UpSampling2D">,
        enable_input_shape<UpSampling2DConfig, None>,
        enable_output_shape<UpSampling2DConfig, None>,
        enable_size<UpSampling2DConfig, 2>,
        enable_interpolation<UpSampling2DConfig, "nearest" >
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = *(std::get<0>(lt));
            unsigned long const stride = *((*this).size().begin());
            std::vector<unsigned long> o_shape = prev_layer.config().output_shape(); //
            better_assert(o_shape.size()==3, fmt::format("Expecting 3D output, but got {}", o_shape.size()));
            o_shape[0] /= stride;
            o_shape[1] /= stride;

            auto const& config = UpSampling2DConfig{*this}.input_shape( prev_layer.config().output_shape() ).output_shape( o_shape );
            return std::make_tuple( std::make_shared<UpSampling2DLayer>( config ), lt );
        }

    };

    ///
    /// @brief UpSampling2D layer.
    ///
    /// \code{.cpp}
    /// auto input = Input()( {12, 12, 3} );
    /// auto l1 = UpSampling2D().size({3,})( input );
    /// \endcode
    ///
    using UpSampling2D = UpSampling2DConfig;

    struct UpSampling2DLayer : Layer<UpSampling2DLayer>
    {
        UpSampling2DConfig config_;
        UpSampling2DLayer( UpSampling2DConfig config ) noexcept : config_(config) {}

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return up_sampling_2d(*(config_.size().begin()))( ex );
        }
    };



#if 0



    struct NegativeLayer;

    struct NegativeConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<NegativeLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Negative layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Negative()( input );
    /// \endcode
    ///
    using Negative = NegativeConfig;

    struct NegativeLayer
    {


        NegativeConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return negative( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SumReduceLayer;

    struct SumReduceConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SumReduceLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief SumReduce layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = SumReduce()( input );
    /// \endcode
    ///
    using SumReduce = SumReduceConfig;

    struct SumReduceLayer
    {


        SumReduceConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return sum_reduce( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct ReduceSumLayer;

    struct ReduceSumConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<ReduceSumLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief ReduceSum layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = ReduceSum()( input );
    /// \endcode
    ///
    using ReduceSum = ReduceSumConfig;

    struct ReduceSumLayer
    {


        ReduceSumConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return reduce_sum( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct MeanReduceLayer;

    struct MeanReduceConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<MeanReduceLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief MeanReduce layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = MeanReduce()( input );
    /// \endcode
    ///
    using MeanReduce = MeanReduceConfig;

    struct MeanReduceLayer
    {


        MeanReduceConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return mean_reduce( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct ReduceMeanLayer;

    struct ReduceMeanConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<ReduceMeanLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief ReduceMean layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = ReduceMean()( input );
    /// \endcode
    ///
    using ReduceMean = ReduceMeanConfig;

    struct ReduceMeanLayer
    {


        ReduceMeanConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return reduce_mean( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct MeanLayer;

    struct MeanConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<MeanLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Mean layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Mean()( input );
    /// \endcode
    ///
    using Mean = MeanConfig;

    struct MeanLayer
    {


        MeanConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return mean( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SquareLayer;

    struct SquareConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SquareLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Square layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Square()( input );
    /// \endcode
    ///
    using Square = SquareConfig;

    struct SquareLayer
    {


        SquareConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return square( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct FlattenLayer;

    struct FlattenConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<FlattenLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Flatten layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Flatten()( input );
    /// \endcode
    ///
    using Flatten = FlattenConfig;

    struct FlattenLayer
    {


        FlattenConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return flatten( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            unsigned long const dims = std::accumulate( input_shape_.begin()+1, input_shape_.end(), 1UL, []( auto x, auto y ){ return x*y; } );
            return std::vector<unsigned long>{ {None, dims} };
        }
    };


    struct IdentityLayer;

    struct IdentityConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<IdentityLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Identity layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Identity()( input );
    /// \endcode
    ///
    using Identity = IdentityConfig;

    struct IdentityLayer
    {


        IdentityConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return identity( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct TransposeLayer;

    struct TransposeConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<TransposeLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Transpose layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Transpose()( input );
    /// \endcode
    ///
    using Transpose = TransposeConfig;

    struct TransposeLayer
    {


        TransposeConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return transpose( ex );
        }

        // TODO: fix here
        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };






    struct OnesLikeLayer;

    struct OnesLikeConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<OnesLikeLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief OnesLike layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = OnesLike()( input );
    /// \endcode
    ///
    using OnesLike = OnesLikeConfig;

    struct OnesLikeLayer
    {


        OnesLikeConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return ones_like( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct ZerosLikeLayer;

    struct ZerosLikeConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<ZerosLikeLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief ZerosLike layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = ZerosLike()( input );
    /// \endcode
    ///
    using ZerosLike = ZerosLikeConfig;

    struct ZerosLikeLayer
    {


        ZerosLikeConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return zeros_like( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SignLayer;

    struct SignConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SignLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Sign layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Sign()( input );
    /// \endcode
    ///
    using Sign = SignConfig;

    struct SignLayer
    {


        SignConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return sign( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct AbsLayer;

    struct AbsConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<AbsLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Abs layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Abs()( input );
    /// \endcode
    ///
    using Abs = AbsConfig;

    struct AbsLayer
    {


        AbsConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return abs( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct AcosLayer;

    struct AcosConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<AcosLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Acos layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Acos()( input );
    /// \endcode
    ///
    using Acos = AcosConfig;

    struct AcosLayer
    {


        AcosConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return acos( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct AcoshLayer;

    struct AcoshConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<AcoshLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Acosh layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Acosh()( input );
    /// \endcode
    ///
    using Acosh = AcoshConfig;

    struct AcoshLayer
    {


        AcoshConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return acosh( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct AsinLayer;

    struct AsinConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<AsinLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Asin layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Asin()( input );
    /// \endcode
    ///
    using Asin = AsinConfig;

    struct AsinLayer
    {


        AsinConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return asin( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct AsinhLayer;

    struct AsinhConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<AsinhLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Asinh layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Asinh()( input );
    /// \endcode
    ///
    using Asinh = AsinhConfig;

    struct AsinhLayer
    {


        AsinhConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return asinh( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct AtanLayer;

    struct AtanConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<AtanLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Atan layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Atan()( input );
    /// \endcode
    ///
    using Atan = AtanConfig;

    struct AtanLayer
    {


        AtanConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return atan( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct AtanhLayer;

    struct AtanhConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<AtanhLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Atanh layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Atanh()( input );
    /// \endcode
    ///
    using Atanh = AtanhConfig;

    struct AtanhLayer
    {


        AtanhConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return atanh( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct CbrtLayer;

    struct CbrtConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<CbrtLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Cbrt layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Cbrt()( input );
    /// \endcode
    ///
    using Cbrt = CbrtConfig;

    struct CbrtLayer
    {


        CbrtConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return cbrt( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct CeilLayer;

    struct CeilConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<CeilLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Ceil layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Ceil()( input );
    /// \endcode
    ///
    using Ceil = CeilConfig;

    struct CeilLayer
    {


        CeilConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return ceil( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct CosLayer;

    struct CosConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<CosLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Cos layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Cos()( input );
    /// \endcode
    ///
    using Cos = CosConfig;

    struct CosLayer
    {


        CosConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return cos( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct CoshLayer;

    struct CoshConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<CoshLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Cosh layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Cosh()( input );
    /// \endcode
    ///
    using Cosh = CoshConfig;

    struct CoshLayer
    {


        CoshConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return cosh( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct ErfLayer;

    struct ErfConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<ErfLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Erf layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Erf()( input );
    /// \endcode
    ///
    using Erf = ErfConfig;

    struct ErfLayer
    {


        ErfConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return erf( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct ErfcLayer;

    struct ErfcConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<ErfcLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Erfc layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Erfc()( input );
    /// \endcode
    ///
    using Erfc = ErfcConfig;

    struct ErfcLayer
    {


        ErfcConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return erfc( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct ExpLayer;

    struct ExpConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<ExpLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Exp layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Exp()( input );
    /// \endcode
    ///
    using Exp = ExpConfig;

    struct ExpLayer
    {


        ExpConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return exp( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct Exp2Layer;

    struct Exp2Config
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<Exp2Layer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Exp2 layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Exp2()( input );
    /// \endcode
    ///
    using Exp2 = Exp2Config;

    struct Exp2Layer
    {


        Exp2Config config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return exp2( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct Expm1Layer;

    struct Expm1Config
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<Expm1Layer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Expm1 layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Expm1()( input );
    /// \endcode
    ///
    using Expm1 = Expm1Config;

    struct Expm1Layer
    {


        Expm1Config config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return expm1( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct FabsLayer;

    struct FabsConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<FabsLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Fabs layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Fabs()( input );
    /// \endcode
    ///
    using Fabs = FabsConfig;

    struct FabsLayer
    {


        FabsConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return fabs( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct FloorLayer;

    struct FloorConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<FloorLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Floor layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Floor()( input );
    /// \endcode
    ///
    using Floor = FloorConfig;

    struct FloorLayer
    {


        FloorConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return floor( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct LLrintLayer;

    struct LLrintConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<LLrintLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief LLrint layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = LLrint()( input );
    /// \endcode
    ///
    using LLrint = LLrintConfig;

    struct LLrintLayer
    {


        LLrintConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return llrint( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct LLroundLayer;

    struct LLroundConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<LLroundLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief LLround layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = LLround()( input );
    /// \endcode
    ///
    using LLround = LLroundConfig;

    struct LLroundLayer
    {


        LLroundConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return llround( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct LogLayer;

    struct LogConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<LogLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Log layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Log()( input );
    /// \endcode
    ///
    using Log = LogConfig;

    struct LogLayer
    {


        LogConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return log( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct Log10Layer;

    struct Log10Config
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<Log10Layer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Log10 layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Log10()( input );
    /// \endcode
    ///
    using Log10 = Log10Config;

    struct Log10Layer
    {


        Log10Config config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return log10( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct Log1pLayer;

    struct Log1pConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<Log1pLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Log1p layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Log1p()( input );
    /// \endcode
    ///
    using Log1p = Log1pConfig;

    struct Log1pLayer
    {


        Log1pConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return log1p( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct Log2Layer;

    struct Log2Config
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<Log2Layer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Log2 layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Log2()( input );
    /// \endcode
    ///
    using Log2 = Log2Config;

    struct Log2Layer
    {


        Log2Config config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return log2( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct LrintLayer;

    struct LrintConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<LrintLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Lrint layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Lrint()( input );
    /// \endcode
    ///
    using Lrint = LrintConfig;

    struct LrintLayer
    {


        LrintConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return lrint( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct LroundLayer;

    struct LroundConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<LroundLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Lround layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Lround()( input );
    /// \endcode
    ///
    using Lround = LroundConfig;

    struct LroundLayer
    {


        LroundConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return lround( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct NearbyintLayer;

    struct NearbyintConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<NearbyintLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Nearbyint layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Nearbyint()( input );
    /// \endcode
    ///
    using Nearbyint = NearbyintConfig;

    struct NearbyintLayer
    {


        NearbyintConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return nearbyint( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct RintLayer;

    struct RintConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<RintLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Rint layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Rint()( input );
    /// \endcode
    ///
    using Rint = RintConfig;

    struct RintLayer
    {


        RintConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return rint( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct RoundLayer;

    struct RoundConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<RoundLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Round layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Round()( input );
    /// \endcode
    ///
    using Round = RoundConfig;

    struct RoundLayer
    {


        RoundConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return round( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SinLayer;

    struct SinConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SinLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Sin layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Sin()( input );
    /// \endcode
    ///
    using Sin = SinConfig;

    struct SinLayer
    {


        SinConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return sin( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SinhLayer;

    struct SinhConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SinhLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Sinh layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Sinh()( input );
    /// \endcode
    ///
    using Sinh = SinhConfig;

    struct SinhLayer
    {


        SinhConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return sinh( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SqrtLayer;

    struct SqrtConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SqrtLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Sqrt layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Sqrt()( input );
    /// \endcode
    ///
    using Sqrt = SqrtConfig;

    struct SqrtLayer
    {


        SqrtConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return sqrt( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct TanLayer;

    struct TanConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<TanLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Tan layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Tan()( input );
    /// \endcode
    ///
    using Tan = TanConfig;

    struct TanLayer
    {


        TanConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return tan( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct TanhLayer;

    struct TanhConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<TanhLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Tanh layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Tanh()( input );
    /// \endcode
    ///
    using Tanh = TanhConfig;

    struct TanhLayer
    {


        TanhConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return tanh( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct TruncLayer;

    struct TruncConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<TruncLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Trunc layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Trunc()( input );
    /// \endcode
    ///
    using Trunc = TruncConfig;

    struct TruncLayer
    {


        TruncConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return trunc( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SoftSignLayer;

    struct SoftSignConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SoftSignLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief SoftSign layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = SoftSign()( input );
    /// \endcode
    ///
    using SoftSign = SoftSignConfig;

    struct SoftSignLayer
    {


        SoftSignConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return soft_sign( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct UnitStepLayer;

    struct UnitStepConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<UnitStepLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief UnitStep layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = UnitStep()( input );
    /// \endcode
    ///
    using UnitStep = UnitStepConfig;

    struct UnitStepLayer
    {


        UnitStepConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return unit_step( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct BinaryStepLayer;

    struct BinaryStepConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<BinaryStepLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief BinaryStep layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = BinaryStep()( input );
    /// \endcode
    ///
    using BinaryStep = BinaryStepConfig;

    struct BinaryStepLayer
    {


        BinaryStepConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return banary_step( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct GaussianLayer;

    struct GaussianConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<GaussianLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Gaussian layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Gaussian()( input );
    /// \endcode
    ///
    using Gaussian = GaussianConfig;

    struct GaussianLayer
    {


        GaussianConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return gaussian( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SoftmaxLayer;

    struct SoftmaxConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SoftmaxLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Softmax layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Softmax()( input );
    /// \endcode
    ///
    using Softmax = SoftmaxConfig;

    struct SoftmaxLayer
    {


        SoftmaxConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return softmax( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SeluLayer;

    struct SeluConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SeluLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Selu layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Selu()( input );
    /// \endcode
    ///
    using Selu = SeluConfig;

    struct SeluLayer
    {


        SeluConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return selu( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SoftplusLayer;

    struct SoftplusConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SoftplusLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Softplus layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Softplus()( input );
    /// \endcode
    ///
    using Softplus = SoftplusConfig;

    struct SoftplusLayer
    {


        SoftplusConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return softplus( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SigmoidLayer;

    struct SigmoidConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SigmoidLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Sigmoid layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Sigmoid()( input );
    /// \endcode
    ///
    using Sigmoid = SigmoidConfig;

    struct SigmoidLayer
    {


        SigmoidConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return sigmoid( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct ReLU6Layer;

    struct ReLU6Config
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<ReLU6Layer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief ReLU6 layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = ReLU6()( input );
    /// \endcode
    ///
    using ReLU6 = ReLU6Config;

    struct ReLU6Layer
    {


        ReLU6Config config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return relu6( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct ExponentialLayer;

    struct ExponentialConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<ExponentialLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Exponential layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Exponential()( input );
    /// \endcode
    ///
    using Exponential = ExponentialConfig;

    struct ExponentialLayer
    {


        ExponentialConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return exponential( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct HardSigmoidLayer;

    struct HardSigmoidConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<HardSigmoidLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief HardSigmoid layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = HardSigmoid()( input );
    /// \endcode
    ///
    using HardSigmoid = HardSigmoidConfig;

    struct HardSigmoidLayer
    {


        HardSigmoidConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return hard_sigmoid( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct GeluLayer;

    struct GeluConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<GeluLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Gelu layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Gelu()( input );
    /// \endcode
    ///
    using Gelu = GeluConfig;

    struct GeluLayer
    {


        GeluConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return gelu( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SwishLayer;

    struct SwishConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SwishLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Swish layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Swish()( input );
    /// \endcode
    ///
    using Swish = SwishConfig;

    struct SwishLayer
    {


        SwishConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return swish( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct SiLULayer;

    struct SiLUConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<SiLULayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief SiLU layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = SiLU()( input );
    /// \endcode
    ///
    using SiLU = SiLUConfig;

    struct SiLULayer
    {


        SiLUConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return SiLU( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct CreLULayer;

    struct CreLUConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<CreLULayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief CreLU layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = CreLU()( input );
    /// \endcode
    ///
    using CreLU = CreLUConfig;

    struct CreLULayer
    {


        CreLUConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return crelu( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct TankShrinkLayer;

    struct TankShrinkConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<TankShrinkLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief TankShrink layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = TankShrink()( input );
    /// \endcode
    ///
    using TankShrink = TankShrinkConfig;

    struct TankShrinkLayer
    {


        TankShrinkConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return tank_shrink( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct MishLayer;

    struct MishConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<MishLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Mish layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Mish()( input );
    /// \endcode
    ///
    using Mish = MishConfig;

    struct MishLayer
    {


        MishConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return mish( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct LishtLayer;

    struct LishtConfig
    {
        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<LishtLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Lisht layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Lisht()( input );
    /// \endcode
    ///
    using Lisht = LishtConfig;

    struct LishtLayer
    {


        LishtConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return lisht( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };



    struct Conv2DLayer;

    struct Conv2DConfig
    {
        unsigned long output_channels_;
        std::vector<unsigned long> kernel_size_;
        std::string padding_ = std::string{"valid"};
        std::vector<unsigned long> strides_ = {1, 1};
        std::vector<unsigned long> dilations_ = {1, 1};
        bool use_bias_ = true;
        float kernel_regularizer_l1_ = 0.0f;
        float kernel_regularizer_l2_ = 0.0f;
        float bias_regularizer_l1_ = 0.0f;
        float bias_regularizer_l2_ = 0.0f;

        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            return std::make_tuple( std::make_shared<Conv2DLayer>( lt, *this ), lt );
        }

    };

    using Conv2D = Conv2DConfig;


    struct Conv2DLayer
    {


        Conv2DConfig config_;
        std::vector<unsigned long> input_shape_;

        variable<tensor<float>> w_;
        variable<tensor<float>> b_;

        template< typename... Layers >
        Conv2DLayer( std::tuple<Layers...> const& lt, Conv2DConfig const& config ) noexcept : config_{ config }
        {
            auto const& prev_layer = std::get<0>( lt );
            auto const& prev_output_shape = (*prev_layer).compute_output_shape();
            input_shape_ = std::vector<unsigned long>{ prev_output_shape.begin()+1, prev_output_shape.end() };
            better_assert( input_shape_.size()==3, "Conv2DLayer: expecting an 3D input." );

            unsigned long const kernel_size_x = config_.kernel_size_[0];
            unsigned long const kernel_size_y = config_.kernel_size_.size() == 2 ? config_.kernel_size_[1] : config_.kernel_size_[0];
            unsigned long const input_channels = input_shape_[2];

            w_ = variable<tensor<float>>{ glorot_uniform<float>({config_.output_channels_, kernel_size_x, kernel_size_y, input_channels}),
                                          config_.kernel_regularizer_l1_, config_.kernel_regularizer_l2_ };
            b_ = variable<tensor<float>>{ zeros<float>({1, 1, config_.output_channels_}),
                                          config_.bias_regularizer_l1_, config_.bias_regularizer_l2_, config_.use_bias_ };
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            unsigned long const row_kernel = config_.kernel_size_[0];
            unsigned long const col_kernel = config_.kernel_size_.size() == 2 ? config_.kernel_size_[1] : config_.kernel_size_[0];
            unsigned long const row_input = input_shape_[0];
            unsigned long const col_input = input_shape_[1];
            unsigned long const row_stride = config_.strides_[0];
            unsigned long const col_stride = config_.strides_.size() == 2 ? config_.strides_[1] : config_.strides_[0];
            unsigned long const row_dilation = config_.dilations_[0];
            unsigned long const col_dilation = config_.dilations_.size() == 2 ? config_.dilations_[1] : config_.dilations_[0];
            unsigned long row_padding = 0;
            unsigned long col_padding = 0;
            if ( config_.padding_ == "same" )
            {
                unsigned long const row_padding_total = (row_kernel + (row_kernel - 1) * (row_dilation - 1) - row_stride);
                better_assert( !(row_padding_total & 0x1), "Expecting total row padding to be even, consider apply zero padding to rescue." );
                unsigned long const col_padding_total = (col_kernel + (col_kernel - 1) * (col_dilation - 1) - col_stride);
                better_assert( !(col_padding_total & 0x1), "Expecting total col padding to be even, consider apply zero-padding to rescue." );
                row_padding = ((row_kernel&1)+row_padding_total) >> 1;
                col_padding = ((col_kernel&1)+col_padding_total) >> 1;
            }

            unsigned long const row_output = ( row_input + 2 * row_padding - ( row_dilation * (row_kernel - 1) + 1 ) ) / row_stride + 1;
            unsigned long const col_output = ( col_input + 2 * col_padding - ( col_dilation * (col_kernel - 1) + 1 ) ) / col_stride + 1;
            return std::vector<unsigned long>{ {None, row_output, col_output, config_.output_channels_} };
        }

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            unsigned long const input_x = input_shape_[0];
            unsigned long const input_y = input_shape_[1];
            unsigned long const stride_x = config_.strides_[0];
            unsigned long const stride_y = config_.strides_.size() == 2 ? config_.strides_[1] : config_.strides_[0];
            unsigned long const dilation_row = config_.dilations_[0];
            unsigned long const dilation_col = config_.dilations_.size() == 2 ? config_.dilations_[1] : config_.dilations_[0];
            return conv2d( input_x, input_y, stride_x, stride_y, dilation_row, dilation_col, config_.padding_ )( ex, w_ ) + b_;
        }

    };





    struct BatchNormalizationLayer;

    struct BatchNormalizationConfig
    {
        float threshold_ = 0.95f;
        float kernel_regularizer_l1_ = 0.0f;
        float kernel_regularizer_l2_ = 0.0f;
        float bias_regularizer_l1_ = 0.0f;
        float bias_regularizer_l2_ = 0.0f;

        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<BatchNormalizationLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief BatchNormalization layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = BatchNormalization(0.9f)( input );
    /// \endcode
    ///
    using BatchNormalization = BatchNormalizationConfig;

    struct BatchNormalizationLayer
    {


        BatchNormalizationConfig config_;
        std::vector<unsigned long> input_shape_;

        variable<tensor<float>> gamma_;
        variable<tensor<float>> beta_;

        BatchNormalizationLayer( BatchNormalizationConfig const& config, std::vector<unsigned long> const& input_shape ) noexcept: config_{config}, input_shape_{input_shape}
        {
            unsigned long const last_dim = *(input_shape_.rbegin());
            gamma_ = variable{ ones<float>( {last_dim, }  ), config.kernel_regularizer_l1_, config_.kernel_regularizer_l2_ };
            beta_ = variable{ zeros<float>( {last_dim, } ), config.bias_regularizer_l1_, config_.bias_regularizer_l2_ };
        }

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return batch_normalization( config_.threshold_ )( ex, gamma_, beta_ );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };


    struct ConcatenateLayer;

    struct ConcatenateConfig
    {
        unsigned long axis_ = None;

        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<ConcatenateLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Concatenate two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {12, 34, 5} );
    /// auto ab = Concatenate()( a, b );
    /// \endcode
    ///
    using Concatenate = ConcatenateConfig;


    struct ConcatenateLayer
    {


        ConcatenateConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        ConcatenateLayer( ConcatenateConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape } { }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return concatenate( config_.axis_ )( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            better_assert( lhs_input_shape_.size() == lhs_input_shape_.size() );
            std::vector<unsigned long> ans{ lhs_input_shape_ };
            unsigned long axis = config_.axis_;
            axis = axis == None ? ans.size()-1 : axis;
            ans[axis] += rhs_input_shape_[axis];
            return ans;
        }
    };




    struct AddLayer;

    struct AddConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<AddLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Add two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {1, 34, 2} );
    /// auto ab = Add()( a, b ); // broadcasting
    /// \endcode
    ///
    using Add = AddConfig;


    struct AddLayer
    {


        AddConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        AddLayer( AddConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape } { }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return ex + ey;
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            unsigned long const dims = std::max( lhs_input_shape_.size(), rhs_input_shape_.size() );

            std::vector<unsigned long> lhs_input_shape = lhs_input_shape_;
            while ( lhs_input_shape.size() < dims )
                lhs_input_shape.insert( lhs_input_shape.begin(), 1UL );

            std::vector<unsigned long> rhs_input_shape = rhs_input_shape_;
            while ( rhs_input_shape.size() < dims )
                rhs_input_shape.insert( rhs_input_shape.begin(), 1UL );

            std::vector<unsigned long> ans( dims, 0 );
            for ( auto idx : range( dims ) )
                ans[idx] = std::max( lhs_input_shape[idx], rhs_input_shape[idx] );

            return ans;
        }
    };



    struct SubtractLayer;

    struct SubtractConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<SubtractLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Subtract two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {1, 1, 2} );
    /// auto ab = Subtract()( a, b ); // broadcasting
    /// \endcode
    ///
    using Subtract = SubtractConfig;


    struct SubtractLayer
    {


        SubtractConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        SubtractLayer( SubtractConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape } { }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return ex - ey;
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            unsigned long const dims = std::max( lhs_input_shape_.size(), rhs_input_shape_.size() );

            std::vector<unsigned long> lhs_input_shape = lhs_input_shape_;
            while ( lhs_input_shape.size() < dims )
                lhs_input_shape.insert( lhs_input_shape.begin(), 1UL );

            std::vector<unsigned long> rhs_input_shape = rhs_input_shape_;
            while ( rhs_input_shape.size() < dims )
                rhs_input_shape.insert( rhs_input_shape.begin(), 1UL );

            std::vector<unsigned long> ans( dims, 0 );
            for ( auto idx : range( dims ) )
                ans[idx] = std::max( lhs_input_shape[idx], rhs_input_shape[idx] );

            return ans;
        }
    };



    struct MultiplyLayer;

    struct MultiplyConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<MultiplyLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Elementwise-multiply two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {12, 34, 2} );
    /// auto ab = Multiply()( a, b );
    /// \endcode
    ///
    using Multiply = MultiplyConfig;


    struct MultiplyLayer
    {


        MultiplyConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        MultiplyLayer( MultiplyConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape } { }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return hadamard_product( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };



    struct DotLayer;

    struct DotConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<DotLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Multiply two matrices;
    ///
    /// \code{.cpp}
    /// auto a = Input( {3, 3} );
    /// auto b = Input( {3, 3} );
    /// auto ab = Dot()( a, b );
    /// \endcode
    ///
    using Dot = DotConfig;


    struct DotLayer
    {


        DotConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        DotLayer( DotConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape } { }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return ex * ey;
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return std::vector<unsigned long>{ { lhs_input_shape_[1], rhs_input_shape_[2] } };
        }
    };



    struct MaximumLayer;

    struct MaximumConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<MaximumLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Element-wise maximum.
    ///
    /// \code{.cpp}
    /// auto a = Input( {3, 3} );
    /// auto b = Input( {3, 3} );
    /// auto ab = Maximum()( a, b );
    /// \endcode
    ///
    using Maximum = MaximumConfig;


    struct MaximumLayer
    {


        MaximumConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        MaximumLayer( MaximumConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape } { }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return maximum( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };





    struct MinimumLayer;

    struct MinimumConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<MinimumLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Element-wise minimum.
    ///
    /// \code{.cpp}
    /// auto a = Input( {3, 3} );
    /// auto b = Input( {3, 3} );
    /// auto ab = Minimum()( a, b );
    /// \endcode
    ///
    using Minimum = MinimumConfig;


    struct MinimumLayer
    {


        MinimumConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        MinimumLayer( MinimumConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape }
        {
            better_assert( lhs_input_shape_ == rhs_input_shape_, "Error: expecting same shape." );
        }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return minimum( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };




    struct Atan2Layer;

    struct Atan2Config
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<Atan2Layer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Element-wise atan2
    ///
    /// \code{.cpp}
    /// auto a = Input( {3, 3} );
    /// auto b = Input( {3, 3} );
    /// auto ab = Atan2()( a, b );
    /// \endcode
    ///
    using Atan2 = Atan2Config;

    struct Atan2Layer
    {


        Atan2Config config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        Atan2Layer( Atan2Config config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape } { }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return atan2( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };




    struct ClipLayer;

    struct ClipConfig
    {
        float lower_ = eps;
        float upper_ = std::numeric_limits<float>::infinity();

        template< typename... Layers >
        auto operator()( std::tuple<Layers...> const& lt ) const noexcept
        {
            auto const& prev_layer = std::get<0>( lt );
            return std::make_tuple( std::make_shared<ClipLayer>(*this, (*prev_layer).compute_output_shape()), lt );
        }

    };

    ///
    /// @brief Clip layer.
    ///
    /// \code{.cpp}
    /// auto input = Input( {12, 3} );
    /// auto l1 = Clip(0.1f)( input ); // filter values less than 0.1f
    /// \endcode
    ///
    using Clip = ClipConfig;

    struct ClipLayer
    {


        ClipConfig config_;
        std::vector<unsigned long> input_shape_;

        template< Expression Ex>
        auto operator()(const Ex& ex ) const noexcept
        {
            return clip(config_.lower_, config_.upper_)( ex );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept { return input_shape_; }
    };




    struct EqualLayer;

    struct EqualConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<EqualLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Elementwise-multiply two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {12, 34, 2} );
    /// auto ab = Equal()( a, b );
    /// \endcode
    ///
    using Equal = EqualConfig;


    struct EqualLayer
    {
        EqualConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        EqualLayer( EqualConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape }
        {
            better_assert( lhs_input_shape_ == rhs_input_shape_, "EqualLayer: expecting same input shape." );
        }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return equal( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };







#endif





}//namespace ceras::keras

#endif//BIYVCSVIYREATABTGLTJXSOTHRCQIOUAUKNTNMSACVTQWWTIDVLGGTFAMNIIFJOXOHFDCEAWI

