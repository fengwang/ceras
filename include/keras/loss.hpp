#ifndef NLBVKWXCAJJTLJCFYILHHMPTRNTFOCOJRCMWOHXHAFHIKNXXFWLBCCSDTYINIPDSJELCXOHAD
#define NLBVKWXCAJJTLJCFYILHHMPTRNTFOCOJRCMWOHXHAFHIKNXXFWLBCCSDTYINIPDSJELCXOHAD

#include "./layer.hpp"
#include "../operation.hpp"

namespace ceras::keras
{


    struct MeanSquaredLogarithmicErrorLayer;

    struct MeanSquaredLogarithmicErrorConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<MeanSquaredLogarithmicErrorLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Elementwise-multiply two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {12, 34, 2} );
    /// auto ab = MeanSquaredLogarithmicError()( a, b );
    /// \endcode
    ///
    using MeanSquaredLogarithmicError = MeanSquaredLogarithmicErrorConfig;


    struct MeanSquaredLogarithmicErrorLayer
    {
        using category = LayerTag;

        MeanSquaredLogarithmicErrorConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        MeanSquaredLogarithmicErrorLayer( MeanSquaredLogarithmicErrorConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape }
        {
            better_assert( lhs_input_shape_ == rhs_input_shape_, "MeanSquaredLogarithmicErrorLayer: expecting same input shape." );
        }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return mean_squared_logarithmic_error( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };



    struct SquaredLossLayer;

    struct SquaredLossConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<SquaredLossLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Elementwise-multiply two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {12, 34, 2} );
    /// auto ab = SquaredLoss()( a, b );
    /// \endcode
    ///
    using SquaredLoss = SquaredLossConfig;


    struct SquaredLossLayer
    {
        using category = LayerTag;

        SquaredLossConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        SquaredLossLayer( SquaredLossConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape }
        {
            better_assert( lhs_input_shape_ == rhs_input_shape_, "SquaredLossLayer: expecting same input shape." );
        }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return squared_loss( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };



    struct MeanSquaredErrorLayer;

    struct MeanSquaredErrorConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<MeanSquaredErrorLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Elementwise-multiply two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {12, 34, 2} );
    /// auto ab = MeanSquaredError()( a, b );
    /// \endcode
    ///
    using MeanSquaredError = MeanSquaredErrorConfig;


    struct MeanSquaredErrorLayer
    {
        using category = LayerTag;

        MeanSquaredErrorConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        MeanSquaredErrorLayer( MeanSquaredErrorConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape }
        {
            better_assert( lhs_input_shape_ == rhs_input_shape_, "MeanSquaredErrorLayer: expecting same input shape." );
        }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return mean_squared_error( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };

    using MSE = MeanSquaredErrorLayer;
    using mean_squared_error = MSE;
    using mse = MSE;



    struct ABSLossLayer;

    struct ABSLossConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<ABSLossLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Elementwise-multiply two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {12, 34, 2} );
    /// auto ab = ABSLoss()( a, b );
    /// \endcode
    ///
    using ABSLoss = ABSLossConfig;


    struct ABSLossLayer
    {
        using category = LayerTag;

        ABSLossConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        ABSLossLayer( ABSLossConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape }
        {
            better_assert( lhs_input_shape_ == rhs_input_shape_, "ABSLossLayer: expecting same input shape." );
        }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return abs_loss( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };



    struct MeanAbsoluteErrorLayer;

    struct MeanAbsoluteErrorConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<MeanAbsoluteErrorLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Elementwise-multiply two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {12, 34, 2} );
    /// auto ab = MeanAbsoluteError()( a, b );
    /// \endcode
    ///
    using MeanAbsoluteError = MeanAbsoluteErrorConfig;
    using MAE = MeanAbsoluteErrorConfig;
    using mean_absolute_error = MAE;
    using mae = MAE;


    struct MeanAbsoluteErrorLayer
    {
        using category = LayerTag;

        MeanAbsoluteErrorConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        MeanAbsoluteErrorLayer( MeanAbsoluteErrorConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape }
        {
            better_assert( lhs_input_shape_ == rhs_input_shape_, "MeanAbsoluteErrorLayer: expecting same input shape." );
        }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return mean_absolute_error( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };



    struct CrossEntropyLayer;

    struct CrossEntropyConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<CrossEntropyLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Elementwise-multiply two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {12, 34, 2} );
    /// auto ab = CrossEntropy()( a, b );
    /// \endcode
    ///
    using CrossEntropy = CrossEntropyConfig;

    using categorical_crossentropy = CrossEntropyConfig;
    using cce = categorical_crossentropy;
    using CCE = cce;


    struct CrossEntropyLayer
    {
        using category = LayerTag;

        CrossEntropyConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        CrossEntropyLayer( CrossEntropyConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape }
        {
            better_assert( lhs_input_shape_ == rhs_input_shape_, "CrossEntropyLayer: expecting same input shape." );
        }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return cross_entropy( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };


    struct BinaryCrossEntropyLayer;

    struct BinaryCrossEntropyConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<BinaryCrossEntropyLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Elementwise-multiply two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {12, 34, 2} );
    /// auto ab = BinaryCrossEntropy()( a, b );
    /// \endcode
    ///
    using BinaryCrossEntropy = BinaryCrossEntropyConfig;


    struct BinaryCrossEntropyLayer
    {
        using category = LayerTag;

        BinaryCrossEntropyConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        BinaryCrossEntropyLayer( BinaryCrossEntropyConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape }
        {
            better_assert( lhs_input_shape_ == rhs_input_shape_, "BinaryCrossEntropyLayer: expecting same input shape." );
        }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return binary_cross_entropy( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };

    using binary_crossentropy = BinaryCrossEntropyLayerConfig;
    using bce = binary_crossentropy;
    using BCE = bce;



    struct HingeLossLayer;

    struct HingeLossConfig
    {
        template< typename... LLayers, typename ... RLayers >
        auto operator()( std::tuple<LLayers...> const& lt, std::tuple<RLayers...> const& rt ) const noexcept
        {
            auto const& prev_layer_0 = std::get<0>( lt );
            auto const& prev_layer_1 = std::get<0>( rt );
            return std::make_tuple( std::make_shared<HingeLossLayer>(*this, (*prev_layer_0).compute_output_shape(), (*prev_layer_1).compute_output_shape()), lt, rt );
        }
    };

    ///
    /// @brief Elementwise-multiply two layers.
    ///
    /// \code{.cpp}
    /// auto a = Input( {12, 34, 2} );
    /// auto b = Input( {12, 34, 2} );
    /// auto ab = HingeLoss()( a, b );
    /// \endcode
    ///
    using HingeLoss = HingeLossConfig;


    struct HingeLossLayer
    {
        using category = LayerTag;

        HingeLossConfig config_;
        std::vector<unsigned long> lhs_input_shape_;
        std::vector<unsigned long> rhs_input_shape_;

        HingeLossLayer( HingeLossConfig config, std::vector<unsigned long> const& lhs_shape, std::vector<unsigned long> const& rhs_shape ) noexcept :
        config_{ config }, lhs_input_shape_{ lhs_shape }, rhs_input_shape_{ rhs_shape }
        {
            better_assert( lhs_input_shape_ == rhs_input_shape_, "HingeLossLayer: expecting same input shape." );
        }

        template< Expression Ex, Expression Ey>
        auto operator()(Ex const& ex, Ey const& ey) const noexcept
        {
            return hinge_loss( ex, ey );
        }

        std::vector<unsigned long> compute_output_shape() const noexcept
        {
            return lhs_input_shape_;
        }
    };













}//namespace ceras::keras

#endif//NLBVKWXCAJJTLJCFYILHHMPTRNTFOCOJRCMWOHXHAFHIKNXXFWLBCCSDTYINIPDSJELCXOHAD

