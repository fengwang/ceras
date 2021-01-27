#ifndef DGIDKOSTMBYAWNFFKALGHEJNQNOVEFFCRCJMOVVQQSNSJQPJMMTFBIVANGSXAWWUKGMEDNOAM
#define DGIDKOSTMBYAWNFFKALGHEJNQNOVEFFCRCJMOVVQQSNSJQPJMMTFBIVANGSXAWWUKGMEDNOAM

#include "../includes.hpp"
#include "./keras_utils.hpp"
#include "../place_holder.hpp"
#include "../variable.hpp"
#include "../operation.hpp"
#include "../tensor.hpp"

namespace Keras
{

    // for layers
    // - record information when calling constructor
    // - buildup computation graph when calling ()
    // - each layer has a _config layer for interfacing

    template< typename Concrete_Layer >
    struct layer_default
    {
        bool trainable_ = true;
        std::string name_;
    };


    struct Input : layer_default<Input>
    {
        std::vector<std::optional<unsigned long>> shape_;
        ceras::place_holder<tensor<float>> place_holder_;

        template< typename ... Integer_Or_NuLL>
        Input( Integer_Or_NuLL ... args )
        {
            shape_.reserve(sizeof...(args));
            (shape_.push_back(std::optional<unsigned long>{std::forward<decltype(args)>(args)}), ...);

            (*this).trainable_ = false;
            (*this).name_ = "Input Layer";
        }

        std::vector<std::optional<unsigned long>> output_shape() const noexcept
        {
            return shape_;
        }

        auto operator()() const noexcept
        {
            return place_holder_;
        }
    };//Input
    using input = Input;

    template< typename Conv2D_Config, typename Input_Layer >
    struct conv2d : layer_default<conv2d<Conv2D_Config, Input_Layer>>
    {
        Conv2D_Config config_;
        Input_Layer input_layer_;

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
            auto input_operation = input_layer_();
            auto conv_kernel = ceras::variable{ ceras::randn<float>({output_filters, row_kernel_, col_kernel_, input_filters_}, 0.0, 10.0f/(std::sqrt(output_filters_*row_kernel_*col_kernel_*input_filters_))) }; //TODO: apply kernel initializer
            std::string const padding = Conv2D_Config::padding_type::data_;
            auto output_operation = ceras::conv2d(row_input_, col_input_, row_stride_, col_stride_, row_dilation_, col_dilation_)(input_operation, conv_kernel);

            if constexpr( std::is_same_v<Conv2D_Config::use_bias_type, use_bias<true>> )
                return output_operation + ceras::variable{ceras::zeros<float>( {output_filters_,} )};
            else
                return output_operation;
        }

        conv2d( Conv2D_Config const& conv2d_config, Input_Layer const& input_layer ) : config_{conv2d_config}, input_layer_{input_layer}
        {
            input_shape_ = input_layer.output_shape_;
            better_assert( input_shape_.size() == 3, "Expecting a 3D input, but got ", input_shape_.size() );

            std::tie( row_input_, col_input_ input_filters_ ) = std::make_tuple( input_shape_[0].value_or(0), input_shape_[1].value_or(0), input_shape_[3].value() );
            std::tie( row_kernel_, col_kernel_ ) = Conv2D_Config::kernel_type::data_;
            std::tie( row_stride_, col_stride_ ) = Conv2D_Config::strides_type::data_;
            std::tie( row_dilation_, col_dilation_ ) = Conv2D_Config::dilation_rate_type::data_;
            output_filters_ = Conv2D_Config::filters_type::data_;
            padding_ = Conv2D_Config::padding_type::data_;

            calculate_output_shape();
        }

        std::vector<std::optional<unsigned long>> output_shape() const noexcept
        {
            return output_shape_;
        }

        auto get_config() const noexcept
        {
            return config_;
        }

        void calculate_output_shape()
        {
            // the last dim is the filters in config
            unsigned long const row_input = input_shape_[0].value_or(0);
            unsigned long const col_input = input_shape_[1].value_or(0);

            auto const [row_kernel, col_kernel] = Conv2D_Config::kernel_type::data_;
            auto const [row_stride, col_stride] = Conv2D_Config::strides_type::data_;
            auto const [row_dilation, col_dilation] = Conv2D_Config::dilation_rate_type::data_;

            unsigned long row_padding = 0;
            unsigned long col_padding = 0;
            if costexpr( std::is_same_v<Conv2D_Config::padding_type,padding<"same">> )
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
            output_shape_[2] = Conv2D_Config::filters_type::data_;
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

        template< typename Input_Layer >
        auto operator ()( Input_layer const& input_layer ) const noexcept
        {
            return conv2d{*this, input_layer};
        }
    };

    template< typename ... Args >
    using Conv2D = Conv2D_Config<Args...>; // to make 'Conv2D<...>{}( input_layer );' work


#if 0
    pseudo code:
        auto input = Input( {std::nullopt, std::nullopt, 3} );
        auto layer_1 = Conv2D< 16, {3, 3}, {1, 1}, "valid" )( input );

#endif

}//namespace Keras

#endif//DGIDKOSTMBYAWNFFKALGHEJNQNOVEFFCRCJMOVVQQSNSJQPJMMTFBIVANGSXAWWUKGMEDNOAM

