#ifndef VIJTSJOWXEEBOVDLNBJPNYPFSYJSCXBQKYYFYWQBAJDQAJNENJIHPXJFXJTNKRGWDTXWVTXJM
#define VIJTSJOWXEEBOVDLNBJPNYPFSYJSCXBQKYYFYWQBAJDQAJNENJIHPXJFXJTNKRGWDTXWVTXJM

#include "../includes.hpp"
#include "../utils/float32.hpp"
#include "../utils/string.hpp"

namespace ceras::keras
{

#if 0

Decouple class fields into field-classes.

Example:

    struct my_layer : enable_shape<a, 1, 1>, enable_input_shape<a, 2, 2>, enable_name<a, "...." >, enable_alpha<a, "1.0" > {};
    auto x = my_layer().shape( {1, 2, 3, 4} ).input_shape( { 4, 5, 6} ).name( "nameofx" ).alpha( 3.14159265f );

#endif

    template< typename T, typename U,  U... default_values >
    struct underlying_default_value
    {
        T val_; // is not a static value

        underlying_default_value() noexcept : val_{default_values...} {}
    };


    template< typename Container >
    struct enable_keras_layer_tag
    {
        typedef int keras_layer_tag; // for dispatching, just in case
    };


    //----------------------------------------------------------------------------------------------------------------------------
    //
    // The code below is generated using jinja2 template, located at './misc/render.py'
    //
    //----------------------------------------------------------------------------------------------------------------------------


    ///
    /// @brief Adding shape to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_shape<X, shape_vals...> {};
    /// auto x = X().shape( new_shape_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_shape
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> shape_;

        Concrete shape( std::vector<unsigned long> new_shape ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.shape_.val_ = new_shape;
            return ans;
        }

        std::vector<unsigned long> shape() const noexcept
        {
            return shape_.val_;
        }
    };

    ///
    /// @brief Adding input_shape to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_input_shape<X, input_shape_vals...> {};
    /// auto x = X().input_shape( new_input_shape_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_input_shape
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> input_shape_;

        Concrete input_shape( std::vector<unsigned long> new_input_shape ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.input_shape_.val_ = new_input_shape;
            return ans;
        }

        std::vector<unsigned long> input_shape() const noexcept
        {
            return input_shape_.val_;
        }
    };

    ///
    /// @brief Adding output_shape to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_output_shape<X, output_shape_vals...> {};
    /// auto x = X().output_shape( new_output_shape_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_output_shape
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> output_shape_;

        Concrete output_shape( std::vector<unsigned long> new_output_shape ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.output_shape_.val_ = new_output_shape;
            return ans;
        }

        std::vector<unsigned long> output_shape() const noexcept
        {
            return output_shape_.val_;
        }
    };

    ///
    /// @brief Adding target_shape to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_target_shape<X, target_shape_vals...> {};
    /// auto x = X().target_shape( new_target_shape_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_target_shape
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> target_shape_;

        Concrete target_shape( std::vector<unsigned long> new_target_shape ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.target_shape_.val_ = new_target_shape;
            return ans;
        }

        std::vector<unsigned long> target_shape() const noexcept
        {
            return target_shape_.val_;
        }
    };

    ///
    /// @brief Adding noise_shape to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_noise_shape<X, noise_shape_vals...> {};
    /// auto x = X().noise_shape( new_noise_shape_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_noise_shape
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> noise_shape_;

        Concrete noise_shape( std::vector<unsigned long> new_noise_shape ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.noise_shape_.val_ = new_noise_shape;
            return ans;
        }

        std::vector<unsigned long> noise_shape() const noexcept
        {
            return noise_shape_.val_;
        }
    };

    ///
    /// @brief Adding cropping to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_cropping<X, cropping_vals...> {};
    /// auto x = X().cropping( new_cropping_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_cropping
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> cropping_;

        Concrete cropping( std::vector<unsigned long> new_cropping ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.cropping_.val_ = new_cropping;
            return ans;
        }

        std::vector<unsigned long> cropping() const noexcept
        {
            return cropping_.val_;
        }
    };

    ///
    /// @brief Adding paddings to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_paddings<X, paddings_vals...> {};
    /// auto x = X().paddings( new_paddings_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_paddings
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> paddings_;

        Concrete paddings( std::vector<unsigned long> new_paddings ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.paddings_.val_ = new_paddings;
            return ans;
        }

        std::vector<unsigned long> paddings() const noexcept
        {
            return paddings_.val_;
        }
    };

    ///
    /// @brief Adding size to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_size<X, size_vals...> {};
    /// auto x = X().size( new_size_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_size
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> size_;

        Concrete size( std::vector<unsigned long> new_size ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.size_.val_ = new_size;
            return ans;
        }

        std::vector<unsigned long> size() const noexcept
        {
            return size_.val_;
        }
    };

    ///
    /// @brief Adding kernel_size to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_kernel_size<X, kernel_size_vals...> {};
    /// auto x = X().kernel_size( new_kernel_size_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_kernel_size
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> kernel_size_;

        Concrete kernel_size( std::vector<unsigned long> new_kernel_size ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.kernel_size_.val_ = new_kernel_size;
            return ans;
        }

        std::vector<unsigned long> kernel_size() const noexcept
        {
            return kernel_size_.val_;
        }
    };

    ///
    /// @brief Adding strides to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_strides<X, strides_vals...> {};
    /// auto x = X().strides( new_strides_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_strides
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> strides_;

        Concrete strides( std::vector<unsigned long> new_strides ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.strides_.val_ = new_strides;
            return ans;
        }

        std::vector<unsigned long> strides() const noexcept
        {
            return strides_.val_;
        }
    };

    ///
    /// @brief Adding dilation_rate to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_dilation_rate<X, dilation_rate_vals...> {};
    /// auto x = X().dilation_rate( new_dilation_rate_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_dilation_rate
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> dilation_rate_;

        Concrete dilation_rate( std::vector<unsigned long> new_dilation_rate ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.dilation_rate_.val_ = new_dilation_rate;
            return ans;
        }

        std::vector<unsigned long> dilation_rate() const noexcept
        {
            return dilation_rate_.val_;
        }
    };

    ///
    /// @brief Adding output_padding to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_output_padding<X, output_padding_vals...> {};
    /// auto x = X().output_padding( new_output_padding_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_output_padding
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> output_padding_;

        Concrete output_padding( std::vector<unsigned long> new_output_padding ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.output_padding_.val_ = new_output_padding;
            return ans;
        }

        std::vector<unsigned long> output_padding() const noexcept
        {
            return output_padding_.val_;
        }
    };

    ///
    /// @brief Adding shared_axes to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_shared_axes<X, shared_axes_vals...> {};
    /// auto x = X().shared_axes( new_shared_axes_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_shared_axes
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> shared_axes_;

        Concrete shared_axes( std::vector<unsigned long> new_shared_axes ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.shared_axes_.val_ = new_shared_axes;
            return ans;
        }

        std::vector<unsigned long> shared_axes() const noexcept
        {
            return shared_axes_.val_;
        }
    };

    ///
    /// @brief Adding axes to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_axes<X, axes_vals...> {};
    /// auto x = X().axes( new_axes_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_axes
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> axes_;

        Concrete axes( std::vector<unsigned long> new_axes ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.axes_.val_ = new_axes;
            return ans;
        }

        std::vector<unsigned long> axes() const noexcept
        {
            return axes_.val_;
        }
    };

    ///
    /// @brief Adding dims to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_dims<X, dims_vals...> {};
    /// auto x = X().dims( new_dims_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_dims
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> dims_;

        Concrete dims( std::vector<unsigned long> new_dims ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.dims_.val_ = new_dims;
            return ans;
        }

        std::vector<unsigned long> dims() const noexcept
        {
            return dims_.val_;
        }
    };

    ///
    /// @brief Adding center to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_center<X, center_val> {};
    /// auto x = X().center( new_center_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, bool default_values >
    struct enable_center
    {
        underlying_default_value<bool, bool, default_values> center_;

        Concrete center( bool new_center ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.center_.val_ = new_center;
            return ans;
        }

        bool center() const noexcept
        {
            return center_.val_;
        }
    };

    ///
    /// @brief Adding mask_zero to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_mask_zero<X, mask_zero_val> {};
    /// auto x = X().mask_zero( new_mask_zero_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, bool default_values >
    struct enable_mask_zero
    {
        underlying_default_value<bool, bool, default_values> mask_zero_;

        Concrete mask_zero( bool new_mask_zero ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.mask_zero_.val_ = new_mask_zero;
            return ans;
        }

        bool mask_zero() const noexcept
        {
            return mask_zero_.val_;
        }
    };

    ///
    /// @brief Adding scale to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_scale<X, scale_val> {};
    /// auto x = X().scale( new_scale_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, bool default_values >
    struct enable_scale
    {
        underlying_default_value<bool, bool, default_values> scale_;

        Concrete scale( bool new_scale ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.scale_.val_ = new_scale;
            return ans;
        }

        bool scale() const noexcept
        {
            return scale_.val_;
        }
    };

    ///
    /// @brief Adding use_scale to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_use_scale<X, use_scale_val> {};
    /// auto x = X().use_scale( new_use_scale_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, bool default_values >
    struct enable_use_scale
    {
        underlying_default_value<bool, bool, default_values> use_scale_;

        Concrete use_scale( bool new_use_scale ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.use_scale_.val_ = new_use_scale;
            return ans;
        }

        bool use_scale() const noexcept
        {
            return use_scale_.val_;
        }
    };

    ///
    /// @brief Adding use_bias to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_use_bias<X, use_bias_val> {};
    /// auto x = X().use_bias( new_use_bias_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, bool default_values >
    struct enable_use_bias
    {
        underlying_default_value<bool, bool, default_values> use_bias_;

        Concrete use_bias( bool new_use_bias ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.use_bias_.val_ = new_use_bias;
            return ans;
        }

        bool use_bias() const noexcept
        {
            return use_bias_.val_;
        }
    };

    ///
    /// @brief Adding normalize to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_normalize<X, normalize_val> {};
    /// auto x = X().normalize( new_normalize_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, bool default_values >
    struct enable_normalize
    {
        underlying_default_value<bool, bool, default_values> normalize_;

        Concrete normalize( bool new_normalize ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.normalize_.val_ = new_normalize;
            return ans;
        }

        bool normalize() const noexcept
        {
            return normalize_.val_;
        }
    };

    ///
    /// @brief Adding alpha to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_alpha<X, alpha_val> {};
    /// auto x = X().alpha( new_alpha_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::float32 default_values="0.0" >
    struct enable_alpha
    {
        //underlying_default_value<float, ceras::float32, default_values> alpha_;
        underlying_default_value<float, decltype(default_values), default_values> alpha_;

        Concrete alpha( float new_alpha ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.alpha_.val_ = new_alpha;
            return ans;
        }

        float alpha() const noexcept
        {
            return alpha_.val_;
        }
    };

    ///
    /// @brief Adding epsilon to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_epsilon<X, epsilon_val> {};
    /// auto x = X().epsilon( new_epsilon_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::float32 default_values >
    struct enable_epsilon
    {
        //underlying_default_value<float, ceras::float32, default_values> epsilon_;
        underlying_default_value<float, decltype(default_values), default_values> epsilon_;

        Concrete epsilon( float new_epsilon ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.epsilon_.val_ = new_epsilon;
            return ans;
        }

        float epsilon() const noexcept
        {
            return epsilon_.val_;
        }
    };

    ///
    /// @brief Adding mask_value to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_mask_value<X, mask_value_val> {};
    /// auto x = X().mask_value( new_mask_value_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::float32 default_values >
    struct enable_mask_value
    {
        underlying_default_value<float, decltype(default_values), default_values> mask_value_;

        Concrete mask_value( float new_mask_value ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.mask_value_.val_ = new_mask_value;
            return ans;
        }

        float mask_value() const noexcept
        {
            return mask_value_.val_;
        }
    };

    ///
    /// @brief Adding max_value to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_max_value<X, max_value_val> {};
    /// auto x = X().max_value( new_max_value_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::float32 default_values >
    struct enable_max_value
    {
        underlying_default_value<float, decltype(default_values), default_values> max_value_;

        Concrete max_value( float new_max_value ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.max_value_.val_ = new_max_value;
            return ans;
        }

        float max_value() const noexcept
        {
            return max_value_.val_;
        }
    };

    ///
    /// @brief Adding momentum to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_momentum<X, momentum_val> {};
    /// auto x = X().momentum( new_momentum_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::float32 default_values >
    struct enable_momentum
    {
        underlying_default_value<float, decltype(default_values), default_values> momentum_;

        Concrete momentum( float new_momentum ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.momentum_.val_ = new_momentum;
            return ans;
        }

        float momentum() const noexcept
        {
            return momentum_.val_;
        }
    };

    ///
    /// @brief Adding negative_slope to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_negative_slope<X, negative_slope_val> {};
    /// auto x = X().negative_slope( new_negative_slope_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::float32 default_values >
    struct enable_negative_slope
    {
        underlying_default_value<float, decltype(default_values), default_values> negative_slope_;

        Concrete negative_slope( float new_negative_slope ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.negative_slope_.val_ = new_negative_slope;
            return ans;
        }

        float negative_slope() const noexcept
        {
            return negative_slope_.val_;
        }
    };

    ///
    /// @brief Adding rate to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_rate<X, rate_val> {};
    /// auto x = X().rate( new_rate_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::float32 default_values >
    struct enable_rate
    {
        underlying_default_value<float, decltype(default_values), default_values> rate_;

        Concrete rate( float new_rate ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.rate_.val_ = new_rate;
            return ans;
        }

        float rate() const noexcept
        {
            return rate_.val_;
        }
    };

    ///
    /// @brief Adding theta to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_theta<X, theta_val> {};
    /// auto x = X().theta( new_theta_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::float32 default_values >
    struct enable_theta
    {
        underlying_default_value<float, decltype(default_values), default_values> theta_;

        Concrete theta( float new_theta ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.theta_.val_ = new_theta;
            return ans;
        }

        float theta() const noexcept
        {
            return theta_.val_;
        }
    };

    ///
    /// @brief Adding threshold to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_threshold<X, threshold_val> {};
    /// auto x = X().threshold( new_threshold_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::float32 default_values >
    struct enable_threshold
    {
        underlying_default_value<float, decltype(default_values), default_values> threshold_;

        Concrete threshold( float new_threshold ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.threshold_.val_ = new_threshold;
            return ans;
        }

        float threshold() const noexcept
        {
            return threshold_.val_;
        }
    };

    ///
    /// @brief Adding activation to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_activation<X, activation_val> {};
    /// auto x = X().activation( new_activation_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_activation
    {
        underlying_default_value<std::string, decltype(default_values), default_values> activation_;

        Concrete activation( std::string new_activation ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.activation_.val_ = new_activation;
            return ans;
        }

        std::string activation() const noexcept
        {
            return activation_.val_;
        }
    };

    ///
    /// @brief Adding activity_regularizer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_activity_regularizer<X, activity_regularizer_val> {};
    /// auto x = X().activity_regularizer( new_activity_regularizer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_activity_regularizer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> activity_regularizer_;

        Concrete activity_regularizer( std::string new_activity_regularizer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.activity_regularizer_.val_ = new_activity_regularizer;
            return ans;
        }

        std::string activity_regularizer() const noexcept
        {
            return activity_regularizer_.val_;
        }
    };

    ///
    /// @brief Adding alpha_constraint to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_alpha_constraint<X, alpha_constraint_val> {};
    /// auto x = X().alpha_constraint( new_alpha_constraint_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_alpha_constraint
    {
        underlying_default_value<std::string, decltype(default_values), default_values> alpha_constraint_;

        Concrete alpha_constraint( std::string new_alpha_constraint ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.alpha_constraint_.val_ = new_alpha_constraint;
            return ans;
        }

        std::string alpha_constraint() const noexcept
        {
            return alpha_constraint_.val_;
        }
    };

    ///
    /// @brief Adding alpha_initializer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_alpha_initializer<X, alpha_initializer_val> {};
    /// auto x = X().alpha_initializer( new_alpha_initializer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_alpha_initializer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> alpha_initializer_;

        Concrete alpha_initializer( std::string new_alpha_initializer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.alpha_initializer_.val_ = new_alpha_initializer;
            return ans;
        }

        std::string alpha_initializer() const noexcept
        {
            return alpha_initializer_.val_;
        }
    };

    ///
    /// @brief Adding alpha_regularizer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_alpha_regularizer<X, alpha_regularizer_val> {};
    /// auto x = X().alpha_regularizer( new_alpha_regularizer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_alpha_regularizer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> alpha_regularizer_;

        Concrete alpha_regularizer( std::string new_alpha_regularizer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.alpha_regularizer_.val_ = new_alpha_regularizer;
            return ans;
        }

        std::string alpha_regularizer() const noexcept
        {
            return alpha_regularizer_.val_;
        }
    };

    ///
    /// @brief Adding beta_constraint to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_beta_constraint<X, beta_constraint_val> {};
    /// auto x = X().beta_constraint( new_beta_constraint_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_beta_constraint
    {
        underlying_default_value<std::string, decltype(default_values), default_values> beta_constraint_;

        Concrete beta_constraint( std::string new_beta_constraint ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.beta_constraint_.val_ = new_beta_constraint;
            return ans;
        }

        std::string beta_constraint() const noexcept
        {
            return beta_constraint_.val_;
        }
    };

    ///
    /// @brief Adding beta_initializer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_beta_initializer<X, beta_initializer_val> {};
    /// auto x = X().beta_initializer( new_beta_initializer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_beta_initializer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> beta_initializer_;

        Concrete beta_initializer( std::string new_beta_initializer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.beta_initializer_.val_ = new_beta_initializer;
            return ans;
        }

        std::string beta_initializer() const noexcept
        {
            return beta_initializer_.val_;
        }
    };

    ///
    /// @brief Adding beta_regularizer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_beta_regularizer<X, beta_regularizer_val> {};
    /// auto x = X().beta_regularizer( new_beta_regularizer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_beta_regularizer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> beta_regularizer_;

        Concrete beta_regularizer( std::string new_beta_regularizer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.beta_regularizer_.val_ = new_beta_regularizer;
            return ans;
        }

        std::string beta_regularizer() const noexcept
        {
            return beta_regularizer_.val_;
        }
    };

    ///
    /// @brief Adding bias_constraint to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_bias_constraint<X, bias_constraint_val> {};
    /// auto x = X().bias_constraint( new_bias_constraint_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_bias_constraint
    {
        underlying_default_value<std::string, decltype(default_values), default_values> bias_constraint_;

        Concrete bias_constraint( std::string new_bias_constraint ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.bias_constraint_.val_ = new_bias_constraint;
            return ans;
        }

        std::string bias_constraint() const noexcept
        {
            return bias_constraint_.val_;
        }
    };

    ///
    /// @brief Adding bias_initializer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_bias_initializer<X, bias_initializer_val> {};
    /// auto x = X().bias_initializer( new_bias_initializer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_bias_initializer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> bias_initializer_;

        Concrete bias_initializer( std::string new_bias_initializer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.bias_initializer_.val_ = new_bias_initializer;
            return ans;
        }

        std::string bias_initializer() const noexcept
        {
            return bias_initializer_.val_;
        }
    };

    ///
    /// @brief Adding bias_regularizer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_bias_regularizer<X, bias_regularizer_val> {};
    /// auto x = X().bias_regularizer( new_bias_regularizer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_bias_regularizer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> bias_regularizer_;

        Concrete bias_regularizer( std::string new_bias_regularizer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.bias_regularizer_.val_ = new_bias_regularizer;
            return ans;
        }

        std::string bias_regularizer() const noexcept
        {
            return bias_regularizer_.val_;
        }
    };

    ///
    /// @brief Adding embedding_initializer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_embedding_initializer<X, embedding_initializer_val> {};
    /// auto x = X().embedding_initializer( new_embedding_initializer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_embedding_initializer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> embedding_initializer_;

        Concrete embedding_initializer( std::string new_embedding_initializer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.embedding_initializer_.val_ = new_embedding_initializer;
            return ans;
        }

        std::string embedding_initializer() const noexcept
        {
            return embedding_initializer_.val_;
        }
    };

    ///
    /// @brief Adding embedding_regularizer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_embedding_regularizer<X, embedding_regularizer_val> {};
    /// auto x = X().embedding_regularizer( new_embedding_regularizer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_embedding_regularizer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> embedding_regularizer_;

        Concrete embedding_regularizer( std::string new_embedding_regularizer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.embedding_regularizer_.val_ = new_embedding_regularizer;
            return ans;
        }

        std::string embedding_regularizer() const noexcept
        {
            return embedding_regularizer_.val_;
        }
    };

    ///
    /// @brief Adding embeding_constraint to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_embeding_constraint<X, embeding_constraint_val> {};
    /// auto x = X().embeding_constraint( new_embeding_constraint_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_embeding_constraint
    {
        underlying_default_value<std::string, decltype(default_values), default_values> embeding_constraint_;

        Concrete embeding_constraint( std::string new_embeding_constraint ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.embeding_constraint_.val_ = new_embeding_constraint;
            return ans;
        }

        std::string embeding_constraint() const noexcept
        {
            return embeding_constraint_.val_;
        }
    };

    ///
    /// @brief Adding gamma_constraint to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_gamma_constraint<X, gamma_constraint_val> {};
    /// auto x = X().gamma_constraint( new_gamma_constraint_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_gamma_constraint
    {
        underlying_default_value<std::string, decltype(default_values), default_values> gamma_constraint_;

        Concrete gamma_constraint( std::string new_gamma_constraint ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.gamma_constraint_.val_ = new_gamma_constraint;
            return ans;
        }

        std::string gamma_constraint() const noexcept
        {
            return gamma_constraint_.val_;
        }
    };

    ///
    /// @brief Adding gamma_initializer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_gamma_initializer<X, gamma_initializer_val> {};
    /// auto x = X().gamma_initializer( new_gamma_initializer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_gamma_initializer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> gamma_initializer_;

        Concrete gamma_initializer( std::string new_gamma_initializer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.gamma_initializer_.val_ = new_gamma_initializer;
            return ans;
        }

        std::string gamma_initializer() const noexcept
        {
            return gamma_initializer_.val_;
        }
    };

    ///
    /// @brief Adding gamma_regularizer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_gamma_regularizer<X, gamma_regularizer_val> {};
    /// auto x = X().gamma_regularizer( new_gamma_regularizer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_gamma_regularizer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> gamma_regularizer_;

        Concrete gamma_regularizer( std::string new_gamma_regularizer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.gamma_regularizer_.val_ = new_gamma_regularizer;
            return ans;
        }

        std::string gamma_regularizer() const noexcept
        {
            return gamma_regularizer_.val_;
        }
    };

    ///
    /// @brief Adding interpolation to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_interpolation<X, interpolation_val> {};
    /// auto x = X().interpolation( new_interpolation_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_interpolation
    {
        underlying_default_value<std::string, decltype(default_values), default_values> interpolation_;

        Concrete interpolation( std::string new_interpolation ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.interpolation_.val_ = new_interpolation;
            return ans;
        }

        std::string interpolation() const noexcept
        {
            return interpolation_.val_;
        }
    };

    ///
    /// @brief Adding kernel_constraint to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_kernel_constraint<X, kernel_constraint_val> {};
    /// auto x = X().kernel_constraint( new_kernel_constraint_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_kernel_constraint
    {
        underlying_default_value<std::string, decltype(default_values), default_values> kernel_constraint_;

        Concrete kernel_constraint( std::string new_kernel_constraint ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.kernel_constraint_.val_ = new_kernel_constraint;
            return ans;
        }

        std::string kernel_constraint() const noexcept
        {
            return kernel_constraint_.val_;
        }
    };

    ///
    /// @brief Adding kernel_initializer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_kernel_initializer<X, kernel_initializer_val> {};
    /// auto x = X().kernel_initializer( new_kernel_initializer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_kernel_initializer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> kernel_initializer_;

        Concrete kernel_initializer( std::string new_kernel_initializer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.kernel_initializer_.val_ = new_kernel_initializer;
            return ans;
        }

        std::string kernel_initializer() const noexcept
        {
            return kernel_initializer_.val_;
        }
    };

    ///
    /// @brief Adding kernel_regularizer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_kernel_regularizer<X, kernel_regularizer_val> {};
    /// auto x = X().kernel_regularizer( new_kernel_regularizer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_kernel_regularizer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> kernel_regularizer_;

        Concrete kernel_regularizer( std::string new_kernel_regularizer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.kernel_regularizer_.val_ = new_kernel_regularizer;
            return ans;
        }

        std::string kernel_regularizer() const noexcept
        {
            return kernel_regularizer_.val_;
        }
    };

    ///
    /// @brief Adding moving_mean_initializer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_moving_mean_initializer<X, moving_mean_initializer_val> {};
    /// auto x = X().moving_mean_initializer( new_moving_mean_initializer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_moving_mean_initializer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> moving_mean_initializer_;

        Concrete moving_mean_initializer( std::string new_moving_mean_initializer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.moving_mean_initializer_.val_ = new_moving_mean_initializer;
            return ans;
        }

        std::string moving_mean_initializer() const noexcept
        {
            return moving_mean_initializer_.val_;
        }
    };

    ///
    /// @brief Adding moving_variance_initializer to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_moving_variance_initializer<X, moving_variance_initializer_val> {};
    /// auto x = X().moving_variance_initializer( new_moving_variance_initializer_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_moving_variance_initializer
    {
        underlying_default_value<std::string, decltype(default_values), default_values> moving_variance_initializer_;

        Concrete moving_variance_initializer( std::string new_moving_variance_initializer ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.moving_variance_initializer_.val_ = new_moving_variance_initializer;
            return ans;
        }

        std::string moving_variance_initializer() const noexcept
        {
            return moving_variance_initializer_.val_;
        }
    };

    ///
    /// @brief Adding name to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_name<X, name_val> {};
    /// auto x = X().name( new_name_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_name
    {
        underlying_default_value<std::string, decltype(default_values), default_values> name_;

        Concrete name( std::string new_name ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.name_.val_ = new_name;
            return ans;
        }

        std::string name() const noexcept
        {
            return name_.val_;
        }
    };

    ///
    /// @brief Adding padding to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_padding<X, padding_val> {};
    /// auto x = X().padding( new_padding_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, ceras::string default_values >
    struct enable_padding
    {
        underlying_default_value<std::string, decltype(default_values), default_values> padding_;

        Concrete padding( std::string new_padding ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.padding_.val_ = new_padding;
            return ans;
        }

        std::string padding() const noexcept
        {
            return padding_.val_;
        }
    };

    ///
    /// @brief Adding axis to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_axis<X, axis_val> {};
    /// auto x = X().axis( new_axis_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long default_values >
    struct enable_axis
    {
        underlying_default_value<unsigned long, unsigned long, default_values> axis_;

        Concrete axis( unsigned long new_axis ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.axis_.val_ = new_axis;
            return ans;
        }

        unsigned long axis() const noexcept
        {
            return axis_.val_;
        }
    };

    ///
    /// @brief Adding batch_size to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_batch_size<X, batch_size_val> {};
    /// auto x = X().batch_size( new_batch_size_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long default_values >
    struct enable_batch_size
    {
        underlying_default_value<unsigned long, unsigned long, default_values> batch_size_;

        Concrete batch_size( unsigned long new_batch_size ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.batch_size_.val_ = new_batch_size;
            return ans;
        }

        unsigned long batch_size() const noexcept
        {
            return batch_size_.val_;
        }
    };

    ///
    /// @brief Adding filters to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_filters<X, filters_val> {};
    /// auto x = X().filters( new_filters_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long default_values >
    struct enable_filters
    {
        underlying_default_value<unsigned long, unsigned long, default_values> filters_;

        Concrete filters( unsigned long new_filters ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.filters_.val_ = new_filters;
            return ans;
        }

        unsigned long filters() const noexcept
        {
            return filters_.val_;
        }
    };

    ///
    /// @brief Adding groups to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_groups<X, groups_val> {};
    /// auto x = X().groups( new_groups_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long default_values >
    struct enable_groups
    {
        underlying_default_value<unsigned long, unsigned long, default_values> groups_;

        Concrete groups( unsigned long new_groups ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.groups_.val_ = new_groups;
            return ans;
        }

        unsigned long groups() const noexcept
        {
            return groups_.val_;
        }
    };

    ///
    /// @brief Adding input_dim to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_input_dim<X, input_dim_val> {};
    /// auto x = X().input_dim( new_input_dim_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long default_values >
    struct enable_input_dim
    {
        underlying_default_value<unsigned long, unsigned long, default_values> input_dim_;

        Concrete input_dim( unsigned long new_input_dim ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.input_dim_.val_ = new_input_dim;
            return ans;
        }

        unsigned long input_dim() const noexcept
        {
            return input_dim_.val_;
        }
    };

    ///
    /// @brief Adding input_length to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_input_length<X, input_length_val> {};
    /// auto x = X().input_length( new_input_length_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long default_values >
    struct enable_input_length
    {
        underlying_default_value<unsigned long, unsigned long, default_values> input_length_;

        Concrete input_length( unsigned long new_input_length ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.input_length_.val_ = new_input_length;
            return ans;
        }

        unsigned long input_length() const noexcept
        {
            return input_length_.val_;
        }
    };

    ///
    /// @brief Adding output_dim to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_output_dim<X, output_dim_val> {};
    /// auto x = X().output_dim( new_output_dim_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long default_values >
    struct enable_output_dim
    {
        underlying_default_value<unsigned long, unsigned long, default_values> output_dim_;

        Concrete output_dim( unsigned long new_output_dim ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.output_dim_.val_ = new_output_dim;
            return ans;
        }

        unsigned long output_dim() const noexcept
        {
            return output_dim_.val_;
        }
    };

    ///
    /// @brief Adding seed to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_seed<X, seed_val> {};
    /// auto x = X().seed( new_seed_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long default_values >
    struct enable_seed
    {
        underlying_default_value<unsigned long, unsigned long, default_values> seed_;

        Concrete seed( unsigned long new_seed ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.seed_.val_ = new_seed;
            return ans;
        }

        unsigned long seed() const noexcept
        {
            return seed_.val_;
        }
    };

    ///
    /// @brief Adding units to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_units<X, units_val> {};
    /// auto x = X().units( new_units_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long default_values >
    struct enable_units
    {
        underlying_default_value<unsigned long, unsigned long, default_values> units_;

        Concrete units( unsigned long new_units ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.units_.val_ = new_units;
            return ans;
        }

        unsigned long units() const noexcept
        {
            return units_.val_;
        }
    };


    ///
    /// @brief Adding kernel_regularizer_l1 to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_kernel_regularizer_l1<X, "0.0"> {};
    /// auto x = X().kernel_regularizer_l1( 1.0e-3f ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, float32 default_values >
    struct enable_kernel_regularizer_l1
    {
        underlying_default_value<float, decltype(default_values), default_values> kernel_regularizer_l1_;

        Concrete kernel_regularizer_l1( float new_kernel_regularizer_l1 ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.kernel_regularizer_l1_.val_ = new_kernel_regularizer_l1;
            return ans;
        }

        float kernel_regularizer_l1() const noexcept
        {
            return kernel_regularizer_l1_.val_;
        }
    };

    ///
    /// @brief Adding kernel_regularizer_l2 to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_kernel_regularizer_l2<X, "1.1"> {};
    /// auto x = X().kernel_regularizer_l2( 0.1 ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, float32 default_values >
    struct enable_kernel_regularizer_l2
    {
        underlying_default_value<float, decltype(default_values), default_values> kernel_regularizer_l2_;

        Concrete kernel_regularizer_l2( float new_kernel_regularizer_l2 ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.kernel_regularizer_l2_.val_ = new_kernel_regularizer_l2;
            return ans;
        }

        float kernel_regularizer_l2() const noexcept
        {
            return kernel_regularizer_l2_.val_;
        }
    };


    ///
    /// @brief Adding bias_regularizer_l1 to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_bias_regularizer_l1<X, "0.0"> {};
    /// auto x = X().bias_regularizer_l1( 1.0e-3f ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, float32 default_values >
    struct enable_bias_regularizer_l1
    {
        underlying_default_value<float, decltype(default_values), default_values> bias_regularizer_l1_;

        Concrete bias_regularizer_l1( float new_bias_regularizer_l1 ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.bias_regularizer_l1_.val_ = new_bias_regularizer_l1;
            return ans;
        }

        float bias_regularizer_l1() const noexcept
        {
            return bias_regularizer_l1_.val_;
        }
    };

    ///
    /// @brief Adding bias_regularizer_l2 to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_bias_regularizer_l2<X, "1.1"> {};
    /// auto x = X().bias_regularizer_l2( 0.1 ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, float32 default_values >
    struct enable_bias_regularizer_l2
    {
        underlying_default_value<float, decltype(default_values), default_values> bias_regularizer_l2_;

        Concrete bias_regularizer_l2( float new_bias_regularizer_l2 ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.bias_regularizer_l2_.val_ = new_bias_regularizer_l2;
            return ans;
        }

        float bias_regularizer_l2() const noexcept
        {
            return bias_regularizer_l2_.val_;
        }
    };

    ///
    /// @brief Adding trainable to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_trainable<X, trainable_val> {};
    /// auto x = X().trainable( new_trainable_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, bool default_values >
    struct enable_trainable
    {
        underlying_default_value<bool, bool, default_values> trainable_;

        Concrete trainable( bool new_trainable ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.trainable_.val_ = new_trainable;
            return ans;
        }

        bool trainable() const noexcept
        {
            return trainable_.val_;
        }
    };

    ///
    /// @brief Adding uses_learning_phase to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_uses_learning_phase<X, uses_learning_phase_val> {};
    /// auto x = X().uses_learning_phase( new_uses_learning_phase_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, bool default_values >
    struct enable_uses_learning_phase
    {
        underlying_default_value<bool, bool, default_values> uses_learning_phase_;

        Concrete uses_learning_phase( bool new_uses_learning_phase ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.uses_learning_phase_.val_ = new_uses_learning_phase;
            return ans;
        }

        bool uses_learning_phase() const noexcept
        {
            return uses_learning_phase_.val_;
        }
    };

    ///
    /// @brief Adding pool_size to the Concrete class.
    ///
    /// \code{.cpp}
    /// struct X : enable_pool_size<X, pool_size_vals...> {};
    /// auto x = X().pool_size( new_pool_size_val ). ... . (... );
    /// \endcode
    ///
    template< typename Concrete, unsigned long... default_values >
    struct enable_pool_size
    {
        underlying_default_value<std::vector<unsigned long>, unsigned long, default_values...> pool_size_;

        Concrete pool_size( std::vector<unsigned long> new_pool_size ) const noexcept
        {
            Concrete ans{ static_cast<Concrete const&>(*this) };
            ans.pool_size_.val_ = new_pool_size;
            return ans;
        }

        std::vector<unsigned long> pool_size() const noexcept
        {
            return pool_size_.val_;
        }
    };





}//namespace ceras

#endif//VIJTSJOWXEEBOVDLNBJPNYPFSYJSCXBQKYYFYWQBAJDQAJNENJIHPXJFXJTNKRGWDTXWVTXJM

