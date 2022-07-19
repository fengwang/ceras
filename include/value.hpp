#ifndef VALUE_HPP_INCLUDED_DS9P8IU4LKJASDOIPUY498YAFKASHFAS9F8Y4OKHDAFSIUOHASDFFS
#define VALUE_HPP_INCLUDED_DS9P8IU4LKJASDOIPUY498YAFKASHFAS9F8Y4OKHDAFSIUOHASDFFS

#include "./includes.hpp"
#include "./tensor.hpp"
#include "./utils/id.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/enable_shared.hpp"
#include "./utils/type2string.hpp"
#include "./utils/fmt.hpp"

namespace ceras
{

    ///
    /// @brief Create a constant scalar.
    ///
    /// \code{.cpp}
    /// value<double> one{ 1.0 };
    /// \endcode
    ///
    template< typename T > requires std::floating_point<T>
    struct value : enable_id< value<T>, "Value" >
    {
        typedef T value_type;
        typedef tensor<value_type> tensor_type;
        value_type data_;

        value() = delete;
        value( value_type v ) noexcept : enable_id<value<T>, "Value">{}, data_{ v } {}
        value( value const& ) noexcept = default;
        value( value && ) noexcept = default;
        value& operator =( value const& ) noexcept = default;
        value& operator =( value && ) noexcept = default;

        void backward( auto ) noexcept { }

        template< Tensor Tsor >
        Tsor const forward( Tsor const& refer ) const
        {
            Tsor ans = ones_like( refer ); // cast it to a tensor
            ans *= data_;
            return ans;
        }

        std::vector<unsigned long> shape() const noexcept
        {
            return std::vector<unsigned long>{ {-1UL,} };
        }

        value_type data() const noexcept
        {
            return data_;
        }

        value_type& data() noexcept
        {
            return data_;
        }

    };//struct value

    template< typename T >
    using scalar = value<T>;

    template< typename T >
    struct is_value : std::false_type {};

    template< typename T >
    struct is_value< value< T > > : std::true_type {};

    template< class T >
    inline constexpr bool is_value_v = is_value<T>::value;

    template< typename T >
    concept Value = is_value_v<T>;


    // for tensor_type deduction in a binary operator
    template< typename L, typename R >
    struct tensor_deduction
    {
        using op_type = std::conditional<is_value_v<L>, R, L>::type;
        using tensor_type = std::remove_cv_t<decltype(std::declval<op_type>().forward())>;
    };

    ///
    /// @brief Dump a value to cpp code.
    ///
    template< Value Val >
    std::tuple<std::string, std::vector<std::string>> const serialize( Val const& v ) noexcept
    {
        std::string value_name = fmt::format( "value_{}", v.id() );
        std::vector<std::string> value_code;
        value_code.emplace_back( fmt::format( "ceras::value<{}> {}( {} );", type2string<typename Val::value_type>(), value_name, v.data() ) );
        return std::forward_as_tuple( value_name, value_code );
    }

}//namespace ceras

#endif//VALUE_HPP_INCLUDED_DS9P8IU4LKJASDOIPUY498YAFKASHFAS9F8Y4OKHDAFSIUOHASDFFS

