#ifndef CAFUTPOUSGTCEEJFWOUMQTWNSGRSWBVLLOSRSEDXYGXKHDEILDOGELEQNQBCVJRTHNETVFBND
#define CAFUTPOUSGTCEEJFWOUMQTWNSGRSWBVLLOSRSEDXYGXKHDEILDOGELEQNQBCVJRTHNETVFBND
namespace ceras
{

    ///
    /// @brief The iterator to the first element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto begin( Tsor const& tsor ) noexcept
    {
        return tsor.begin();
    }
    ///
    /// @brief The iterator to the first element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto cbegin( Tsor const& tsor ) noexcept
    {
        return tsor.begin();
    }

    ///
    /// @brief The iterator to the first element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto begin( Tsor& tsor ) noexcept
    {
        return tsor.begin();
    }

    ///
    /// @brief The iterator to the element following the last element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto end( Tsor const& tsor ) noexcept
    {
        return tsor.end();
    }
    ///
    /// @brief The iterator to the element following the last element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto cend( Tsor const& tsor ) noexcept
    {
        return tsor.end();
    }

    ///
    /// @brief The iterator to the element following the last element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto end( Tsor& tsor ) noexcept
    {
        return tsor.end();
    }



    ///
    /// @brief The reverse iterator to the first element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto rbegin( Tsor const& tsor ) noexcept
    {
        return tsor.rbegin();
    }
    ///
    /// @brief The reverse iterator to the first element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto crbegin( Tsor const& tsor ) noexcept
    {
        return tsor.crbegin();
    }

    ///
    /// @brief The reverse iterator to the first element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto rbegin( Tsor& tsor ) noexcept
    {
        return tsor.rbegin();
    }

    ///
    /// @brief The reverse iterator to the element following the last element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto rend( Tsor const& tsor ) noexcept
    {
        return tsor.rend();
    }
    ///
    /// @brief The reverse iterator to the element following the last element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto crend( Tsor const& tsor ) noexcept
    {
        return tsor.crend();
    }

    ///
    /// @brief The reverse iterator to the element following the last element of the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto rend( Tsor& tsor ) noexcept
    {
        return tsor.rend();
    }

    ///
    /// @brief The reference to the first element in the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto front( Tsor& tsor ) noexcept
    {
        return tsor.front();
    }

    ///
    /// @brief The reference to the first element in the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto front( Tsor const& tsor ) noexcept
    {
        return tsor.front();
    }

    ///
    /// @brief The reference to the last element in the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto back( Tsor& tsor ) noexcept
    {
        return tsor.back();
    }

    ///
    /// @brief The reference to the last element in the tensor.
    ///
    template< Tensor Tsor >
    constexpr auto back( Tsor const& tsor ) noexcept
    {
        return tsor.back();
    }



    ///
    /// @brief Checks if the container has elements.
    ///
    template< Tensor Tsor >
    [[nodiscard]] constexpr bool empty( Tsor const& tsor ) noexcept
    {
        return tsor.empty();
    }


    ///
    /// @brief Number of elements in the tensor.
    ///
    template< Tensor Tsor >
    constexpr unsigned long size( Tsor const& tsor ) noexcept
    {
        return tsor.size();
    }


    ///
    /// @brief Dimension of the tensor.
    ///
    template< Tensor Tsor >
    constexpr unsigned long ndim( Tsor const& tsor ) noexcept
    {
        return tsor.ndim();
    }


    ///
    /// @brief Reset all emements in the tensor
    ///
    template< Tensor Tsor >
    constexpr unsigned long reset( Tsor& tsor, typename Tsor::value_type val=0 ) noexcept
    {
        return tsor.reset( val );
    }


    ///
    /// @brief Shape of the tensor
    ///
    template< Tensor Tsor >
    constexpr auto shape( Tsor const& tsor ) noexcept
    {
        return tsor.shape();
    }

    ///
    /// @brief A deep copy of the tensor
    ///
    template< Tensor Tsor >
    constexpr auto deep_copy( Tsor const& tsor ) noexcept
    {
        return tsor.deep_copy();
    }


    ///
    /// @brief Resize the tensor to a new shape. Size of the tensor might change.
    ///
    template< Tensor Tsor >
    constexpr auto resize( Tsor& tsor, std::vector<unsigned long> const& new_shape ) noexcept
    {
        return tsor.resize( new_shape );
    }


    ///
    /// @brief Resize the tensor to a new shape. Size of the tensor remains the same as before.
    ///
    template< Tensor Tsor >
    constexpr auto reshape( Tsor& tsor, std::vector<unsigned long> const& new_shape ) noexcept
    {
        return tsor.reshape( new_shape );
    }

    ///
    /// @brief Returns pointer to the underlying array serving as element storage.
    ///
    template< Tensor Tsor >
    constexpr auto data( Tsor const& tsor ) noexcept
    {
        return tsor.data();
    }

    ///
    /// @brief Returns pointer to the underlying array serving as element storage.
    ///
    template< Tensor Tsor >
    constexpr auto data( Tsor& tsor ) noexcept
    {
        return tsor.data();
    }

    ///
    /// @brief Applying element-wise operation on tensor.
    ///
    template< Tensor Tsor, typename Function >
    constexpr auto map( Tsor& tsor, Function f )
    {
        tsor.map( f );
    }

    ///
    /// @brief Cast to a new underlying type.
    ///
    template< Tensor Tsor, typename T >
    constexpr auto as_type( Tsor const& tsor ) noexcept
    {
        return tsor.template as_type<T>();
    }



}//namespace ceras

#endif//CAFUTPOUSGTCEEJFWOUMQTWNSGRSWBVLLOSRSEDXYGXKHDEILDOGELEQNQBCVJRTHNETVFBND

