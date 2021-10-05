#ifndef VIEW_HPP_INCLUDED_4903090RETOPIEPOIWEPRTIOEWR09UD0G9USDGFIOSODGFISOIGFG8
#define VIEW_HPP_INCLUDED_4903090RETOPIEPOIWEPRTIOEWR09UD0G9USDGFIOSODGFISOIGFG8

#include "../includes.hpp"
#include "./better_assert.hpp"
#include "./stride_iterator.hpp"

namespace ceras
{

    template< typename T >
    struct view_1d
    {
        T* data;
        unsigned long dims;

        constexpr T& operator[]( unsigned long idx ) noexcept { return data[idx]; }
        constexpr T const& operator[]( unsigned long idx ) const noexcept { return data[idx]; }
    };// view_1d

    template< typename T >
    using array_view = view_1d<T>;

    template< typename T, typename A >
    struct tensor;

    template< typename T >
    struct view_2d
    {
        typedef T value_type;
        typedef value_type* row_type;
        typedef const value_type* const_row_type;
        typedef stride_iterator<value_type*> col_type;
        typedef stride_iterator<const value_type*> const_col_type;

        T* data_;
        unsigned long row_;
        unsigned long col_;
        bool transposed_;

        template<typename A>
        constexpr view_2d( tensor<T, A>& tsor, unsigned long row, unsigned long col, bool transposed=false ) noexcept : data_{tsor.data()}, row_{row}, col_{col}, transposed_{transposed} {}

        constexpr view_2d( T* data, unsigned long row, unsigned long col, bool transposed=false ) noexcept : data_{data}, row_{row}, col_{col}, transposed_{transposed} {}

        // should have a template specialization for view_2d of const T*, but here ommited with 'const_cast' as operator []  should be specialized in that case
        constexpr view_2d( const T* data, unsigned long row, unsigned long col, bool transposed=false ) noexcept : data_{const_cast<T*>(data)}, row_{row}, col_{col}, transposed_{transposed} {}

        constexpr T* operator[]( unsigned long index )
        {
            if ( transposed_ )
                return data_ + index * row_;
            return data_ + index * col_;
        }

        constexpr const T* operator[]( unsigned long index ) const
        {
            if ( transposed_ )
                return data_ + index * row_;
            return data_ + index * col_;
        }

        constexpr auto shape() const noexcept { return std::make_pair( row_, col_ ); }
        constexpr unsigned long size() const noexcept { return row_ * col_; }

        constexpr T* data() noexcept { return data_; }
        constexpr const T* data() const noexcept { return data_; }

        constexpr T* begin() noexcept { return data_; }
        constexpr const T* end() const noexcept { return begin()+size(); }

        constexpr unsigned long row() const noexcept { return row_; }
        constexpr unsigned long col() const noexcept { return col_; }

        constexpr row_type row_begin( unsigned long index = 0 ) noexcept { return begin() + index * col(); }
        constexpr row_type row_end( unsigned long index = 0 ) noexcept { return begin() + (index+1) * col(); }

        constexpr const_row_type row_begin( unsigned long index = 0 ) const noexcept { return begin() + index * col(); }
        constexpr const_row_type row_end( unsigned long index = 0 ) const noexcept { return begin() + (index+1) * col(); }

        constexpr col_type col_begin( unsigned long index = 0 ) noexcept { return col_type{ begin() + index, col() }; }
        constexpr col_type col_end( unsigned long index = 0 ) noexcept { return col_begin(index) + row(); }

        constexpr const_col_type col_begin( unsigned long index = 0 ) const noexcept { return const_col_type{ begin() + index, col() }; }
        constexpr const_col_type col_end( unsigned long index = 0 ) const noexcept { return col_begin(index) + row(); }
    };

    template< typename T >
    using matrix_view = view_2d<T>;

    template< typename T >
    struct view_3d
    {
        T* data_;
        unsigned long row_;
        unsigned long col_;
        unsigned long channel_;

        constexpr view_3d( T* data, unsigned long row, unsigned long col, unsigned long channel ) noexcept : data_{data}, row_{row}, col_{col}, channel_{channel} {}

        constexpr auto operator[]( unsigned long index ) noexcept
        {
            return view_2d{ data_+index*col_*channel_, col_, channel_ };
        }

        constexpr auto operator[]( unsigned long index ) const noexcept
        {
            return view_2d{ data_+index*col_*channel_, col_, channel_ };
        }
    };

    template< typename T >
    using cube_view = view_3d<T>;

    ///
    /// A class viewing a 1-D array as a 4-D tensor. This class is useful when treating an array as a typical 4-D tensor in a neural network, with a shape of [batch_size, row, column, channel].
    ///
    template< typename T >
    struct view_4d
    {
        T* data_; ///< The pointer to the start position of the 1-D array.
        unsigned long batch_size_; ///< The batch size of the 4-D tensor, also the first dimension of the tensor.
        unsigned long row_; ///< The row of the 4-D tensor, also the second dimension of the tensor.
        unsigned long col_; ///< The column of the 4-D tensor, also the third dimension of the tensor.
        unsigned long channel_; ///< The channel of the 4-D tensor, also the last dimension of the tensor.

        ///
        /// Constructor of view_4d
        /// @param data The raw pointer to the start position of the 1-D array.
        /// @param batch_size The first dimension of the 4-D tensor, also for the batch size in the CNN layers.
        /// @param row The second dimension of the 4-D tensor, also for the row in the CNN layers.
        /// @param col The third dimension of the 4-D tensor, also for the column in the CNN layers.
        /// @param channel The last dimension of the 4-D tensor, also for the channel in the CNN layers.
        ///
        constexpr view_4d( T* data=nullptr, unsigned long batch_size=0, unsigned long row=0, unsigned long col=0, unsigned long channel=0 ) noexcept : data_{data}, batch_size_{batch_size}, row_{row}, col_{col}, channel_{channel} {}

        ///
        /// Giving a view_3d interface for operator [].
        /// @param index The first dimension of the 4-D tensor.
        ///
        /// Example usage:
        ///
        /// @code
        ///     std::vector<float> array;
        ///     array.resize( 16*8*8*3 );
        ///     auto t = view_4d{ array.data(), 16, 8, 8, 3 };
        ///     t[0][1][2][3] = 1.0;
        /// @endcode
        ///
        constexpr auto operator[]( unsigned long index ) noexcept
        {
            return view_3d{ data_+index*row_*col_*channel_, row_, col_, channel_ };
        }


        ///
        /// Giving a view_3d interface for operator [].
        /// @param index The first dimension of the 4-D tensor.
        ///
        /// Example usage:
        ///
        /// @code
        ///     std::vector<float> array;
        ///     array.resize( 16*8*8*3 );
        ///     // operations on `array`
        ///     auto t = view_4d{ array.data(), 16, 8, 8, 3 };
        ///     float v0123 = t[0][1][2][3];
        /// @endcode
        ///
        constexpr auto operator[]( unsigned long index ) const noexcept
        {
            return view_3d{ data_+index*row_*col_*channel_, row_, col_, channel_ };
        }
    }; // struct view_4d

    template<typename T >
    using tesseract_view = view_4d<T>;


    template<typename T, unsigned long N>
    struct view;

    template<typename T>
    struct view<T, 1> : view_1d<T>
    {
        using view_1d<T>::view_1d;
    };

    template<typename T>
    struct view<T, 2> : view_2d<T>
    {
        using view_2d<T>::view_2d;
    };

    template<typename T>
    struct view<T, 3> : view_3d<T>
    {
        using view_3d<T>::view_3d;
    };

    template<typename T>
    struct view<T, 4> : view_4d<T>
    {
        using view_4d<T>::view_4d;

        view( T* data, std::array<unsigned long, 4> const& shape ) noexcept : view_4d<T>{ data, shape[0], shape[1], shape[2], shape[3] } {}
    };

    ///
    /// @brief N-Dimentional view of 1D memory.
    ///
    /// \code{.cpp}
    /// auto t = random<float>( {1, 2, 3, 4, 5, 6, 7} );
    /// auto v = view<float, 7>{ t.data(), {1, 1, 6, 4, 5, 6, 7} }; // view as different shape tensor
    /// std::cout << v[0][0][5][3][4][5][6];
    /// \endcode
    ///
    template< typename T, unsigned long N >
    struct view
    {
        T* data_;
        std::array<unsigned long, N> shape_;

        constexpr view( T* data, std::array<unsigned long, N> const& shape ) noexcept :  data_{ data }, shape_{ shape } {}

        view<T, N-1> operator []( unsigned long index ) noexcept
        {
            unsigned long first_dim = shape_[0];
            better_assert( index < first_dim, "Expecting a dimension smaller than ", first_dim, " but got ", index );
            unsigned long offsets = index * std::accumulate( shape_.begin()+1, shape_.end(), 1UL, [](unsigned long a, unsigned long b){ return a*b; } );

            std::array<unsigned long, N-1> new_shape;
            std::copy( shape_.begin()+1, shape_.end(), new_shape.begin() );
            return view<T, N-1>{ data_+offsets, new_shape };
        }

        view<T, N-1> operator []( unsigned long index ) const noexcept
        {
            unsigned long first_dim = shape_[0];
            better_assert( index < first_dim, "Expecting a dimension smaller than ", first_dim, " but got ", index );
            unsigned long offsets = index * std::accumulate( shape_.begin()+1, shape_.end(), 1UL, [](unsigned long a, unsigned long b){ return a*b; } );

            std::array<unsigned long, N-1> new_shape;
            std::copy( shape_.begin()+1, shape_.end(), new_shape.begin() );
            return view<T, N-1>{ data_+offsets, new_shape };
        }

    }; // struct view


}//namespace ceras

#endif//VIEW_HPP_INCLUDED_4903090RETOPIEPOIWEPRTIOEWR09UD0G9USDGFIOSODGFISOIGFG8

