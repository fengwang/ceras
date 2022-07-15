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

    template< Tensor Tsor, typename CharT, typename Traits >
    std::basic_ostream<CharT, Traits>& operator << ( std::basic_ostream<CharT, Traits>& os_, Tsor const& tsor )
    {
        typedef typename Tsor::value_type value_type;
        std::basic_ostringstream<CharT, Traits> os;
        os.flags(os_.flags());
        os.imbue(os_.getloc());
        os.precision(os_.precision());
        //shape
        os << "shape: [ ";
        std::copy( tsor.shape_.begin(), tsor.shape_.end(), std::ostream_iterator<unsigned long>{os, " "} );
        os << "]\n";

        //data
        os << "data:\n{\n";
        if ( tsor.shape().size() < 2 )
        {
            std::copy( tsor.data(), tsor.data()+tsor.size(), std::ostream_iterator<value_type>{os, "\t"} );
            os << "\n";
        }
        else
        {
            auto const& shape = tsor.shape();
            unsigned long const dims = shape.size();
            unsigned long const last_dim = shape[dims-1];
            unsigned long const abbreviated_rows= std::reduce( shape.begin(), shape.begin()+dims-1, 1UL, []( unsigned long x, unsigned long y ){ return x*y; } );
            for ( auto idx = 0UL; idx != abbreviated_rows; ++idx )
            {
                std::copy( tsor.data()+idx*last_dim, tsor.data()+(idx+1)*last_dim, std::ostream_iterator<value_type>{os, "\t"} );
                os << "\n";
            }
        }
        os << "}\n";

        return os_ << os.str();
    }


    template <Tensor Tsor>
    Tsor broadcast_tensor( Tsor const& tsor, std::vector<unsigned long> const& new_shape ) noexcept
    {
        // case of same shapes
        if ( tsor.shape() == new_shape )
            return tsor;

        auto _ans = tsor;
        std::vector<unsigned long> updated_shape = _ans.shape();
        if (updated_shape.size() < new_shape.size())
        {
            for ([[maybe_unused]]auto _ : range(new_shape.size()-updated_shape.size()))
                updated_shape.insert( updated_shape.begin(), 1 );
            _ans.reshape( updated_shape );
        }

        // case of same shapes after 1-padding
        if ( updated_shape == new_shape )
            return _ans;


        // finding the expanding dimension
        long int dim_to_expand = updated_shape.size()-1;
        while( dim_to_expand >= 0 )
        {
            if ( new_shape[dim_to_expand] != updated_shape[dim_to_expand] )
                break;
            --dim_to_expand;
        }

        better_assert( updated_shape[dim_to_expand] == 1, fmt::format("expecting the expanding dimension to be 1, but got {}", updated_shape[dim_to_expand]) );

        // [headings..][1][tailings...] <- updated_shape
        // [headings..][x][tailings...] <- new_shape
        unsigned long const headings = std::accumulate( updated_shape.begin(), updated_shape.begin()+dim_to_expand, 1UL, []( auto x, auto y ) noexcept { return x*y; } );
        unsigned long const repeats = new_shape[dim_to_expand];
        unsigned long const tailings = std::accumulate( updated_shape.begin()+dim_to_expand, updated_shape.end(), 1UL, []( auto x, auto y ) noexcept { return x*y; } );

        std::vector<unsigned long> expanded_shape = updated_shape;
        expanded_shape[dim_to_expand] = new_shape[dim_to_expand];
        Tsor ans{ expanded_shape };
        view_3d v3{ ans.data(), headings, repeats, tailings }; // 3D view of ans
        view_2d v2{ _ans.data(), headings, tailings }; // 2D view of _ans

        for ( auto r : range( headings ) )
            for ( auto c : range( repeats ) )
                for ( auto ch : range( tailings ) )
                    v3[r][c][ch] = v2[r][ch];

        ans.reshape( expanded_shape );
        return broadcast_tensor( ans, new_shape ); // not necessarily done in a single loop
    }

    ///
    /// @brief Calculate the broadcasting shape for two tensors.
    /// Examples taken from numpy:
    ///
    ///  A      (2d array):  5 x 4
    ///  B      (1d array):      1
    ///  Result (2d array):  5 x 4
    ///
    ///  A      (2d array):  5 x 4
    ///  B      (1d array):      4
    ///  Result (2d array):  5 x 4
    ///
    ///  A      (3d array):  15 x 3 x 5
    ///  B      (3d array):  15 x 1 x 5
    ///  Result (3d array):  15 x 3 x 5
    ///
    ///  A      (3d array):  15 x 3 x 5
    ///  B      (2d array):       3 x 5
    ///  Result (3d array):  15 x 3 x 5
    ///
    ///  A      (3d array):  15 x 3 x 5
    ///  B      (2d array):       3 x 1
    ///  Result (3d array):  15 x 3 x 5
    ///
    /// @param shape_a Tensor shape.
    /// @param shape_b Tensor shape.
    ///
    /// @return Broadcasted shape of \ref shape_a and \ref shape_b.
    ///
    ///
    inline std::vector<unsigned long> broadcast_shape( std::vector<unsigned long> const& shape_a, std::vector<unsigned long> const& shape_b ) noexcept
    {
        if (shape_a == shape_b)
            return shape_a;

        std::vector<unsigned long> ans;
        long int const size_a = shape_a.size();
        long int const size_b = shape_b.size();

        for ( long int idx = 0; true; ++idx )
        {
            long int const a_index = size_a - 1 - idx;
            long int const b_index = size_b - 1 - idx;

            if (a_index < 0 && b_index < 0)
                break;

            unsigned long const dim_a = (a_index < 0) ? 1 : shape_a[a_index];
            unsigned long const dim_b = (b_index < 0) ? 1 : shape_b[b_index];

            if (dim_a == 1)
            {
                ans.push_back( dim_b );
                continue;
            }

            if (dim_b == 1)
            {
                ans.push_back( dim_a );
                continue;
            }

            better_assert( dim_a == dim_b, fmt::format("broadcasting: expecting same dimension, but got dim_a = {}, dim_b = {}", dim_a, dim_b) );
            ans.push_back( dim_a );
        }

        std::reverse( ans.begin(), ans.end() );
        return ans;
    }


    // C <= A * B
    // where A or A' is [m x n], B or B' is [n x k] and C is [m x k]
    template< typename T > requires std::floating_point<T>
    void gemm_cpu( T const* A, bool a_transposed, T const* B, bool b_transposed, unsigned long m, unsigned long n, unsigned long k, T* C )
    {
        auto a_view = view_2d{ A, m, n, a_transposed };
        auto b_view = view_2d{ B, n, k, b_transposed };
        auto c_view = view_2d{ C, m, k };

        std::fill_n( C, m*k, T{0} );

        if ( a_transposed == false && b_transposed == false )
            for ( auto r = 0UL; r != m; ++r )
                for ( auto idx = 0UL; idx != n; ++idx )
                    for ( auto c = 0UL; c != k; ++c )
                        c_view[r][c] += a_view[r][idx] * b_view[idx][c];
        else if ( a_transposed == false && b_transposed == true )
            for ( auto r = 0UL; r != m; ++r )
                for ( auto idx = 0UL; idx != n; ++idx )
                    for ( auto c = 0UL; c != k; ++c )
                        c_view[r][c] += a_view[r][idx] * b_view[c][idx];
        else if ( a_transposed == true && b_transposed == false )
            for ( auto r = 0UL; r != m; ++r )
                for ( auto idx = 0UL; idx != n; ++idx )
                    for ( auto c = 0UL; c != k; ++c )
                        c_view[r][c] += a_view[idx][r] * b_view[idx][c];
        else
            for ( auto r = 0UL; r != m; ++r )
                for ( auto idx = 0UL; idx != n; ++idx )
                    for ( auto c = 0UL; c != k; ++c )
                        c_view[r][c] += a_view[idx][r] * b_view[c][idx];
    }

    // this function is used to update the threshod 'cuda_gemm_threshold' defined in '../config.hpp', only considering float case
    inline void update_cuda_gemm_threshold()
    {
        if constexpr( cuda_mode == 0 )
        {
            cuda_gemm_threshold = std::numeric_limits<unsigned long>::max(); // very larger threshold to stop from using CUDA
        }
        else
        {
            //warm-up GPU
            {
                auto A = tensor<float>({128, 128});
                auto B = tensor<float>({128, 128});
                auto C = tensor<float>({128, 128});
                cuda_gemm( A.data(), false, B.data(), false, 128, 128, 128, C.data() );
            }

            unsigned long dim = 16;
            unsigned long increasement = 16;

            while ( true )
            {
                auto A = tensor<float>( {dim*dim,} );
                auto B = tensor<float>( {dim*dim,} );
                auto C = tensor<float>( {dim*dim,} );
                unsigned long t_gpu = time_it( [&](){ cuda_gemm( A.data(), false, B.data(), false, dim, dim, dim, C.data() ); });
                unsigned long t_cpu = time_it( [&](){ gemm_cpu( A.data(), false, B.data(), false, dim, dim, dim, C.data() ); });

                if ( t_cpu > t_gpu ) break;

                dim += increasement;
            }

            cuda_gemm_threshold = dim * dim * dim;
        }
    }

    // C <= A * B
    // where A or A' is [m x n], B or B' is [n x k] and C is [m x k]
    template< typename T > requires std::floating_point<T>
    void gemm( T const* A, bool a_transposed, T const* B, bool b_transposed, unsigned long m, unsigned long n, unsigned long k, T* C )
    {
        if ( cuda_gemm_threshold == 0 ) // global variable defined in config.h
            update_cuda_gemm_threshold();

        if constexpr( cuda_mode )
        {
            unsigned long const operations = m * n * k;

            if ( operations >= cuda_gemm_threshold )
                cuda_gemm( A, a_transposed, B, b_transposed, m, n, k, C );
            else
                gemm_cpu( A, a_transposed, B, b_transposed, m, n, k, C );
        }
        else if constexpr( cblas_mode )
        {
            cblas_gemm( A, a_transposed, B, b_transposed, m, n, k, C );
        }
        else
        {
            gemm_cpu( A, a_transposed, B, b_transposed, m, n, k, C );
        }
    }

    template< typename T >  requires std::floating_point<T> // this one only for non-transposed 2d View
    void gemm( view_2d<T> const& x, view_2d<T> const& y, view_2d<T>& ans ) //note: direct copy of x and y
    {
        auto const [x_row, x_col] = x.shape();
        auto const [y_row, y_col] = y.shape();
        auto const [a_row, a_col] = ans.shape();

        better_assert( x_row == a_row );
        better_assert( y_col == a_col );
        better_assert( x_col == y_row, "Expecting x_col == y_row, but x_col = ", x_col, ", and y_row = ", y_row );

        gemm( x.data(), x.transposed_, y.data(), y.transposed_, x_row, x_col, y_col, ans.data() );
    }

    // always prefer channel-last data format
    // Example:
    //
    // ( 3x2 )  + ( 1x2 )  --> ( 3x2 )
    //
    // [ 1, 2 ]                [ 0, 3 ]
    // [ 3, 4 ] + [  -1, 1 ] = [ 2, 5 ]
    // [ 5, 6 ]                [ 4, 7 ]
    //
    template< Tensor Tsor >
    Tsor add( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        auto const& broadcasted_shape = broadcast_shape( lhs.shape(), rhs.shape() );
        auto llhs = broadcast_tensor( lhs, broadcasted_shape );
        auto const& rrhs = broadcast_tensor( rhs, broadcasted_shape );

        for_each( llhs.begin(), llhs.end(), rrhs.begin(), []( auto& x, auto const& y ) noexcept { x += y; } );
        return llhs;
    }

    template< Tensor Tsor >
    Tsor operator + ( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        return add( lhs, rhs );
    }

    template< Tensor Tsor >
    Tsor operator + ( typename Tsor::value_type const& lhs, Tsor const& rhs ) noexcept
    {
        auto ans = rhs.deep_copy();
        ans.map( [lhs]( auto& v ){ v += lhs; } );
        return ans;
    }

    template< Tensor Tsor >
    Tsor operator + ( Tsor const& lhs, typename Tsor::value_type const& rhs ) noexcept
    {
        return rhs + lhs;
    }

    template< Tensor Tsor >
    Tsor minus( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        auto const& broadcasted_shape = broadcast_shape( lhs.shape(), rhs.shape() );
        auto llhs = broadcast_tensor( lhs, broadcasted_shape );
        auto const& rrhs = broadcast_tensor( rhs, broadcasted_shape );

        for_each( llhs.begin(), llhs.end(), rrhs.begin(), []( auto& x, auto const& y ) noexcept { x -= y; } );
        return llhs;
        //return add( lhs, -rhs );
    }

    template< Tensor Tsor >
    Tsor operator - ( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        return minus( lhs, rhs );
    }

    template< Tensor Tsor >
    Tsor operator - ( typename Tsor::value_type const& lhs, Tsor const& rhs ) noexcept
    {
        auto ans = rhs.deep_copy();
        ans.map( [lhs]( auto& v ){ v = lhs - v; } );
        return ans;
    }

    template< Tensor Tsor >
    Tsor operator - ( Tsor const& lhs, typename Tsor::value_type const& rhs ) noexcept
    {
        auto ans = lhs.deep_copy();
        ans.map( [rhs]( auto& v ){ v -= rhs; } );
        return ans;
    }

    template< Tensor Tsor >
    Tsor operator * ( typename Tsor::value_type const& lhs, Tsor const& rhs ) noexcept
    {
        auto ans = rhs.deep_copy();
        ans.map( [lhs]( auto& v ){ v *= lhs; } );
        return ans;
    }

    template< Tensor Tsor >
    Tsor operator * ( Tsor const& lhs, typename Tsor::value_type const& rhs ) noexcept
    {
        return rhs * lhs;
    }

    template< Tensor Tsor >
    Tsor operator / ( Tsor const& lhs, typename Tsor::value_type const& rhs ) noexcept
    {
        auto ans = lhs.deep_copy();
        ans.map( [rhs]( auto& v ){ v /= rhs; } );
        return ans;
    }

    template< Tensor Tsor >
    Tsor reshape( Tsor const& ts, std::vector<unsigned long> const& new_shape )
    {
        Tsor ans = ts;
        return ans.reshape( new_shape );
    }

    template< Tensor Tsor >
    void multiply( Tsor const& lhs, Tsor const& rhs, Tsor& ans ) noexcept
    {
        if ( 1 == lhs.ndim() )
            return multiply( reshape( lhs, {1UL, lhs.size()} ), rhs, ans );

        if ( 1 == rhs.ndim() )
            return multiply( lhs, reshape( rhs, {lhs.size(), 1UL} ), ans );

        better_assert( 2 == rhs.ndim(), "expecting rhs tensor has 2 dimensions, but got ", rhs.ndim() );

        if ( 2 == lhs.ndim() )
        {
            typedef typename Tsor::value_type value_type;
            auto const& lhs_shape = lhs.shape();
            auto const& rhs_shape = rhs.shape();

            view_2d<value_type> const x{ lhs.data(), lhs_shape[0], lhs_shape[1] };
            view_2d<value_type> const y{ rhs.data(), rhs_shape[0], rhs_shape[1] };
            auto const [row, col] = std::make_pair( lhs_shape[0], rhs_shape[1] );
            ans.resize( {row, col} );
            view_2d<value_type> z{ ans.data(), row, col };
            gemm( x, y, z );
            return;
        }
        better_assert( false, "dimension not match, lhs dimension is ", lhs.ndim() );
    }

    template< Tensor Tsor >
    Tsor multiply( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        Tsor ans;
        multiply( lhs, rhs, ans );
        return ans;
    }

    template< Tensor Tsor >
    Tsor operator * ( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        return multiply( lhs, rhs );
    }

    // caution: only valid for channel last case
    template< Tensor Tsor >
    Tsor elementwise_product( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        unsigned long const l_size = lhs.size();
        unsigned long const r_size = rhs.size();
        if ( l_size < r_size ) return elementwise_product(rhs, lhs);

        unsigned long const repeats = l_size / r_size;
        better_assert( (r_size * repeats) == l_size, "Dimension is not match!" );

        Tsor ans = lhs.deep_copy();
        for ( auto idx : range( repeats ) )
            for ( auto jdx : range( r_size ) )
            {
                ans[idx*r_size+jdx] *= rhs[jdx];
            }

        return ans;
    }

    template< Tensor Tsor >
    Tsor hadamard_product( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        return elementwise_product( lhs, rhs );
    }

    template< Tensor Tsor >
    Tsor elementwise_divide( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        better_assert( lhs.shape() == rhs.shape(), "Shape not match!" );
        Tsor ans{ lhs.shape() };
        for_each( lhs.begin(), lhs.end(), rhs.begin(), ans.begin(), []( auto x, auto y, auto& z ){ z = x/y; } );
        return ans;
    }

    template< Tensor Tsor >
    Tsor repeat( Tsor const& tsor, unsigned long n )
    {
        Tsor ans{ {n, tsor.size()} };
        {
            auto itor = ans.data();
            for  ( auto idx : range(n) )
                std::copy( tsor.begin(), tsor.end(), itor + idx * tsor.size() );

            std::vector<unsigned long>  new_shape;
            new_shape.push_back( n );
            std::copy( tsor.shape().begin(), tsor.shape().end(), std::back_inserter( new_shape ) );
            ans.reshape( new_shape );
        }

        return ans;
    }

    template< Tensor Tsor >
    Tsor reduce_sum( Tsor const& tsor )
    {
        auto result = std::reduce( tsor.data(), tsor.data()+tsor.size(), typename Tsor::value_type{0} );
        return Tsor{ std::vector<unsigned long>{1}, {result,} };
    }

    template< Tensor Tsor >
    Tsor reduce_mean( Tsor const& tsor )
    {
        auto ans = reduce_sum( tsor );
        ans /= tsor.size();
        return ans;
    }

    template< Tensor Tsor >
    Tsor clip( Tsor& tsor, typename Tsor::value_type lower = 0, typename Tsor::value_type upper = 1 )
    {
        for_each( tsor.data(), tsor.data()+tsor.size(), [lower, upper]( typename Tsor::value_type& v ){ v = std::min( upper, v ); v = std::max( lower, v ); }  );
        return tsor;
    }

    template< Tensor Tsor >
    Tsor squeeze( Tsor const& tsor )
    {
        Tsor ans{ tsor };

        if ( 0 == tsor.size() )
            return ans;

        if ( 1 == tsor.size() )
        {
            ans.reshape( {1,} );
            return ans;
        }

        std::vector<unsigned long> new_shape;
        for ( auto s : tsor.shape() )
            if (s > 1 )
                new_shape.push_back( s );
        ans.reshape( new_shape );
        return ans;
    }

    template< typename T, typename A=default_allocator<T> >
    tensor<T,A> randn( std::vector<unsigned long> const& shape, T mean=T{0}, T stddev=T{1} )
    {
        std::normal_distribution<T> distribution( mean, stddev );
        tensor<T,A> ans{ shape };
        std::generate( ans.data(), ans.data()+ans.size(), [&distribution](){ return distribution(random_generator); } );
        return ans;
    }

    template< typename T, typename A=default_allocator<T> >
    tensor<T,A> truncated_normal( std::vector<unsigned long> const& shape, T mean=T{0}, T stddev=T{1}, T lower=T{0}, T upper=T{1} )
    {
        std::normal_distribution<T> distribution( mean, stddev );
        tensor<T,A> ans{ shape };
        for ( auto& v : ans )
        {
            for ( ;; )
            {
                T x = distribution(random_generator);
                if ( x >= lower && x <= upper )
                {
                    v = x;
                    break;
                }
            }
        }

        return ans;
    }

    template< Tensor Tsor >
    Tsor poisson( Tsor const& tsor )
    {
        std::poisson_distribution<long> distribution( 1 ); // Note: only integer type accepted for a std::poisson distribution
        Tsor ans{ tsor.shape() };
        for ( auto idx : range( ans.size() ) ) // Note: cannot parallel here
        {
            long const v = static_cast<long>(tsor[idx]);
            ans[idx] = distribution( random_generator, std::poisson_distribution<long>::param_type(v) );
        }
        return ans;
    }

    template< typename T, typename A=default_allocator<T> >
    tensor<T,A> random( std::vector<unsigned long> const& shape, T min=T{0}, T max=T{1} )
    {
        std::uniform_real_distribution<T> distribution( min, max );
        tensor<T,A> ans{ shape };
        std::generate( ans.data(), ans.data()+ans.size(), [&distribution](){ return distribution(random_generator); } );
        return ans;
    }

    template< Tensor Tsor >
    Tsor random_like( Tsor const& tsor, typename Tsor::value_type min = 0, typename Tsor::value_type max = 1 )
    {
        return random<typename Tsor::value_type, typename Tsor::allocator>( tsor.shape(), min, max );
    }

    template< Tensor Tsor >
    Tsor randn_like( Tsor const& tsor, typename Tsor::value_type mean = 0, typename Tsor::value_type stddev = 1 )
    {
        return randn<typename Tsor::value_type, typename Tsor::allocator>( tsor.shape(), mean, stddev );
    }

    // TODO glorot_normal
    //
    // Glorot, Xavier, and Yoshua Bengio. “Understanding the Difficulty of Training Deep Feedforward Neural Networks.” In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, 249–256, 2010.
    template< typename T, typename A=default_allocator<T> >
    tensor<T,A> glorot_uniform( std::initializer_list<unsigned long> shape )
    {
        T prev_dim = *(shape.begin());
        T next_dim = *(shape.begin()+1);
        T const bound = std::sqrt( T{6} / (std::max(T{1}, prev_dim+next_dim)) );
        return random<T,A>( shape, -bound, bound );
    }

    template< Tensor Tsor >
    Tsor copy( Tsor const& tsor )
    {
        return deep_copy( tsor );
    }

    template< Tensor Tsor >
    Tsor concatenate( Tsor const& lhs, Tsor const& rhs, unsigned long axis=0 ) noexcept
    {
        if ( lhs.ndim() < rhs.ndim() )
            return concatenate( rhs, lhs, axis );

        // axis alignment
        if ( lhs.ndim() > rhs.ndim() )
        {
            unsigned long const dims_to_repeat = std::accumulate( lhs.shape().begin(), lhs.shape().begin()+lhs.ndim()-rhs.ndim(), 1UL, [](auto x, auto y ){ return x*y; } );
            auto new_rhs = repeat( rhs, dims_to_repeat );
            std::vector<unsigned long> new_shape{ lhs.shape().begin(), lhs.shape().begin()+lhs.ndim()-rhs.ndim() };
            std::copy( rhs.shape().begin(), rhs.shape().end(), std::back_inserter( new_shape ) );
            new_rhs.reshape( new_shape );
            return concatenate( lhs, new_rhs, axis );
        }

        auto l_shape = lhs.shape();
        auto r_shape = rhs.shape();
        better_assert( (l_shape.size() == r_shape.size()), "dimension not match, lhs dim is ", l_shape.size(), " and last dim ", *(l_shape.rbegin()),  ", but rhs dim is ", r_shape.size(), " where the last dim ", *(r_shape.rbegin()) );
        axis = (axis == (unsigned long)(-1)) ? (l_shape.size()-1) : axis;
        better_assert( (l_shape.size() > axis), "axis is too large: axis-", axis, " but allowed range-[0,", l_shape.size()-1, "]" );
        better_assert( (std::vector<unsigned long>{l_shape.begin(), l_shape.begin()+axis} == std::vector<unsigned long>{r_shape.begin(), r_shape.begin()+axis}) );
        better_assert( (std::vector<unsigned long>{l_shape.begin()+axis+1, l_shape.end()} == std::vector<unsigned long>{r_shape.begin()+axis+1, r_shape.end()}) );

        unsigned long const memory_copy_times = std::max( 1UL, std::accumulate( l_shape.begin(), l_shape.begin()+axis, 1UL, []( auto x, auto y ){ return x*y; } ) );
        unsigned long const l_memory_stride = std::accumulate( l_shape.begin()+axis, l_shape.end(), 1UL, []( auto x, auto y ){ return x*y; } );
        unsigned long const r_memory_stride = std::accumulate( r_shape.begin()+axis, r_shape.end(), 1UL, []( auto x, auto y ){ return x*y; } );

        std::vector<unsigned long> result_shape = l_shape;
        result_shape[axis] += r_shape[axis];
        Tsor ans{ result_shape };
        auto target_memory_position = ans.data();

        for ( auto idx = 0UL; idx != memory_copy_times; ++idx )
        {
            std::copy_n( lhs.data() + l_memory_stride*idx, l_memory_stride, target_memory_position );
            target_memory_position += l_memory_stride;
            std::copy_n( rhs.data() + r_memory_stride*idx, r_memory_stride, target_memory_position );
            target_memory_position += r_memory_stride;
        }

        return ans;
    }

    template< Tensor Tsor >
    Tsor repmat( Tsor const& tsor, unsigned long row_rep, unsigned long col_rep )
    {
        better_assert( tsor.shape().size() == 2, "Only 2D array has repmat method, the input array has ", tsor.shape().size(), " dimensions!" );
        auto const& old_shape = tsor.shape();
        auto const [old_row, old_col] = std::make_pair( old_shape[0], old_shape[1] );

        Tsor ans{ {old_row*row_rep, old_col*col_rep} };
        // fill cols
        for ( auto rdx = 0UL; rdx != row_rep; ++rdx )
            for ( auto cdx = 0UL; cdx != col_rep; ++cdx )
                for ( auto odx = 0UL; odx != old_row; ++odx )
                    std::copy_n( tsor.data() + odx * old_col, old_col, ans.data() + old_row * old_col * rdx * col_rep + odx * old_col * col_rep + old_col * cdx );

        return ans;
    }

    template< typename T, typename A=default_allocator<T> >
    constexpr tensor<T,A> zeros( std::vector<unsigned long> const& shape )
    {
        return {shape};
    }

    template< Tensor Tsor >
    constexpr Tsor zeros_like( Tsor const& tsor )
    {
        return {tsor.shape()};
    }

    template< typename T, typename A=default_allocator<T> >
    constexpr tensor<T,A> ones( std::vector<unsigned long> const& shape )
    {
        tensor<T, A> ans{ shape };
        std::fill( ans.data(), ans.data() + ans.size(), T{1} );
        return ans;
    }

    template< Tensor Tsor >
    constexpr Tsor ones_like( Tsor const& tsor )
    {
        return ones<typename Tsor::value_type, typename Tsor::allocator>( tsor.shape() );
    }

    template< Tensor Tsor >
    auto max( Tsor const& tsor )
    {
        typedef typename Tsor::value_type value_type;
        better_assert( tsor.size() != 0, "tensor::max error: input tensor should not be empty!" );
        if ( tsor.size() == 0 ) return value_type{0};
        value_type ans = std::numeric_limits<value_type>::min();
        for ( auto idx : range( tsor.size() ) )
            ans = std::max( tsor[idx], ans );
        return ans;
    }

    template< Tensor Tsor >
    auto amax( Tsor const& tsor )
    {
        return max( tsor );
    }

    template< Tensor Tsor >
    auto min( Tsor const& tsor )
    {
        typedef typename Tsor::value_type value_type;
        better_assert( tsor.size() != 0, "tensor::min error: input tensor should not be empty!" );
        if ( tsor.size() == 0 ) return value_type{0};
        value_type ans = std::numeric_limits<value_type>::max();
        for ( auto idx : range( tsor.size() ) )
            ans = std::min( tsor[idx], ans );
        return ans;
    }

    template< Tensor Tsor >
    auto amin( Tsor const& tsor )
    {
        return min( tsor );
    }

    template< Tensor Tsor >
    auto sum( Tsor const& tsor )
    {
        typedef typename Tsor::value_type value_type;
        better_assert( tsor.size() != 0, "tensor::sum error: input tensor should not be empty!" );
        return std::accumulate( tsor.data(), tsor.data()+tsor.size(), value_type{0} );
    }

    template< Tensor Tsor >
    auto mean( Tsor const& tsor )
    {
        better_assert( tsor.size() != 0, "tensor::mean error: input tensor should not be empty!" );
        if ( 0 == tsor.size() ) return typename Tsor::value_type{0};
        return sum( tsor ) / tsor.size();
    }

    template< Tensor Tsor >
    auto norm( Tsor const& tsor )
    {
        typedef typename Tsor::value_type value_type;
        better_assert( tsor.size() != 0, "tensor::sum error: input tensor should not be empty!" );
        return std::sqrt( std::accumulate( tsor.data(), tsor.data()+tsor.size(), value_type{0}, []( value_type x, value_type y ){ return x + y*y; }  ) ) / static_cast<value_type>( tsor.size() );
    }

    template< Tensor Tsor >
    Tsor abs( Tsor const& tsor )
    {
        auto ans = tsor.deep_copy();
        ans.map( []( auto& x ){ x = std::abs( x ); } );
        return ans;
    }

    template< Tensor Tsor >
    Tsor softmax( Tsor const& tsor )
    {
        typedef typename Tsor::value_type value_type;
        better_assert( !tsor.empty(), "softmax argument is an empty tensor. " );
        Tsor ans = tsor.deep_copy();
        unsigned long const last_dim = *(tsor.shape().rbegin());
        unsigned long const rem_dim = tsor.size() / last_dim;
        view_2d<value_type> mat{ ans.data(), rem_dim, last_dim };
        for ( auto idx : range( rem_dim ) )
        {
            value_type const mx = *std::max_element( mat[idx], mat[idx+1] );
            for_each( mat[idx], mat[idx+1], [mx]( auto& v ){ v -= mx; } );
            value_type const ac = std::accumulate( mat[idx], mat[idx+1], value_type{0}, []( value_type init, value_type val ){ return init + std::exp(val); } );
            for_each( mat[idx], mat[idx+1], [ac]( auto& v ){ v = std::exp(v) / (ac+eps); } );
        }
        return ans;
    }

    template< Tensor Tsor >
    bool has_nan( Tsor const& tsor )
    {
        return (tsor.data() + tsor.size()) != std::find_if( tsor.data(), tsor.data()+tsor.size(), []( auto const& v ){ return std::isnan( v ); } );
    }

    template< Tensor Tsor >
    bool has_inf( Tsor const& tsor )
    {
        return (tsor.data() + tsor.size()) != std::find_if( tsor.data(), tsor.data()+tsor.size(), []( auto const& v ){ return std::isinf( v ); } );
    }

    template< Tensor Tsor >
    bool is_valid( Tsor const& tsor )
    {
        return (!has_nan(tsor)) && (!has_inf(tsor));
    }

    template< Tensor Tsor, typename Function >
    Tsor reduce( Tsor const& ts, unsigned long axis, typename Tsor::value_type const& init, Function const& func, bool keepdims=false ) noexcept
    {
        if ( ts.empty() ) return ts;

        axis = (axis == static_cast<unsigned long>( -1 )) ? ts.ndim()-1 : axis;
        better_assert( axis < ts.ndim(), "Error with tensor::reduce, input axis ", axis, " is too large for a tensor with ", ts.ndim(), " dimensions." );

        std::vector<unsigned long> _shape = ts.shape();
        unsigned long const pres = std::reduce( _shape.begin(), _shape.begin()+axis, 1Ul, []( unsigned long x, unsigned long y ){ return x*y; } );
        unsigned long const post = std::reduce( _shape.begin()+axis+1, _shape.end(), 1Ul, []( unsigned long x, unsigned long y ){ return x*y; } );

        unsigned long const n = _shape[axis];
        _shape[axis] = 1UL;

        Tsor ans{ _shape };
        auto itor = ans.begin();
        for ( auto idx : range( pres ) )
            for ( auto jdx : range( post ) )
            {
                auto start = ts.begin() + idx * post * n + jdx;
                stride_iterator si{ start, static_cast<std::int64_t>(post) };
                *itor++ = std::reduce( si, si+n, init, func );
            }

        if ( !keepdims )
        {
            std::copy( _shape.begin()+axis+1, _shape.end(), _shape.begin()+axis );
            _shape.resize( _shape.size() - 1 );
            ans.reshape( _shape );
        }

        return ans;
    }

    template <Tensor Tsor>
    Tsor sum( Tsor const& ts, unsigned long axis, bool keepdims=false ) noexcept
    {
        return reduce( ts, axis, typename Tsor::value_type{0}, []( auto const& a, auto const& b ){ return a+b; }, keepdims );
    }

    template <Tensor Tsor> requires std::floating_point<typename Tsor::value_type>
    Tsor mean( Tsor const& ts, unsigned long axis, bool keepdims=false ) noexcept
    {
        typedef typename Tsor::value_type value_type;
        axis = ( axis == static_cast<unsigned long>( -1 ) ) ? ts.ndim()-1 : axis;
        auto const& _shape = ts.shape();
        return reduce( ts, axis, value_type{0}, []( auto const& a, auto const& b ){ return a+b; }, keepdims ) / static_cast<value_type>( _shape[axis] );
    }

    template <Tensor Tsor> requires std::floating_point<typename Tsor::value_type>
    Tsor variance( Tsor const& ts, unsigned long axis, bool keepdims=false ) noexcept
    {
        Tsor x = mean( ts, axis, true );
        x = x - ts;
        for_each( x.begin(), x.end(), [](auto& v){ v *= v; } );
        return mean( x, axis, keepdims );
    }

    template <Tensor Tsor> requires std::floating_point<typename Tsor::value_type>
    Tsor standard_deviation( Tsor const& ts, unsigned long axis, bool keepdims=false ) noexcept
    {
        Tsor x = variance( ts, axis, keepdims );
        for_each( x.begin(), x.end(), [](auto& v){ v = std::sqrt(v); } );
        return x;
    }

    template <Tensor Tsor> requires std::floating_point<typename Tsor::value_type>
    typename Tsor::value_type var( Tsor const& ts ) noexcept
    {
        auto x = ts - mean(ts);
        return std::inner_product( x.begin(), x.end(), x.begin(), typename Tsor::value_type{0} );
    }

    template <Tensor Tsor> requires std::floating_point<typename Tsor::value_type>
    typename Tsor::value_type std( Tsor const& ts ) noexcept
    {
        return std::sqrt( var(ts) );
    }

    template <Tensor Tsor>
    Tsor max( Tsor const& ts, unsigned long axis, bool keepdims=false ) noexcept
    {
        return reduce( ts, axis, std::numeric_limits<typename Tsor::value_type>::min(), []( auto const& a, auto const& b ){ return a > b ? a : b; }, keepdims );
    }

    template <Tensor Tsor>
    Tsor min( Tsor const& ts, unsigned long axis, bool keepdims=false ) noexcept
    {
        return reduce( ts, axis, std::numeric_limits<typename Tsor::value_type>::max(), []( auto const& a, auto const& b ){ return a < b ? a : b; }, keepdims );
    }

    template < typename T, typename A=default_allocator<T> > requires std::floating_point<T>
    tensor<T,A> linspace( T start, T stop, unsigned long num, bool endpoint=true ) noexcept
    {
        better_assert( num > 1, "tensor::linspace: expecting number larger than 1, but got ", num );

        unsigned long const segs = endpoint ? num-1 : num;
        T const distance = ( stop - start ) / segs;

        tensor<T,A> ans{ {num,} };
        for ( auto idx : range( num ) )
            ans[idx] = start + distance * idx; // 1D view of the tensor

        return ans;
    }

    template<class _Tp, class _CharT, class _Traits, class _Alloc>
    std::basic_istream<_CharT, _Traits>& read_tensor(std::basic_istream<_CharT, _Traits>& __is, tensor<_Tp, _Alloc>& __x)
    {
        better_assert( __is.good(), "Error with the istream!" );

        // read the first line to extract shape
        std::vector<unsigned long> shape;
        {
            std::string s_shape;
            std::getline( __is, s_shape );
            std::stringstream ss( s_shape );
            std::copy( std::istream_iterator<unsigned long>( ss ), std::istream_iterator<unsigned long>(), std::back_inserter( shape ) );
        }

        // read data
        std::vector< _Tp > buff;
        {
            std::string cache;
            std::getline( __is, cache );
            std::stringstream ss( cache );
            std::copy( std::istream_iterator< _Tp >( ss ), std::istream_iterator< _Tp >(), std::back_inserter( buff ) );
        }

        // copy and return
        tensor<_Tp, _Alloc> ans{ shape };
        __x.resize( shape );
        {
            better_assert( __x.size() == buff.size(), "tensor::loadtxt: shape suggests size of ", __x.size(), " but got ", buff.size() );
            std::copy( buff.begin(), buff.end(), __x.begin() );
        }

        return __is;
    }

    template<class _Tp, class _CharT, class _Traits, class _Alloc>
    std::basic_ostream<_CharT, _Traits>& write_tensor(std::basic_ostream<_CharT, _Traits>& __os, tensor<_Tp, _Alloc> const& __x)
    {
        std::basic_ostringstream<_CharT, _Traits> __s;
        __s.flags(__os.flags());
        __s.imbue(__os.getloc());
        __s.precision(__os.precision());

        {//write shape
            auto const& shape = __x.shape();
            std::copy( shape.begin(), shape.end(), std::ostream_iterator<unsigned long>{ __os, " " } );
            __os << "\n";
        }
        {//write data
            std::copy( __x.begin(), __x.end(), std::ostream_iterator<_Tp>{ __os, " " } );
        }
        __os << "\n";

        return __os;
    }

    //
    // file format:
    //
    // first line: shape
    // second line: data
    //
    // example of a tensor of shape (2, 3):
    //
    // 2 3
    // 0.910905 0.525709 0.584262 0.34063 0.613034 0.0803866
    //
    template < typename T, typename A=default_allocator<T> >
    tensor<T,A> load_tensor( std::string const& file_name )
    {
        tensor<T, A> ans;
        std::ifstream ifs{ file_name };
        read_tensor( ifs, ans );
        ifs.close();
        return ans;
    }

    template< Tensor Tsor >
    void save_tensor( std::string const& file_name, Tsor const& tsor )
    {
        std::ofstream ofs{ file_name };
        write_tensor( ofs, tsor );
        ofs.close();
    }

    namespace
    {

        template< Tensor Tsor>
        void flip_3D( Tsor const& tsor, Tsor& ans, int axis )
        {
            std::vector<unsigned long> shape = tsor.shape();
            auto [r, c, ch] = std::make_tuple( shape[0], shape[1], shape[2] );

            auto src = view_3d{ tsor.data(), r, c, ch };
            auto dst = view_3d{ ans.data(), r, c, ch };

            if ( 0 == axis )
            {
                for ( auto _r : range(r) )
                    for ( auto _c : range(c) )
                        for ( auto _ch : range(ch) )
                            dst[r-_r-1][_c][_ch] = src[_r][_c][_ch];
                return;
            }

            if ( 1 == axis )
            {
                for ( auto _r : range(r) )
                    for ( auto _c : range(c) )
                        for ( auto _ch : range(ch) )
                            dst[_r][c-_c-1][_ch] = src[_r][_c][_ch];
                return;
            }

            for ( auto _r : range(r) )
                for ( auto _c : range(c) )
                    for ( auto _ch : range(ch) )
                        dst[_r][_c][ch-_ch-1] = src[_r][_c][_ch];
        }

        // impls flip for 1D, 2D, 3D and 4D.
        // ceras only considers up to 4D tensors.
        template< Tensor Tsor>
        void flip_1D( Tsor const& tsor, Tsor& ans, int /* axis can only be 0 for 1D case */ )
        {
            std::copy( tsor.begin(), tsor.end(), ans.rbegin() );
        }

        template< Tensor Tsor>
        void flip_2D( Tsor const& tsor, Tsor& ans, int axis )
        {
            std::vector<unsigned long> const& shape = tsor.shape();
            std::vector<unsigned long> const new_shape{ {shape[0], shape[1], 1} };
            Tsor _tsor = tsor;
            _tsor.reshape( new_shape );
            ans.reshape( new_shape );

            flip_3D( _tsor, ans, axis );
            ans.reshape( shape );
        }


        template< Tensor Tsor>
        void flip_4D( Tsor const& tsor, Tsor& ans, int axis )
        {
            std::vector<unsigned long> const& shape = tsor.shape();

            if ( 0 == axis || 1 == axis) // merge dim 2 and dim 3 for case of (0, 1), then flip along axis
            {
                std::vector<unsigned long> const new_shape{ {shape[0], shape[1], shape[2]*shape[3] } };
                Tsor _tsor = tsor;
                _tsor.reshape( new_shape );
                ans.reshape( new_shape );
                flip_3D( _tsor, ans, axis );
                ans.reshape( shape );

                return;
            }

            // merge dim 0 and dim 1 for case of (2, 3), then flip along axis-1
            std::vector<unsigned long> const new_shape{ {shape[0]*shape[1], shape[2], shape[3] } };
            Tsor _tsor = tsor;
            _tsor.reshape( new_shape );
            ans.reshape( new_shape );
            flip_3D( _tsor, ans, axis-1 );
            ans.reshape( shape );
        }
    }

    template< Tensor Tsor >
    void flip( Tsor const& tsor, int axis, Tsor& ans )
    {
        if ( 0 == tsor.size() )
            return;

        unsigned long const ndim = tsor.ndim();
        better_assert( (4 >= ndim), fmt::format( "Expect a tensor up to 4D, but got {} dimensions.", ndim ) );

        if (-1 == axis)
            axis = static_cast<int>( ndim - 1 );

        better_assert( (axis < static_cast<int>(ndim)), fmt::format( "Expect a smaller axis, but got {} for a tensor with {} dimensions", axis, ndim ) );

        ans.resize( tsor.shape() );
        switch (ndim)
        {
            case 1 :
                flip_1D( tsor, ans, axis );
                break;
            case 2 :
                flip_2D( tsor, ans, axis );
                break;
            case 3 :
                flip_3D( tsor, ans, axis );
                break;
            default: // up to 4D
                flip_4D( tsor, ans, axis );
        }
    }

    template< Tensor Tsor >
    Tsor flip( Tsor const& tsor, int axis = -1 )
    {
        Tsor ans;
        flip( tsor, axis, ans );
        return ans;
    }


}//namespace ceras

#endif//CAFUTPOUSGTCEEJFWOUMQTWNSGRSWBVLLOSRSEDXYGXKHDEILDOGELEQNQBCVJRTHNETVFBND

