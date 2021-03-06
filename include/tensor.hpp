#ifndef HQKGLAXWWVFBFHQNHBVTQJKGUFTPCQPTPXDVNOSBDJIBHITCEKDISJYNAMCPLJDURURDAISFV
#define HQKGLAXWWVFBFHQNHBVTQJKGUFTPCQPTPXDVNOSBDJIBHITCEKDISJYNAMCPLJDURURDAISFV

#include "./includes.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/range.hpp"
#include "./utils/stride_iterator.hpp"
#include "./utils/for_each.hpp"
#include "./utils/buffered_allocator.hpp"
#include "./utils/debug.hpp"
#include "./utils/id.hpp"
#include "./backend/cuda.hpp"

namespace ceras
{
    // random_seed for random numbers
    static unsigned long random_seed = std::chrono::system_clock::now().time_since_epoch().count();
    // static random number random_generator
    static std::mt19937 random_generator{random_seed};

    template< typename T >
    using default_allocator = buffered_allocator<T, 256>;
    //using default_allocator = std::allocator<T>;
    //

    template< typename Tsor >
    struct tensor_log
    {
        tensor_log() noexcept
        {
            auto& zen = static_cast<Tsor&>(*this);
            debug_log( zen.name_, " created directly, with id ", zen.id_ );
        }
        tensor_log( tensor_log const& ) noexcept
        {
            auto& zen = static_cast<Tsor&>(*this);
            debug_log( zen.name_, " created using copy ctor, id ", zen.id_, " and size ", zen.size() );
        }
        tensor_log( tensor_log && ) noexcept
        {
            auto& zen = static_cast<Tsor&>(*this);
            debug_log( zen.name_, " created using move ctor, id ", zen.id_, " and size ", zen.size() );
        }
        tensor_log& operator = ( tensor_log const& ) noexcept
        {
            auto& zen = static_cast<Tsor&>(*this);
            debug_log( zen.name_, " created from copy assignment, id ", zen.id_, " and size ", zen.size() );
            return *this;
        }
        tensor_log& operator = ( tensor_log && ) noexcept
        {
            auto& zen = static_cast<Tsor&>(*this);
            debug_log( zen.name_, " created from move assigment, id ", zen.id_, " and size ", zen.size() );
            return *this;
        }
    };

    // Beware: shallow copy
    // TODO: impl tensor_view, enabling stride dataset
    template< typename T, typename Allocator = default_allocator<T> >
    struct tensor : enable_id<tensor<T, Allocator>, "Tensor">//, tensor_log<tensor<T, Allocator>>
    {
        typedef T value_type;
        typedef Allocator allocator;
        typedef std::vector<T, Allocator> vector_type;
        typedef std::shared_ptr<vector_type> shared_vector;
        typedef tensor self_type;

        std::vector<unsigned long> shape_;
        unsigned long memory_offset_;
        shared_vector vector_; //shared across different instances

        tensor slice( unsigned long m, unsigned long n ) const noexcept
        {
            better_assert( m < n, "starting dimension larger than then ending dimension." );

            better_assert( !shape_.empty(), "Cannot slice an empty tensor." );

            unsigned long first_dim = shape_[0];
            better_assert( n <= first_dim, "this tensor only has ", first_dim, " at the first dimension, too small for n = ", n );

            unsigned long rest_dims = std::accumulate( shape_.begin()+1, shape_.end(), 1UL, []( auto x, auto y ){ return x*y; } );

            tensor ans = *this;
            ans.shape_[0] = n - m;
            ans.memory_offset_ = rest_dims * m + memory_offset_;
            return ans;
        }

        constexpr auto begin() noexcept
        {
            return data();
        }

        constexpr auto begin() const noexcept
        {
            return data();
        }

        constexpr auto cbegin() const noexcept
        {
            return begin();
        }

        constexpr auto end() noexcept
        {
            return begin() + size();
        }

        constexpr auto end() const noexcept
        {
            return begin() + size();
        }

        constexpr auto cend() const noexcept
        {
            return  end();
        }

        constexpr self_type& reset( T val = T{0} )
        {
            std::fill_n( data(), size(), val );
            return *this;
        }

        constexpr unsigned long ndim() const noexcept
        {
            return shape_.size();
        }

        constexpr self_type& deep_copy( self_type const& other )
        {
            (*this).resize( other.shape() );
            std::copy_n( other.data(), size(), (*this).data() );
            return *this;
        }

        constexpr self_type const deep_copy() const
        {
            self_type ans{ shape_ };
            std::copy_n( data(), size(), ans.data() );
            return ans;
        }

        constexpr self_type const copy() const
        {
            return deep_copy();
        }

        // 1-D view
        constexpr value_type& operator[]( unsigned long idx )
        {
            return *(data()+idx);
        }

        // 1-D view
        constexpr value_type const& operator[]( unsigned long idx ) const
        {
            return *(data()+idx);
        }

        tensor() : shape_{std::vector<unsigned long>{}}, memory_offset_{0}, vector_{std::make_shared<vector_type>()}
        {
            //debug_print( "Creating an empty tensor ", (*this).id_ );
        }

        //
        // tensor<double> A{ {2,2}, { 1.0, 1.0, 1.0, 1.0} };
        //
        constexpr tensor( std::vector<unsigned long> const& shape, std::initializer_list<T> init, const Allocator& alloc = Allocator() ) : shape_{ shape }, memory_offset_{0}, vector_{std::make_shared<vector_type>(init, alloc)}
        {
            better_assert( (*vector_).size() == std::accumulate( shape_.begin(), shape_.end(), 1UL, [](auto x, auto y){ return x*y; } ), "Expecting vector has same size as the shape indicates." );
            //debug_print( "Creating a tensor from <shape,<init>> ", (*this).id_, " and size ", (*this).size() );
        }

        constexpr tensor( std::vector<unsigned long> const& shape ):shape_{shape}, memory_offset_{0}, vector_{ std::make_shared<vector_type>( std::accumulate( shape_.begin(), shape_.end(), 1UL, []( auto x, auto y ){ return x*y; } ), T{0} ) }
        {
            //debug_print( "Creating a tensor from <shape> ", (*this).id_, " and size ", (*this).size() );
        }

        constexpr tensor( std::vector<unsigned long> const& shape, T init ):shape_{shape}, memory_offset_{0}, vector_{ std::make_shared<vector_type>( std::accumulate( shape_.begin(), shape_.end(), 1UL, []( auto x, auto y ){ return x*y; } ), T{0} ) }
        {
            //debug_print( "Creating a tensor from <shape,init> ", (*this).id_, " and size ", (*this).size() );
            std::fill( begin(), end(), init );
        }

        constexpr tensor( tensor const& other, unsigned long memory_offset ) : shape_{ other.shape_ }, memory_offset_{ memory_offset }, vector_{ other.vector_ } {}

        /*
        std::vector<unsigned long> shape_;
        unsigned long memory_offset_;
        shared_vector vector_; //shared across different instances
        */

        constexpr tensor( self_type const& other ) noexcept : shape_{ other.shape_ }, memory_offset_{ other.memory_offset_ }
        {
            vector_ = other.vector_;
            (*this).id_ = other.id_;
            //debug_print( "Tensor copy construction: prev tensor ", other.id_, " has ", other.size(), " elements." );
        }

        constexpr tensor( self_type && other ) noexcept : shape_{ other.shape_ }, memory_offset_{ other.memory_offset_ }
        {
            vector_ = other.vector_;
            (*this).id_ = other.id_;
            //debug_print( "Tensor move construction: prev tensor ", other.id_, " has ", other.size(), " elements." );
        }

        constexpr self_type& operator = ( self_type const& other ) noexcept
        {
            shape_ = other.shape_;
            memory_offset_ = other.memory_offset_;
            vector_ = other.vector_;
            (*this).id_ = other.id_;
            //debug_print( "Tensor copy assigment: prev tensor ", other.id_, " has ", other.size(), " elements." );
            return *this;
        }
        constexpr self_type& operator = ( self_type && other ) noexcept
        {
            shape_ = other.shape_;
            memory_offset_ = other.memory_offset_;
            vector_ = other.vector_;
            (*this).id_ = other.id_;
            //debug_print( "Tensor move assigment: prev tensor ", other.id_, " has ", other.size(), " elements." );
            return *this;
        }

        constexpr std::vector< unsigned long > const& shape() const noexcept { return shape_; }

        constexpr unsigned long size() const noexcept
        {
            if ( !vector_ )
                return 0;
            return (*vector_).size() - memory_offset_;
        }

        constexpr self_type& resize( std::vector< unsigned long > const& new_shape )
        {
            unsigned long const new_size = std::accumulate( new_shape.begin(), new_shape.end(), 1UL, [](auto x, auto y){ return x*y; } );
            if( (*this).size() != new_size )
            {
                (*vector_).resize(new_size);
                memory_offset_ = 0UL;
            }
            (*this).shape_ = new_shape;
            return *this;
        }

        //reshape is resize
        constexpr self_type& reshape( std::vector< unsigned long > const& new_shape )
        {
            unsigned long const new_size = std::accumulate( new_shape.begin(), new_shape.end(), 1UL, [](auto x, auto y){ return x*y; } );
            better_assert( (*this).size() == new_size, "reshape: expecting same size, but the original size is ", (*this).size(), ", and the new size is ", new_size );
            (*this).shape_ = new_shape;
            return *this;
        }

        [[nodiscard]] constexpr bool empty() const noexcept
        {
            return 0 == shape_.size();
        }

        constexpr value_type* data() noexcept
        {
            return (*vector_).data() + memory_offset_;
        }

        constexpr const value_type* data() const noexcept
        {
            return (*vector_).data() + memory_offset_;
        }

        // element-wise operation
        //
        //    tensor<double> x{...};
        //    x.map( []( double v ){ return 1.0/v+1.0; } );
        //
        template< typename Function >
        constexpr self_type& map( Function const& f )
        {
            std::for_each( (*this).data(), (*this).data()+(*this).size(), [&f]( auto& v ){ f(v); } );
            return *this;
        }

        constexpr self_type& operator += ( self_type const& other )
        {
            better_assert( shape() == other.shape(), "Error with tensor::operator += : Shape mismatch!" );
            std::transform( data(), data()+size(), other.data(), data(), []( auto x, auto y ){ return x+y; } );
            return *this;
        }

        constexpr self_type& operator += ( value_type x )
        {
            std::for_each( data(), data()+size(), [x]( value_type& v ){ v += x; } );
            return *this;
        }

        constexpr self_type& operator -= ( self_type const& other )
        {
            better_assert( shape() == other.shape(), "Error with tensor::operator -=: Shape not match!" );
            std::transform( data(), data()+size(), other.data(), data(), []( auto x, auto y ){ return x-y; } );
            return *this;
        }

        constexpr self_type& operator -= ( value_type x )
        {
            std::for_each( data(), data()+size(), [x]( auto& v ){ v -= x; } );
            return *this;
        }

        constexpr self_type& operator *= ( self_type const& other )
        {
            better_assert( shape() == other.shape(), "Shape not match!" );
            std::transform( data(), data()+size(), other.data(), data(), []( auto x, auto y ){ return x*y; } );
            return *this;
        }

        constexpr self_type& operator *= ( value_type x )
        {
            std::for_each( data(), data()+size(), [x]( auto& v ){ v *= x; } );
            return *this;
        }

        constexpr self_type& operator /= ( self_type const& other )
        {
            better_assert( shape() == other.shape(), "Shape not match!" );
            std::transform( data(), data()+size(), other.data(), data(), []( auto x, auto y ){ return x/y; } );
            return *this;
        }

        constexpr self_type& operator /= ( value_type x )
        {
            std::for_each( data(), data()+size(), [x]( auto& v ){ v /= x; } );
            return *this;
        }

        constexpr self_type const operator - () const
        {
            self_type ans = (*this).deep_copy();
            std::for_each( ans.data(), ans.data()+size(), []( auto& v ){ v = -v; } );
            return  ans;
        }

    };

    template <typename T, typename A=default_allocator<T> >
    constexpr tensor<T, A> as_tensor( T val ) noexcept
    {
        tensor<T, A> ans{ {1,} };
        ans[0] = val;
        return ans;
    }

    template< typename T >
    struct is_tensor : std::false_type {};

    template< typename T, typename A >
    struct is_tensor< tensor< T, A> > : std::true_type {};

    template< class T >
    inline constexpr bool is_tensor_v = is_tensor<T>::value;

    template< typename T >
    concept Tensor = is_tensor_v<T>;

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

    template< typename T >
    struct view_2d
    {
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

        constexpr auto shape() const noexcept
        {
            return std::make_pair( row_, col_ );
        }

        constexpr unsigned long size() const noexcept
        {
            return row_ * col_;
        }

        constexpr T* data() noexcept
        {
            return data_;
        }

        constexpr const T* data() const noexcept
        {
            return data_;
        }
    };

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
                for ( auto c = 0UL; c != k; ++c )
                    for ( auto idx = 0UL; idx != n; ++idx )
                        c_view[r][c] += a_view[r][idx] * b_view[idx][c];
        else if ( a_transposed == false && b_transposed == true )
            for ( auto r = 0UL; r != m; ++r )
                for ( auto c = 0UL; c != k; ++c )
                    for ( auto idx = 0UL; idx != n; ++idx )
                        c_view[r][c] += a_view[r][idx] * b_view[c][idx];
        else if ( a_transposed == true && b_transposed == false )
            for ( auto r = 0UL; r != m; ++r )
                for ( auto c = 0UL; c != k; ++c )
                    for ( auto idx = 0UL; idx != n; ++idx )
                        c_view[r][c] += a_view[idx][r] * b_view[idx][c];
        else
            for ( auto r = 0UL; r != m; ++r )
                for ( auto c = 0UL; c != k; ++c )
                    for ( auto idx = 0UL; idx != n; ++idx )
                        c_view[r][c] += a_view[idx][r] * b_view[c][idx];
    }

    // this function is used to update the threshod 'cuda_gemm_threshold' defined in '../config.hpp', only considering float case
    inline void update_cuda_gemm_threshold()
    {
        if constexpr( cuda_mode == 0 )
        {
            cuda_gemm_threshold = std::numeric_limits<unsigned long>::max();
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
        better_assert( x_col == y_row, "Expecting x_row == y_col, but x_col = ", x_col, ", and y_row = ", y_row );

        gemm( x.data(), x.transposed_, y.data(), y.transposed_, x_row, x_col, y_col, ans.data() );
    }

    // always taking channel last data formate
    // Example:
    //
    // ( 3x2 )  + ( 1x2 )  --> ( 3x2 )
    //
    // [ 1, 2 ]                [ 0, 3 ]
    // [ 3, 4 ] + [  -1, 1 ] = [ 2, 5 ]
    // [ 5, 6 ]                [ 4, 7 ]
    //
    //template< typename T, typename A >
    //tensor<T, A> add( tensor<T, A> const& lhs, tensor<T, A> const& rhs ) noexcept
    //TODO: fix cases like [31, 1, 31, 31, 1] + [31, 1, 1, 31]
    template< Tensor Tsor >
    Tsor add( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        unsigned long const l_size = lhs.size();
        unsigned long const r_size = rhs.size();
        if ( l_size < r_size ) return rhs + lhs;

        unsigned long const repeats = l_size / r_size;
        better_assert( (r_size * repeats) == l_size, "Dimension is not match!" );

        Tsor ans = lhs.deep_copy();
        for ( auto idx : range( repeats ) )
            for ( auto jdx : range( r_size ) )
                ans[idx*r_size+jdx] = lhs[idx*r_size+jdx] + rhs[jdx];

        return ans;
    }

    template< Tensor Tsor >
    Tsor operator + ( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        return add( lhs, rhs );
    }

    template< Tensor Tsor >
    Tsor minus( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        return add( lhs, -rhs );
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
        /*
        unsigned long const l_size = lhs.size();
        unsigned long const r_size = rhs.size();
        if ( l_size < r_size ) return (-rhs) + lhs;

        unsigned long const repeats = l_size / r_size;
        better_assert( (r_size * repeats) == l_size, "Dimension is not match!" );

        Tsor ans = lhs.deep_copy();
        for ( auto idx : range( repeats ) )
            for ( auto jdx : range( r_size ) )
                ans[idx*r_size+jdx] = lhs[idx*r_size+jdx] - rhs[jdx];

        return ans;
        */
    }

    template< Tensor Tsor >
    Tsor operator - ( Tsor const& lhs, typename Tsor::value_type const& rhs ) noexcept
    {
        auto ans = lhs.deep_copy();
        ans.map( [rhs]( auto& v ){ v -= rhs; } );
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
            //Tsor ans{ std::vector<unsigned long>{ {row, col} } };
            ans.resize( {row, col} );
            view_2d<value_type> z{ ans.data(), row, col };
            gemm( x, y, z );
            //return ans;
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

#if 0
        if ( 1 == lhs.ndim() )
            return multiply( reshape( lhs, {1UL, lhs.size()} ), rhs );

        if ( 1 == rhs.ndim() )
            return multiply( lhs, reshape( rhs, {lhs.size(), 1UL} ) );

        better_assert( 2 == rhs.ndim(), "expecting rhs tensor has 2 dimensions, but got ", rhs.ndim() );

        if ( 2 == lhs.ndim() )
        {
            typedef typename Tsor::value_type value_type;
            auto const& lhs_shape = lhs.shape();
            auto const& rhs_shape = rhs.shape();

            view_2d<value_type> const x{ lhs.data(), lhs_shape[0], lhs_shape[1] };
            view_2d<value_type> const y{ rhs.data(), rhs_shape[0], rhs_shape[1] };
            auto const [row, col] = std::make_pair( lhs_shape[0], rhs_shape[1] );
            Tsor ans{ std::vector<unsigned long>{ {row, col} } };
            view_2d<value_type> z{ ans.data(), row, col };
            gemm( x, y, z );
            return ans;
        }

        /*
        if ( 3 == lhs.ndim() )
        {
            typedef typename Tsor::value_type value_type;
            auto const& lhs_shape = lhs.shape();
            auto const [batch_size, r, c] = std::make_tuple( lhs_shape[0], lhs_shape[1], lhs_shape[2] );
            auto const& rhs_shape = rhs.shape();
            auto const [_r, _c] = std::make_tuple( rhs_shape[0], rhs_shape[1]  );

            better_assert( c == _r, "expecting dimension match, but last dim of lhs is ", c, ", and first dim of rhs is ", _r );

            Tsor ans{ std::vector<unsigned long>{ {batch_size, r, _c} } };
            view_2d<value_type> const y{ rhs.data(), _r, _c };
            for ( auto bs : range( batch_size ) )
            {
                view_2d<value_type> const x{ lhs.data()+bs*r*c, r, c };
                view_2d<value_type> z{ ans.data()+r*_c, r, _c };
                gemm( x, y, z );
            }

            return ans;
        }
        */

        better_assert( false, "dimension not match, lhs dimension is ", lhs.ndim() );

        return Tsor{};
#endif
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
                //ans[idx*r_size+jdx] *= rhs[jdx];
            }

        return ans;
    }

    /*
    template< Tensor Tsor >
    Tsor elementwise_product( Tsor const& lhs, Tsor const& rhs ) noexcept
    {
        better_assert( lhs.shape() == rhs.shape(), "Shape not match!" );
        better_assert( !has_nan( lhs ), "elementwise_product: lhs tensor contains Nan!" );
        better_assert( !has_nan( rhs ), "elementwise_product: rhs tensor contains Nan!" );

        Tsor ans{ lhs.shape() };
        for_each( lhs.begin(), lhs.end(), rhs.begin(), ans.begin(), []( auto x, auto y, auto& z ){ z = x*y; } );
        return ans;
    }
    */

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

    template< Tensor Tsor > requires std::floating_point<typename Tsor::value_type>
    Tsor operator / ( Tsor const& lhs, typename Tsor::value_type const& rhs ) noexcept
    {
        Tsor ans = lhs.deep_copy();
        for_each( ans.begin(), ans.end(), [&rhs]( auto& x ){ x /= rhs; } );
        return ans;
    }

    template< Tensor Tsor >
    Tsor operator * ( typename Tsor::value_type const& lhs, Tsor const& rhs ) noexcept
    {
        auto ans = rhs.deep_copy();
        ans *= lhs;
        return ans;
    }

    template< Tensor Tsor >
    Tsor operator * ( Tsor const& lhs, typename Tsor::value_type const& rhs ) noexcept
    {
        return rhs * lhs;
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
        std::for_each( tsor.data(), tsor.data()+tsor.size(), [lower, upper]( typename Tsor::value_type& v ){ v = std::min( upper, v ); v = std::max( lower, v ); }  );
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
            ans.shape_.resize( 1 );
            return ans;
        }

        std::vector<unsigned long> new_shape;
        std::copy_if( ans.shape_.begin(), ans.shape_.end(), std::back_inserter( new_shape ), []( unsigned long n ) { return n != 1UL; } );
        ans.shape_.swap( new_shape );
        return  ans;
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
    Tsor deep_copy( Tsor const& tsor )
    {
        return tsor.deep_copy();
    }

    template< Tensor Tsor >
    Tsor copy( Tsor const& tsor )
    {
        return deep_copy( tsor );
    }

    template< Tensor Tsor >
    Tsor concatenate( Tsor const& lhs, Tsor const& rhs, unsigned long axis=0 ) noexcept
    {
        auto l_shape = lhs.shape();
        auto r_shape = rhs.shape();
        better_assert( (l_shape.size() == r_shape.size()), "dimension not match, lhs dim is ", l_shape.size(), ", but rhs dim is ", r_shape.size() );
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

    template< Tensor Tsor >
    constexpr bool empty(  Tsor const& tsor ) noexcept
    {
        return tsor.size() == 0;
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
        debug_print( "calling softmax(tsor) with tensor id ", tsor.id_ );
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
        debug_print( "softmax(tsor) returns answer tensor with id ", ans.id_ );
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
        //std::cerr << "variance input got:\n" << ts << std::endl;
        Tsor x = mean( ts, axis, true );
        //std::cerr << "variance got:\n" << x << std::endl;
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
    tensor<T,A> loadtxt( std::string const& file_name )
    {
        std::ifstream ifs( file_name );
        better_assert( ifs.good(), "tensor::loadtxt: failed to open file ", file_name );

        // read the first line to extract shape
        std::vector<unsigned long> shape;
        {
            std::string s_shape;
            std::getline( ifs, s_shape );
            std::stringstream ss( s_shape );
            std::copy( std::istream_iterator<unsigned long>( ss ), std::istream_iterator<unsigned long>(), std::back_inserter( shape ) );
        }

        // read the data
        std::vector< T > buff;
        {
            std::stringstream iss;
            std::copy( std::istreambuf_iterator< char >( ifs ), std::istreambuf_iterator< char >(), std::ostreambuf_iterator< char >( iss ) );
            std::copy( std::istream_iterator< T >( iss ), std::istream_iterator< T >(), std::back_inserter( buff ) );
        }


        // copy and return
        tensor<T,A> ans{ shape };
        {
            better_assert( ans.size() == buff.size(), "tensor::loadtxt: shape suggests size of ", ans.size(), " but got ", buff.size() );
            std::copy( buff.begin(), buff.end(), ans.begin() );
        }
        return ans;
    }

    template< Tensor Tsor >
    void savetxt( std::string const& file_name, Tsor const& tsor )
    {
        std::ofstream ofs{ file_name };
        {//write shape
            auto const& shape = tsor.shape();
            std::copy( shape.begin(), shape.end(), std::ostream_iterator<unsigned long>{ ofs, " " } );
            ofs << "\n";
        }
        {//write data
            std::copy( tsor.begin(), tsor.end(), std::ostream_iterator<typename Tsor::value_type>{ ofs, " " } );
        }
        ofs.close();
    }

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
    struct view_4d
    {
        T* data_;
        unsigned long batch_size_;
        unsigned long row_;
        unsigned long col_;
        unsigned long channel_;

        constexpr view_4d( T* data, unsigned long batch_size, unsigned long row, unsigned long col, unsigned long channel ) noexcept : data_{data}, batch_size_{batch_size}, row_{row}, col_{col}, channel_{channel} {}

        constexpr auto operator[]( unsigned long index ) noexcept
        {
            return view_3d{ data_+index*row_*col_*channel_, row_, col_, channel_ };
        }

        constexpr auto operator[]( unsigned long index ) const noexcept
        {
            return view_3d{ data_+index*row_*col_*channel_, row_, col_, channel_ };
        }
    };

}//namespace ceras

#endif//HQKGLAXWWVFBFHQNHBVTQJKGUFTPCQPTPXDVNOSBDJIBHITCEKDISJYNAMCPLJDURURDAISFV

