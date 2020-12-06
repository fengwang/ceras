#include <algorithm>
#include <chrono>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstdint>
#include <fstream>
#include <functional>
//#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

constexpr double eps = 1.0e-8;

static unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
static std::mt19937 generator{seed};

template< typename T, typename Allocator = std::allocator<T>> requires std::floating_point<T>
struct tensor
{
    typedef T value_type;
    typedef Allocator allocator;
    typedef std::vector<T, Allocator> vector_type;
    typedef std::shared_ptr<vector_type> shared_vector;
    typedef tensor self_type;

    std::vector<std::size_t> shape_;
    shared_vector vector_;

    constexpr auto begin() noexcept { return data(); }

    constexpr auto begin() const noexcept { return data(); }

    constexpr auto end() noexcept { return begin() + size(); }

    constexpr auto end() const noexcept { return begin() + size(); }

    constexpr std::size_t ndim() const noexcept { return shape_.size(); }

    constexpr self_type const deep_copy() const
    {
        self_type ans{ shape_ };
        std::copy_n( data(), size(), ans.data() );
        return ans;
    }

    //constexpr self_type const copy() const { return deep_copy(); }

    constexpr value_type& operator[]( std::size_t idx ) { return *( data() + idx ); }

    constexpr value_type const& operator[]( std::size_t idx ) const { return *( data() + idx ); }

    tensor() {}

    //constexpr tensor( std::vector<std::size_t> const& shape, std::initializer_list<T> init, const Allocator& alloc = allocator() ) : shape_{ shape }, vector_{std::make_shared<vector_type>( init, alloc )} {}
    constexpr tensor( std::vector<std::size_t> const& shape, std::vector<T> init, const Allocator& alloc = allocator() ) : shape_{ shape }, vector_{std::make_shared<vector_type>( init, alloc )} {}

    constexpr tensor( std::vector<std::size_t> const& shape ): shape_{shape}, vector_{ std::make_shared<vector_type>( std::accumulate( shape_.begin(), shape_.end(), 1UL, []( auto x, auto y ) { return x * y; } ), T{0} ) } {}

    //constexpr tensor( std::initializer_list<std::size_t> shape ) : tensor{ std::vector<std::size_t>{shape} } {}

    constexpr tensor( value_type const& v ) : shape_{ std::vector<std::size_t>{1UL, 1UL} }, vector_{ std::make_shared<vector_type>( 1Ul, v ) } {}

    constexpr tensor( self_type const& ) = default;
    constexpr tensor( self_type&& ) noexcept = default;
    constexpr self_type& operator = ( self_type const& ) = default;
    constexpr self_type& operator = ( self_type&& ) noexcept = default;

    constexpr self_type& operator = ( T const& v ) noexcept { std::fill( begin(), end(), v ); return *this;}

    constexpr std::vector< std::size_t > const& shape() const noexcept { return shape_; }

    constexpr std::size_t size() const noexcept { return ( *vector_ ).size(); }

    constexpr self_type& reshape( std::vector< std::size_t > const& new_shape ) { ( *this ).shape_ = new_shape; return *this; }

    //constexpr self_type& reshape( std::initializer_list< std::size_t > new_shape ) { return reshape( std::vector< std::size_t > {new_shape} ); }

    [[nodiscard]] constexpr bool empty() const noexcept { return 0 == shape_.size(); }

    constexpr value_type* data() noexcept { return ( *vector_ ).data(); }

    constexpr const value_type* data() const noexcept { return ( *vector_ ).data(); }

    template< typename Function >
    constexpr self_type& map( Function const& f )
    {
        std::for_each( ( *this ).data(), ( *this ).data() + ( *this ).size(), [&f]( auto & v ) { f( v ); } );
        return *this;
    }

    constexpr self_type& operator += ( self_type const& other )
    {
        std::transform( data(), data() + size(), other.data(), data(), []( auto x, auto y ) { return x + y; } );
        return *this;
    }

    constexpr self_type& operator += ( value_type x )
    {
        std::for_each( data(), data() + size(), [x]( value_type & v ) { v += x; } );
        return *this;
    }

    constexpr self_type& operator -= ( self_type const& other )
    {
        std::transform( data(), data() + size(), other.data(), data(), []( auto x, auto y ) { return x - y; } );
        return *this;
    }

    constexpr self_type& operator -= ( value_type x )
    {
        std::for_each( data(), data() + size(), [x]( auto & v ) { v -= x; } );
        return *this;
    }

    constexpr self_type& operator *= ( value_type x )
    {
        std::for_each( data(), data() + size(), [x]( auto & v ) { v *= x; } );
        return *this;
    }

    constexpr self_type& operator /= ( value_type x )
    {
        std::for_each( data(), data() + size(), [x]( auto & v ) { v /= x; } );
        return *this;
    }

    constexpr self_type const operator - () const
    {
        self_type ans = ( *this ).deep_copy();
        std::for_each( ans.data(), ans.data() + size(), []( auto & v ) { v = -v; } );
        return ans;
    }
};

template< typename T > struct is_tensor : std::false_type {};

template< typename T, typename A > struct is_tensor<tensor< T, A>> : std::true_type {};

template< class T > inline constexpr bool is_tensor_v = is_tensor<T>::value;

template< typename T > concept Tensor = is_tensor_v<T>;

template< typename T > requires std::floating_point<T>
struct view_2d
{
    T* data_;
    std::size_t row_;
    std::size_t col_;
    bool transposed_;

    template<typename A>
    constexpr view_2d( tensor<T, A>& tsor, std::size_t row, std::size_t col, bool transposed = false ) noexcept : data_{tsor.data()}, row_{row}, col_{col}, transposed_{transposed} {}

    constexpr view_2d( T* data, std::size_t row, std::size_t col, bool transposed = false ) noexcept : data_{data}, row_{row}, col_{col}, transposed_{transposed} {}
    constexpr view_2d( const T* data, std::size_t row, std::size_t col, bool transposed = false ) noexcept : data_{const_cast<T*>( data )}, row_{row}, col_{col}, transposed_{transposed} {}

    constexpr T* operator[]( std::size_t index )
    {
        if ( transposed_ ) return data_ + index * row_;
        return data_ + index * col_;
    }
    constexpr const T* operator[]( std::size_t index ) const
    {
        if ( transposed_ ) return data_ + index * row_;
        return data_ + index * col_;
    }
    constexpr auto shape() const noexcept
    {
        return std::make_pair( row_, col_ );
    }
    constexpr std::size_t size() const noexcept
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

template< typename T > requires std::floating_point<T>
void gemm( T const* A, bool a_transposed, T const* B, bool b_transposed, std::size_t m, std::size_t n, std::size_t k, T* C )
{
    static_assert( std::is_floating_point_v<T>, "T is not a floating point type." );
    auto a_view = view_2d{ A, m, n, a_transposed };
    auto b_view = view_2d{ B, n, k, b_transposed };
    auto c_view = view_2d{ C, m, k };
    std::fill_n( C, m * k, T{0} );

    for ( auto r = 0UL; r != m; ++r )
        for ( auto c = 0UL; c != k; ++c )
            for ( auto idx = 0UL; idx != n; ++idx )
            {
                if ( a_transposed == false && b_transposed == false )
                    c_view[r][c] += a_view[r][idx] * b_view[idx][c];

                else if ( a_transposed == false && b_transposed == true )
                    c_view[r][c] += a_view[r][idx] * b_view[c][idx];

                else if ( a_transposed == true && b_transposed == false )
                    c_view[r][c] += a_view[idx][r] * b_view[idx][c];

                else
                    c_view[r][c] += a_view[idx][r] * b_view[c][idx];
            }
}

template< typename T > requires std::floating_point<T>
void gemm( view_2d<T> const& x, view_2d<T> const& y, view_2d<T>& ans )
{
    auto const [x_row, x_col] = x.shape();
    auto const [y_row, y_col] = y.shape();
    gemm( x.data(), false, y.data(), false, x_row, x_col, y_col, ans.data() );
}

template< Tensor Tsor >
Tsor reshape( Tsor const& tsor, std::vector<std::size_t> const& shape )
{
    Tsor ans{ tsor };
    return ans.reshape( shape );
}

// TODO: last dim(s) broadcasting only, works for this app only
template< Tensor Tsor >
Tsor add( Tsor const& lhs, Tsor const& rhs ) noexcept
{
    std::size_t const l_size = lhs.size();
    std::size_t const r_size = rhs.size();

    std::size_t const repeats = l_size / r_size;
    Tsor ans = lhs.deep_copy();

    for ( auto idx = 0UL; idx != repeats; ++idx )
        for ( auto jdx = 0UL; jdx != r_size; ++jdx )
            ans[idx * r_size + jdx] = lhs[idx * r_size + jdx] + rhs[jdx];

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
    ans.map( [lhs]( auto & v ) { v = lhs - v; } );
    return ans;
}
template< Tensor Tsor >
Tsor operator - ( Tsor const& lhs, typename Tsor::value_type const& rhs ) noexcept
{
    auto ans = lhs.deep_copy();
    ans.map( [rhs]( auto & v ) { v -= rhs; } );
    return ans;
}

template< Tensor Tsor >
Tsor multiply( Tsor const& lhs, Tsor const& rhs ) noexcept
{
    if ( 1 == lhs.ndim() )
        return multiply( reshape( lhs, {1UL, lhs.size()} ), rhs );

    if ( 1 == rhs.ndim() )
        return multiply( lhs, reshape( rhs, {lhs.size(), 1UL} ) );

    typedef typename Tsor::value_type value_type;
    auto const& lhs_shape = lhs.shape();
    auto const& rhs_shape = rhs.shape();

    view_2d<value_type> const x{ lhs.data(), lhs_shape[0], lhs_shape[1] };
    view_2d<value_type> const y{ rhs.data(), rhs_shape[0], rhs_shape[1] };
    auto const [row, col] = std::make_pair( lhs_shape[0], rhs_shape[1] );
    Tsor ans{ std::vector<std::size_t>{ {row, col} } };
    view_2d<value_type> z{ ans.data(), row, col };
    gemm( x, y, z );
    return ans;
}

template< Tensor Tsor >
Tsor operator * ( Tsor const& lhs, Tsor const& rhs ) noexcept
{
    return multiply( lhs, rhs );
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

template< typename T, typename A = std::allocator<T>>
tensor<T, A> ones( std::vector<std::size_t> shape )
{
    tensor<T, A> ans{ shape };
    std::fill_n( ans.data(), ans.size(), T{1} );
    return ans;
}

template< typename T, typename A = std::allocator<T>>
tensor<T, A> randn( std::vector<std::size_t> shape, T mean = T{0}, T stddev = T{1} )
{
    std::normal_distribution<T> distribution( mean, stddev );
    tensor<T, A> ans{ shape };
    std::generate( ans.data(), ans.data() + ans.size(), [&distribution]() { return distribution( generator ); } );
    return ans;
}

template< typename T, typename A = std::allocator<T>>
tensor<T, A> zeros( std::vector<std::size_t> const& shape )
{
    return {shape};
}

template< Tensor Tsor >
auto sum( Tsor const& tsor )
{
    typedef typename Tsor::value_type value_type;
    return std::accumulate( tsor.data(), tsor.data() + tsor.size(), value_type{0} );
}

/*
template< Tensor Tsor >
auto mean( Tsor const& tsor )
{
    if ( 0 == tsor.size() ) return typename Tsor::value_type{0};

    return sum( tsor ) / tsor.size();
}
*/

template< Tensor Tsor >
Tsor abs( Tsor const& tsor )
{
    auto ans = tsor.deep_copy();
    ans.map( []( auto & x )
    {
        x = std::abs( x );
    } );
    return ans;
}
template< Tensor Tsor >
Tsor softmax( Tsor const& tsor )
{
    typedef typename Tsor::value_type value_type;
    Tsor ans = tsor.deep_copy();
    std::size_t const last_dim = *( tsor.shape().rbegin() );
    std::size_t const rem_dim = tsor.size() / last_dim;
    view_2d<value_type> mat{ ans.data(), rem_dim, last_dim };

    for ( auto idx = 0UL; idx != rem_dim; ++idx )
    {
        value_type const mx = *std::max_element( mat[idx], mat[idx + 1] );
        std::for_each( mat[idx], mat[idx + 1], [mx]( auto & v ) { v -= mx; } );
        value_type const ac = std::accumulate( mat[idx], mat[idx + 1], value_type{0}, []( value_type init, value_type val ) { return init + std::exp( val ); } );
        std::for_each( mat[idx], mat[idx + 1], [ac]( auto & v ) { v = std::exp( v ) / ( ac + eps ); } );
    }

    return ans;
}

template< Tensor Tsor, typename Function >
Tsor reduce( Tsor const& ts, std::size_t axis, typename Tsor::value_type const& init, Function const& func, bool keepdims = false ) noexcept
{
    if ( ts.empty() ) return ts;

    axis = ( axis == static_cast<std::size_t>( -1 ) ) ? ts.ndim() - 1 : axis;
    std::vector<std::size_t> _shape = ts.shape();
    std::size_t const pres = std::reduce( _shape.begin(), _shape.begin() + axis, 1Ul, []( std::size_t x, std::size_t y ) { return x * y; } );
    std::size_t const post = std::reduce( _shape.begin() + axis + 1, _shape.end(), 1Ul, []( std::size_t x, std::size_t y ) { return x * y; } );
    std::size_t const n = _shape[axis];
    _shape[axis] = 1UL;
    Tsor ans{ _shape };
    auto itor = ans.begin();

    for ( auto idx = 0UL; idx != pres; ++idx )
        for ( auto jdx = 0UL; jdx != post; ++jdx )
        {
            auto start = ts.begin() + idx * post * n + jdx;
            auto val = init;
            for ( auto si = start; si != start+n*post; si += post )
            {
                val = func( val, *si );
            }
            *itor++ = val;
        }

    if ( !keepdims )
    {
        std::copy( _shape.begin() + axis + 1, _shape.end(), _shape.begin() + axis );
        _shape.resize( _shape.size() - 1 );
        ans.reshape( _shape );
    }

    return ans;
}

template <Tensor Tsor>
Tsor sum( Tsor const& ts, std::size_t axis, bool keepdims = false ) noexcept
{
    return reduce( ts, axis, typename Tsor::value_type{0}, []( auto const & a, auto const & b )
    {
        return a + b;
    }, keepdims );
}

template< typename T, typename A = std::allocator<T>>
struct place_holder
{
    std::shared_ptr<tensor<T, A>> data_;
    place_holder()
    {
    }
    ~place_holder()
    {
    }
    tensor<T, A> const forward() const
    {
        return *data_;
    }
    void bind( tensor<T, A> const& data )
    {
        data_ = std::make_shared<tensor<T, A>>( data );
    }
    void reset()
    {
        (*data_) = T{0};
    }
    void backward( auto ) const noexcept
    {
    }
};

template< typename T >
struct is_place_holder : std::false_type {};

template< typename T, typename A >
struct is_place_holder< place_holder< T, A>> : std::true_type {};

template< class T >
inline constexpr bool is_place_holder_v = is_place_holder<T>::value;

template< typename T >
concept Place_Holder = is_place_holder_v<T>;

namespace _private
{
    struct id
    {
        int value_;
        id( int value = 0 ) noexcept: value_ {value} {}
    };
};
inline int generate_uid() noexcept
{
    static _private::id id_generator;
    int ans = id_generator.value_;
    ++id_generator.value_;
    return ans;
}

template< typename T, typename A >
struct session;

template< typename T, typename A >
std::reference_wrapper<session<T, A>> get_default_session();

template< typename T, typename A = std::allocator<T>>
struct variable
{
    int id_;
    std::shared_ptr<tensor<T, A>> data_;
    std::shared_ptr<tensor<T, A>> gradient_;
    std::shared_ptr<tensor<T, A>> old_gradient_;
    variable( tensor<T, A> const& data ) :
        id_{ generate_uid() },
        data_{ std::make_shared<tensor<T, A>>( data ) },
        gradient_{ std::make_shared<tensor<T, A>>( data.shape() ) },
        old_gradient_{std::make_shared<tensor<T, A>>( data.shape() )}
    {}
    variable() = delete;
    void backward( auto const& grad )
    {
        *gradient_ += grad;
        auto& ss = get_default_session<T, A>().get();
        ss.remember( *this );
    }
    tensor<T, A> const forward() const
    {
        std::swap( *gradient_, *old_gradient_ );
        ( *gradient_ ) = T{0};
        return *data_;
    }
};

template< typename T >
struct is_variable : std::false_type {};

template< typename T, typename A >
struct is_variable< variable< T, A>> : std::true_type {};

template< class T >
inline constexpr bool is_variable_v = is_variable<T>::value;

template< typename T >
concept Variable = is_variable_v<T>;

struct operator_type_wrapper
{
    template< typename T >
    T operator()( T const& t ) const noexcept
    {
        return t;
    };
    template< typename T >
    std::reference_wrapper<T> operator()( T& t ) const noexcept
    {
        return std::ref( t );
    };
    template< typename T, typename A >
    std::reference_wrapper<place_holder<T, A> const> operator()( place_holder<T, A> const& ph ) const noexcept
    {
        return std::cref( ph );
    }
};

struct forward_wrapper
{
    template< typename T >
    auto operator() ( T& t ) const noexcept
    {
        return t.forward();
    }
    template< typename T, typename A >
    auto operator() ( place_holder<T, A> const& ph ) const noexcept
    {
        return ph.forward();
    }
    template< typename T >
    auto operator() ( std::reference_wrapper<T const> t ) const noexcept
    {
        return t.get().forward();
    };
    template< typename T >
    auto operator() ( std::reference_wrapper<T> t ) const noexcept
    {
        return t.get().forward();
    };
};

struct backward_wrapper
{
    template< typename Op >
    auto operator() ( Op& op ) const noexcept
    {
        return [&op]<typename T, typename A>( tensor<T, A> const & grad )
        {
            op.backward( grad );
        };
    }
    template< typename T, typename A >
    auto operator() ( std::reference_wrapper<place_holder<T, A> const> ) const noexcept
    {
        return []( auto ) {};
    };
    template< typename T, typename A >
    auto operator() ( place_holder<T, A> ) const noexcept
    {
        return []( auto ) {};
    };
    template< typename Op >
    auto operator() ( std::reference_wrapper<Op> op ) noexcept
    {
        return [op]<typename T, typename A>( tensor<T, A> const & grad )
        {
            op.get().backward( grad );
        };
    }
};

template< typename Operator, typename Forward_Action, typename Backward_Action >
struct unary_operator
{
    decltype( operator_type_wrapper{}( std::declval<Operator>() ) ) op_;
    Forward_Action forward_action_;
    Backward_Action backward_action_;
    typedef decltype( std::declval<Forward_Action>()( std::declval < decltype( forward_wrapper{}( op_ ) ) > () ) ) tensor_type;
    tensor_type input_data_;
    tensor_type output_data_;
    unary_operator( Operator const& op, Forward_Action const& forward_action, Backward_Action const& backward_action ) noexcept :
        op_{operator_type_wrapper{}( op )}, forward_action_{ forward_action }, backward_action_{ backward_action } {}
    auto forward()
    {
        input_data_ = forward_wrapper{}( op_ );
        output_data_ = forward_action_( input_data_ );
        return output_data_;
    }
    template< typename T, typename A >
    void backward( tensor<T, A> const& grad )
    {
        auto const& current_gradient = backward_action_( input_data_, output_data_, grad );
        backward_wrapper{}( op_ )( current_gradient );
    }
};
static auto constexpr make_unary_operator = []( auto const& unary_forward_action, auto const& unary_backward_action ) noexcept
{
    return [&unary_forward_action, &unary_backward_action]( auto const & op ) noexcept
    {
        return unary_operator{ op, unary_forward_action, unary_backward_action };
    };
};
template< typename Lhs_Operator, typename Rhs_Operator, typename Forward_Action, typename Backward_Action >
struct binary_operator
{
    decltype( operator_type_wrapper{}( std::declval<Lhs_Operator>() ) ) lhs_op_;
    decltype( operator_type_wrapper{}( std::declval<Rhs_Operator>() ) ) rhs_op_;
    Forward_Action const& forward_action_;
    Backward_Action const& backward_action_;
    typedef decltype( std::declval<Forward_Action>()( std::declval < decltype( forward_wrapper{}( lhs_op_ ) ) > (), std::declval < decltype( forward_wrapper{}( rhs_op_ ) ) > () ) ) tensor_type;
    tensor_type lhs_input_data_;
    tensor_type rhs_input_data_;
    tensor_type output_data_;
    binary_operator( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op, Forward_Action const& forward_action, Backward_Action const& backward_action ) noexcept :
        lhs_op_{operator_type_wrapper{}( lhs_op )}, rhs_op_{operator_type_wrapper{}( rhs_op )}, forward_action_{ forward_action }, backward_action_{ backward_action } {}
    auto forward()
    {
        lhs_input_data_ = forward_wrapper{}( lhs_op_ );
        rhs_input_data_ = forward_wrapper{}( rhs_op_ );
        output_data_ = forward_action_( lhs_input_data_, rhs_input_data_ );
        return output_data_;
    }
    template< typename T, typename A >
    void backward( tensor<T, A> const& grad )
    {
        auto const& [current_gradient_lhs, current_gradient_rhs] = backward_action_( lhs_input_data_, rhs_input_data_, output_data_, grad );
        backward_wrapper{}( lhs_op_ )( current_gradient_lhs );
        backward_wrapper{}( rhs_op_ )( current_gradient_rhs );
    }
};
static auto constexpr make_binary_operator = []( auto const& binary_forward_action, auto const& binary_backward_action ) noexcept
{
    return [&binary_forward_action, &binary_backward_action]( auto const & lhs_op, auto const & rhs_op ) noexcept
    {
        return binary_operator{ lhs_op, rhs_op, binary_forward_action, binary_backward_action };
    };
};

template< typename T > struct is_operator : std::false_type {};

template< typename Lhs_Operator, typename Rhs_Operator, typename Forward_Action, typename Backward_Action >
struct is_operator< binary_operator<Lhs_Operator, Rhs_Operator, Forward_Action, Backward_Action>> : std::true_type {};

template< typename Operator, typename Forward_Action, typename Backward_Action >
struct is_operator< unary_operator<Operator, Forward_Action, Backward_Action>> : std::true_type {};

template< class T >
inline constexpr bool is_operator_v = is_operator<T>::value;

template< typename T >
concept Operator = is_operator_v<T>;

template< typename T >
concept Operation = Operator<T> || Variable<T> || Place_Holder<T>;

template< typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator>&& Operation<Rhs_Operator>
auto constexpr plus( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
{
    return make_binary_operator
    (
        []<Tensor Tsor>( Tsor const & lhs_tensor, Tsor const & rhs_tensor ) noexcept
        {
            return add( lhs_tensor, rhs_tensor );
        },
        []<Tensor Tsor>( Tsor const & lhs_input, Tsor const & rhs_input, Tsor const&, Tsor const grad ) noexcept
        {
            auto const& grad_fun = [&grad]( Tsor const & input )
            {
                Tsor ans = grad.deep_copy();

                while( input.ndim() < ans.ndim() )
                    ans = sum( ans, 0 );

                auto const& shape = input.shape();

                for ( auto axis = 0UL; axis != input.ndim(); ++axis )
                    if ( shape[axis] == 1 )
                        ans = sum( ans, axis, true );

                return ans;
            };
            return std::make_tuple( grad_fun( lhs_input ), grad_fun( rhs_input ) );
        }
    )( lhs_op, rhs_op );
}
template< typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator>&& Operation<Rhs_Operator>
auto constexpr operator + ( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
{
    return plus( lhs_op, rhs_op );
}
template< typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator>&& Operation<Rhs_Operator>
auto constexpr operator * ( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
{
    return make_binary_operator
    (
        []<Tensor Tsor>( Tsor const & lhs_tensor, Tsor const & rhs_tensor ) noexcept
        {
            return multiply( lhs_tensor, rhs_tensor );
        },
        []<Tensor Tsor>( Tsor const & lhs_input, Tsor const & rhs_input, Tsor const&, Tsor const grad ) noexcept
        {
            auto const& g_shape = grad.shape();
            auto const[m, n] = std::make_tuple( g_shape[0], g_shape[1] );
            auto const k = *( lhs_input.shape().rbegin() );
            Tsor lhs_grad{ lhs_input.shape() };
            gemm( grad.data(), false, rhs_input.data(), true, m, n, k, lhs_grad.data() );
            Tsor rhs_grad{ rhs_input.shape() };
            gemm( lhs_input.data(), true, grad.data(), false, k, m, n, rhs_grad.data() );
            return std::make_tuple( lhs_grad, rhs_grad );
        }
    )( lhs_op, rhs_op );
}

template <typename Op> requires Operation<Op>
auto constexpr relu( Op const& op ) noexcept
{
    return make_unary_operator
    (
        []<Tensor Tsor>( Tsor const & input ) noexcept
        {
            Tsor ans{ input.shape() };
            for ( auto idx = 0UL; idx != ans.size(); ++idx )
                ans[idx] = std::max( input[idx], typename Tsor::value_type{0} );
            return ans;
        },
        []<Tensor Tsor>( Tsor const & input, Tsor const&, Tsor const & grad ) noexcept
        {
            typedef typename Tsor::value_type value_type;
            Tsor ans = grad;
            for ( auto idx = 0UL; idx != ans.size(); ++idx )
                ans[idx] = ( input[idx] > value_type{0} ) ? grad[idx] : value_type{0};
            return ans;
        }
    )( op );
}

template < typename Lhs_Operator, typename Rhs_Operator > requires Operation<Lhs_Operator>&& Operation<Rhs_Operator>
auto constexpr cross_entropy_loss( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op ) noexcept
{
    return make_binary_operator
    (
        []<Tensor Tsor>( Tsor const & ground_truth_input, Tsor const & prediction_input ) noexcept
        {
            typedef typename Tsor::value_type value_type;
            auto const& sm = softmax( prediction_input );
            value_type ans{0};

            for ( auto idx = 0UL; idx != ground_truth_input.size(); ++idx )
                ans -= ground_truth_input[idx] * std::log( std::max( static_cast<value_type>( eps ), sm[idx] ) );

            ans /= *( ground_truth_input.shape().begin() );
            return Tsor(ans);
        },
        []<Tensor Tsor>( Tsor const & ground_truth_input, Tsor const & prediction_input, Tsor const&, Tsor const& ) noexcept
        {
            Tsor ground_truth_gradient = ground_truth_input;
            Tsor sm = softmax( prediction_input ) - ground_truth_input;
            return make_tuple( ground_truth_gradient, sm );
        }
    )( lhs_op, rhs_op );
}

template< typename T >
struct singleton
{
        typedef T value_type;
        typedef singleton self_type;
    private:
        singleton( const self_type& );
        self_type& operator = ( const self_type& );
        singleton();
    private:
        struct constuctor
        {
            constuctor()
            {
                self_type::instance();
            }
            inline void null_action() const { }
        };
        static constuctor constuctor_;
    public:
        static value_type&
        instance()
        {
            static value_type instance_;
            constuctor_.null_action();
            return instance_;
        }
};
template<typename T>
typename singleton<T>::constuctor singleton<T>::constuctor_;

template< typename T, typename A = std::allocator<T>>
struct session
{
    typedef tensor<T, A> tensor_type;
    typedef place_holder<T, A> place_holder_type;
    typedef variable<T, A> variable_type;

    std::vector<std::reference_wrapper<place_holder_type>> place_holders_;
    std::map<int, std::reference_wrapper<variable_type>> variables_;

    session()
    {
        singleton<session<T, A>*>::instance() = this;
    }
    session( session const& ) = delete;
    session( session&& ) = delete;
    session& operator=( session const& ) = delete;
    session& operator=( session&& ) = delete;

    void rebind( place_holder_type& p_holder, tensor_type const& value ) { p_holder.bind( value ); }

    void bind( place_holder_type& p_holder, tensor_type const& value )
    {
        p_holder.bind( value );
        place_holders_.emplace_back( std::ref( p_holder ) );
    }

    void remember( variable_type& v ) { variables_.insert( {v.id_, std::ref( v )} ); }

    template< typename Operation >
    auto run( Operation& op ) const { return op.forward(); }

    ~session()
    {
        for ( auto& p_holder : place_holders_ ) p_holder.get().reset();

        place_holders_.clear();
        variables_.clear();
        singleton<session<T, A>*>::instance() = nullptr;
    }
};
template< typename T, typename A >
std::reference_wrapper<session<T, A>> get_default_session()
{
    auto p_session = singleton<session<T, A>*>::instance();
    return std::ref( *p_session );
}

template< typename Loss, typename T >
struct gradient_descent
{
    Loss& loss_;
    T learning_rate_;
    T momentum_;
    gradient_descent( Loss& loss, std::size_t batch_size, T learning_rate = 1.0e-3, T momentum = 0.0 ) noexcept : loss_( loss ), learning_rate_( learning_rate ), momentum_( momentum )
    {
        learning_rate_ /= static_cast<T>( batch_size );
    }
    void forward()
    {
        loss_.backward( ones<T>( {1,} ) );
        auto& ss = get_default_session<T, std::allocator<T>>().get();

        for ( auto [id, v] : ss.variables_ )
        {
            *( v.get().data_ ) -= learning_rate_ * ( *( v.get().gradient_ ) ) * ( 1.0 - momentum_ );
            *( v.get().data_ ) -= learning_rate_ * ( *( v.get().old_gradient_ ) ) * momentum_;
        }
    }
};

std::string const training_image_path{ "./dataset/mnist/train-images-idx3-ubyte" };
std::string const training_label_path{ "./dataset/mnist/train-labels-idx1-ubyte" };
std::string const testing_image_path{ "./dataset/mnist/t10k-images-idx3-ubyte" };
std::string const testing_label_path{ "./dataset/mnist/t10k-labels-idx1-ubyte" };

std::vector<std::uint8_t> load_binary( std::string const& filename )
{
    std::ifstream ifs( filename, std::ios::binary );
    std::vector<char> buff{ ( std::istreambuf_iterator<char>( ifs ) ), ( std::istreambuf_iterator<char>() ) };
    std::vector<std::uint8_t> ans( buff.size() );
    std::copy( buff.begin(), buff.end(), reinterpret_cast<char*>( ans.data() ) );
    return ans;
}

int main()
{
    std::vector<std::uint8_t> training_images = load_binary( training_image_path );
    std::vector<std::uint8_t> training_labels = load_binary( training_label_path );

    auto input = place_holder<double> {};
    auto w1 = variable<double> { randn<double>( {28 * 28, 256}, 0.0, 1.0 / std::sqrt(1.0*28*28*256) ) };
    auto b1 = variable<double> { zeros<double>( { 256, } ) };
    auto l1 = relu( input * w1 + b1 );
    auto w2 = variable<double> { randn<double>( {256, 128}, 0.0, 1.0 / std::sqrt(1.0*256*128) ) };
    auto b2 = variable<double> { zeros<double>( { 128, } ) };
    auto l2 = relu( l1 * w2 + b2 );
    auto w3 = variable<double> { randn<double>( {128, 10}, 0.0, 1.0 / std::sqrt(1.0*128*10) ) };
    auto b3 = variable<double> { zeros<double>( { 10, } ) };
    auto output = l2 * w3 + b3;
    auto ground_truth = place_holder<double> {};
    auto loss = cross_entropy_loss( ground_truth, output );

    std::size_t const batch_size = 10;
    tensor<double> input_images{ {batch_size, 28 * 28} };
    tensor<double> output_labels{ {batch_size, 10} };
    std::size_t const epoch = 2;
    std::size_t const iteration_per_epoch = 60000 / batch_size;
    session<double> s;
    s.bind( input, input_images );
    s.bind( ground_truth, output_labels );
    double learning_rate = 1.0e-1f;
    auto optimizer = gradient_descent{ loss, batch_size, learning_rate };

    for ( auto e = 0UL; e != epoch; ++e )
    {
        for ( auto i = 0UL; i != iteration_per_epoch; ++i )
        {
            std::size_t const image_offset = 16 + i * batch_size * 28 * 28;

            for ( auto j = 0UL; j != batch_size*28*28; ++j )
                input_images[j] = static_cast<double>( training_images[j + image_offset] ) / 127.5f - 1.0f;

            std::size_t const label_offset = 8 + i * batch_size * 1;
            std::fill_n( output_labels.data(), output_labels.size(), 0.0f );

            for ( auto j = 0UL; j != batch_size * 1; ++ j )
            {
                std::size_t const label = static_cast<std::size_t>( training_labels[j + label_offset] );
                output_labels[j * 10 + label] = 1.0f;
            }

            auto current_error = s.run( loss );
            std::cout << "Loss at epoch " << e << " index: " << ( i + 1 )*batch_size << ":\t" << current_error[0] << "\r" << std::flush;

            s.run( optimizer );
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    unsigned long const new_batch_size = 1;

    std::vector<std::uint8_t> testing_images = load_binary( testing_image_path );
    std::vector<std::uint8_t> testing_labels = load_binary( testing_label_path );
    std::size_t const testing_iterations = 10000 / new_batch_size;

    tensor<double> new_input_images{ {new_batch_size, 28 * 28} };
    s.bind( input, new_input_images );

    unsigned long errors = 0;

    for ( auto i = 0UL; i != testing_iterations; ++i )
    {
        std::size_t const image_offset = 16 + i * new_batch_size * 28 * 28;

        for ( auto j = 0UL; j != new_batch_size*28*28; ++j )
            new_input_images[j] = static_cast<double>( testing_images[j + image_offset] ) / 127.5f - 1.0f;

        auto prediction = s.run( output );
        prediction.reshape( {prediction.size(), } );
        std::size_t const predicted_number = std::max_element( prediction.begin(), prediction.end() ) - prediction.begin();

        std::size_t const label_offset = 8 + i * new_batch_size * 1;
        std::size_t const ground_truth = testing_labels[label_offset];

        if ( predicted_number != ground_truth )
        {
            errors += 1;
            std::cout << "Prediction error at index " << i << ": predicted " << predicted_number << ", but the ground_truth is " << ground_truth << std::endl;
        }

    }

    double const err = 1.0 * errors / 10000;
    std::cout << "Prediction error on the testing set is " << err << std::endl;

    return 0;
}
