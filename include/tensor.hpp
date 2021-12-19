#ifndef HQKGLAXWWVFBFHQNHBVTQJKGUFTPCQPTPXDVNOSBDJIBHITCEKDISJYNAMCPLJDURURDAISFV
#define HQKGLAXWWVFBFHQNHBVTQJKGUFTPCQPTPXDVNOSBDJIBHITCEKDISJYNAMCPLJDURURDAISFV

#include "./backend/cblas.hpp"
#include "./backend/cuda.hpp"
#include "./config.hpp"
#include "./includes.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/buffered_allocator.hpp"
#include "./utils/debug.hpp"
#include "./utils/fmt.hpp"
#include "./utils/for_each.hpp"
#include "./utils/id.hpp"
#include "./utils/range.hpp"
#include "./utils/stride_iterator.hpp"
#include "./utils/view.hpp"

namespace ceras
{
    ///
    /// @brief Random seed for the tensor library.
    ///
    /// To reproduce the result involving random variates such as `rand`, `normal`, `poisson`, it is necessary to fix the random seed by
    /// \code{.cpp}
    /// random_seed=42;
    /// \endcode
    ///
    static unsigned long random_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // static random number random_generator
    static std::mt19937 random_generator{random_seed};

    template< typename T >
    using default_allocator = buffered_allocator<T, 256>;
    //using default_allocator = std::allocator<T>;


    template< typename T, typename Allocator = default_allocator<T> >
    struct tensor : enable_id<tensor<T, Allocator>, "Tensor">
    {
        typedef T value_type;
        typedef Allocator allocator;
        typedef std::vector<T, Allocator> vector_type;
        typedef std::shared_ptr<vector_type> shared_vector;
        typedef tensor self_type;

        std::vector<unsigned long> shape_;
        unsigned long memory_offset_;
        shared_vector vector_;

        ///
        /// @breif Construct an empty vector
        ///
        tensor() : shape_{std::vector<unsigned long>{}}, memory_offset_{0}, vector_{std::make_shared<vector_type>()} { }

        ///
        /// @brief Construct a vector with the specified shape, initialized value and a (default) allocator.
        ///
        constexpr tensor( std::vector<unsigned long> const& shape, std::initializer_list<T> init, const Allocator& alloc = Allocator() ) : shape_{shape}, memory_offset_{0}, vector_{std::make_shared<vector_type>(init, alloc)}
        {
            better_assert( (*vector_).size() == std::accumulate( shape_.begin(), shape_.end(), 1UL, [](auto x, auto y){ return x*y; } ), "Expecting vector has same size as the shape indicates." );
        }

        ///
        /// @brief Construct a vector with the specified shape. All values initialized to default. With a default constructed allocator
        ///
        constexpr tensor( std::vector<unsigned long> const& shape ) : shape_{shape}, memory_offset_{0},
                                                                      vector_{std::make_shared<vector_type>(std::accumulate(shape_.begin(), shape_.end(), 1UL, [](auto x, auto y){return x*y;} ), T{0})}{}

        ///
        /// @brief Construct a vector with the specified shape and all values initialized to `init`. With a default constructed allocator
        ///
        constexpr tensor( std::vector<unsigned long> const& shape, T init ) : shape_{shape}, memory_offset_{0},
                                                                              vector_{std::make_shared<vector_type>(std::accumulate(shape_.begin(), shape_.end(), 1UL, [](auto x, auto y){return x*y;}), T{0})}
        {
            std::fill( begin(), end(), init );
        }

        constexpr tensor( tensor const& other, unsigned long memory_offset ) noexcept : shape_{ other.shape_ }, memory_offset_{ memory_offset }, vector_{ other.vector_ } {}

        ///
        /// @brief Copy-ctor.
        ///
        constexpr tensor( self_type const& other ) noexcept : shape_{ other.shape_ }, memory_offset_{ other.memory_offset_ }
        {
            vector_ = other.vector_;
            (*this).id_ = other.id_;
        }

        ///
        /// @brief Move-ctor.
        ///
        constexpr tensor( self_type && other ) noexcept : shape_{ other.shape_ }, memory_offset_{ other.memory_offset_ }
        {
            vector_ = other.vector_;
            (*this).id_ = other.id_;
        }

        ///
        /// @brief Copy-assignment.
        ///
        constexpr self_type& operator = ( self_type const& other ) noexcept
        {
            shape_ = other.shape_;
            memory_offset_ = other.memory_offset_;
            vector_ = other.vector_;
            (*this).id_ = other.id_;
            return *this;
        }

        ///
        /// @brief Move-assignment.
        ///
        constexpr self_type& operator = ( self_type && other ) noexcept
        {
            shape_ = other.shape_;
            memory_offset_ = other.memory_offset_;
            vector_ = other.vector_;
            (*this).id_ = other.id_;
            return *this;
        }

        ///
        /// @brief Iterator to the first element of the tensor.
        ///
        constexpr auto begin() noexcept
        {
            return data();
        }

        ///
        /// @brief Iterator to the first element of the tensor.
        ///
        constexpr auto begin() const noexcept
        {
            return data();
        }

        ///
        /// @brief Iterator to the first element of the tensor.
        ///
        constexpr auto cbegin() const noexcept
        {
            return begin();
        }

        ///
        /// @brief Iterator to the element following the last element of the tensor.
        ///
        constexpr auto end() noexcept
        {
            return begin() + size();
        }

        ///
        /// @brief Iterator to the element following the last element of the tensor.
        ///
        constexpr auto end() const noexcept
        {
            return begin() + size();
        }

        ///
        /// @brief Iterator to the element following the last element of the tensor.
        ///
        constexpr auto cend() const noexcept
        {
            return  end();
        }


        ///
        /// @brief Reverse iterator to the first element of the tensor.
        ///
        constexpr auto rbegin() noexcept
        {
            return make_reverse_iterator( end() );
        }

        ///
        /// @brief Reverse iterator to the first element of the tensor.
        ///
        constexpr auto rbegin() const noexcept
        {
            return make_reverse_iterator( end() );
        }

        ///
        /// @brief Reverse iterator to the first element of the tensor.
        ///
        constexpr auto crbegin() const noexcept
        {
            return make_reverse_iterator( cend() );
        }

        ///
        /// @brief Reverse iterator to the element following the last element of the tensor.
        ///
        constexpr auto rend() noexcept
        {
            return make_reverse_iterator( begin() );
        }

        ///
        /// @brief Reverse iterator to the element following the last element of the tensor.
        ///
        constexpr auto rend() const noexcept
        {
            return make_reverse_iterator( begin() );
        }

        ///
        /// @brief Reverse iterator to the element following the last element of the tensor.
        ///
        constexpr auto crend() const noexcept
        {
            return make_reverse_iterator( cbegin() );
        }

        ///
        /// @breif Reference to the first element in the tensor.
        ///
        constexpr auto front()
        {
            return (*vector_).front();
        }

        ///
        /// @breif Reference to the first element in the tensor.
        ///
        constexpr auto front() const
        {
            return (*vector_).front();
        }


        ///
        /// @breif Reference to the last element in the tensor.
        ///
        constexpr auto back()
        {
            return (*vector_).back();
        }

        ///
        /// @breif Reference to the last element in the tensor.
        ///
        constexpr auto back() const
        {
            return (*vector_).back();
        }


        ///
        /// @brief Number of elements in the tensor.
        ///
        constexpr unsigned long size() const noexcept
        {
            if ( !vector_ ) return 0;
            return (*vector_ ).size() - memory_offset_;
            //return std::accumulate( shape_.begin(), shape_.end(), 1UL, [](unsigned long x, unsigned long y){return x*y;} );
        }


        ///
        /// @brief Check if the tensor has elements.
        ///
        [[nodiscard]] constexpr bool empty() const noexcept
        {
            return cbegin() == cend();
        }



        ///
        /// Resetting all elements in the tensor to a fixed value (default to 0), without change the shape.
        ///
        /// Example code:
        /// \code{.cpp}
        /// tensor<float> ts;
        /// ts.reset( 0.0f );
        /// \endcode
        ///
        constexpr self_type& reset( T val = T{0} )
        {
            std::fill_n( data(), size(), val );
            return *this;
        }

        ///
        /// @brief Dimension of the tensor
        ///
        constexpr unsigned long ndim() const noexcept
        {
            return shape_.size();
        }

        ///
        /// @brief Shape of the tensor.
        ///
        constexpr std::vector<unsigned long> const& shape() const noexcept
        {
            return shape_;
        }


        ///
        /// @brief A deep copy of the tensor.
        ///
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

        ///
        /// @brief Resize the tensor with a new shape.
        ///
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

        ///
        /// @brief Reshape tensor. -1 indicates the dimension needs recalculating.
        ///
        /// \code{.cpp}
        /// tensor<float> t{ {2, 3, 4} };
        /// auto t1 = t.reshape( {3, 8} );
        /// auto t2 = t.reshape( {1, 4, -1UL} );
        /// \endcode
        ///
        constexpr self_type& reshape( std::vector<unsigned long> const& new_shape )
        {
            std::vector<unsigned long> _new_shape = new_shape;
            if ( *(_new_shape.rbegin()) == static_cast<unsigned long>( -1 ) )
                *(_new_shape.rbegin()) = (*this).size() / std::accumulate( _new_shape.begin(), _new_shape.end()-1, 1Ul, []( unsigned long x, unsigned long y ){ return x*y; } );

            unsigned long const new_size = std::accumulate( _new_shape.begin(), _new_shape.end(), 1UL, [](auto x, auto y){ return x*y; } );
            if ( (*this).size() != new_size ) return resize( _new_shape );

            better_assert( (*this).size() == new_size, "reshape: expecting same size, but the original size is ", (*this).size(), ", and the new size is ", new_size );
            (*this).shape_ = _new_shape;
            return *this;
        }

        //mapping a smaller tensor on a larger one
        constexpr self_type& shrink_to( std::vector< unsigned long > const& new_shape )
        {
            unsigned long const new_size = std::accumulate( new_shape.begin(), new_shape.end(), 1UL, [](auto x, auto y){ return x*y; } );
            better_assert( (*this).size() >= new_size, "reshape: expecting smaller size, but the original size is ", (*this).size(), ", and the new size is ", new_size );
            (*this).shape_ = new_shape;
            return *this;
        }

        //adjust the memory offset
        constexpr self_type& creep_to( unsigned long new_memory_offset )
        {
            (*this).memory_offset_ = new_memory_offset;
            return *this;
        }

        ///
        /// @brief Returns pointer to the underlying array serving as element storage.
        ///
        /// The pointer is such that range [data(); data() + size()) is always a valid range,
        /// even if the container is empty (data() is not dereferenceable in that case).
        ///
        constexpr value_type* data() noexcept
        {
            return (*vector_).data() + memory_offset_;
        }

        ///
        /// @brief Returns pointer to the underlying array serving as element storage.
        ///
        /// The pointer is such that range [data(); data() + size()) is always a valid range,
        /// even if the container is empty (data() is not dereferenceable in that case).
        ///
        constexpr const value_type* data() const noexcept
        {
            return (*vector_).data() + memory_offset_;
        }

        ///
        /// @brief Applying element-wise operation on each element in the tensor.
        ///
        /// \code{.cpp}
        ///    tensor<double> x{...};
        ///    x.map( []( double v ){ return 1.0/v+1.0; } );
        /// \endcode
        ///
        template< typename Function >
        constexpr self_type& map( Function const& f )
        {
            for_each( (*this).data(), (*this).data()+(*this).size(), [&f]( auto& v ){ f(v); } );
            return *this;
        }

        constexpr self_type& operator += ( self_type const& other )
        {
            //better_assert( shape() == other.shape(), "Error with tensor::operator += : Shape mismatch! -- current shape is ", shape(), " and other tensor shape is ", other.shape() );
            better_assert( shape() == other.shape(), fmt::format("Error with tensor::operator += : Shape mismatch! This shape is {}, while other shape is {}.", shape(), other.shape() ) );
            std::transform( data(), data()+size(), other.data(), data(), []( auto x, auto y ){ return x+y; } );
            return *this;
        }

        constexpr self_type& operator += ( value_type x )
        {
            for_each( data(), data()+size(), [x]( value_type& v ){ v += x; } );
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
            for_each( data(), data()+size(), [x]( auto& v ){ v -= x; } );
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
            for_each( data(), data()+size(), [x]( auto& v ){ v *= x; } );
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
            for_each( data(), data()+size(), [x]( auto& v ){ v /= x; } );
            return *this;
        }

        constexpr self_type const operator - () const
        {
            self_type ans = (*this).deep_copy();
            for_each( ans.data(), ans.data()+size(), []( auto& v ){ v = -v; } );
            return  ans;
        }

        constexpr value_type as_scalar() const noexcept
        {
            better_assert( size() == 1, "Expecting tensor has a single value, but got ", size() );
            return *begin();
        }

        template< typename U >
        constexpr auto as_type() const noexcept
        {
            tensor<U, typename std::allocator_traits<Allocator>::rebind_alloc<U>> ans{ (*this).shape() };
            std::copy( (*this).begin(), (*this).end(), ans.begin() );
            return ans;
        }

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


}//namespace ceras

#include "./tensor.tcc"

#endif//HQKGLAXWWVFBFHQNHBVTQJKGUFTPCQPTPXDVNOSBDJIBHITCEKDISJYNAMCPLJDURURDAISFV

