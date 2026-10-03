#ifndef HQKGLAXWWVFBFHQNHBVTQJKGUFTPCQPTPXDVNOSBDJIBHITCEKDISJYNAMCPLJDURURDAISFV
#define HQKGLAXWWVFBFHQNHBVTQJKGUFTPCQPTPXDVNOSBDJIBHITCEKDISJYNAMCPLJDURURDAISFV

#include "./backend/cblas.hpp"
#include "./backend/cuda.hpp"
#include "./config.hpp"
#include "./utils/checked_size.hpp"
#include "./includes.hpp"
#include "./utils/better_assert.hpp"
#include "./utils/cached_allocator.hpp"
#include "./utils/buffered_allocator.hpp"
#include "./utils/debug.hpp"
#include "./utils/fmt.hpp"
#include "./utils/for_each.hpp"
#include "./utils/id.hpp"
#include "./utils/range.hpp"
#include "./utils/stride_iterator.hpp"
#include "./utils/view.hpp"
#include "./utils/vector.hpp"
#include "./utils/type2string.hpp"

namespace ceras
{
    struct tensor_parse_limits {
        std::size_t max_rank=64;
        std::size_t max_elements=16*1024*1024;
        std::size_t max_bytes=256*1024*1024;
        std::size_t max_line_bytes=256*1024*1024;
    };
    inline thread_local tensor_parse_limits tensor_io_limits;

    ///
    /// @brief Random seed for the tensor library.
    ///
    /// To reproduce the result involving random variates such as `rand`, `normal`, `poisson`, it is necessary to fix the random seed by
    /// \code{.cpp}
    /// seed_random(42);
    /// \endcode
    ///
    inline thread_local unsigned long random_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // static random number random_generator
    inline thread_local std::mt19937 random_generator{random_seed};

    inline thread_local std::mt19937* active_random=nullptr;
    inline std::mt19937& random_engine() {return active_random ? *active_random : random_generator;}
    inline void seed_random(unsigned long seed) { random_seed=seed; random_engine().seed(seed); }

    template< typename T >
    using default_allocator = cached_allocator<T>;
    //using default_allocator = std::allocator<T>;


    template< typename T, typename Allocator = default_allocator<T> >
    struct tensor : enable_id<tensor<T, Allocator>, "Tensor">
    {
        typedef T value_type;
        typedef Allocator allocator;
        typedef vector<T, Allocator> vector_type;
        typedef std::shared_ptr<vector_type> shared_vector;
        typedef tensor self_type;

        // TODO: with buffered_allocator
        //std::vector<unsigned long> shape_;
        std::vector<unsigned long> shape_;
        shared_vector vector_;

        ///
        /// @breif Construct an empty vector
        ///
        tensor() : shape_{}, vector_{std::make_shared<vector_type>()} { }

        ///
        /// @brief Construct a vector with the specified shape, initialized value and a (default) allocator.
        ///
        template<typename Another_Alloc>
        constexpr tensor( std::vector<unsigned long, Another_Alloc> const& shape, std::initializer_list<T> init ) :
        shape_{shape.begin(), shape.end()}, vector_{std::make_shared<vector_type>(init)}
        {
            if (vector_->size() != checked_elements(shape_)) throw std::invalid_argument("initializer count differs from shape");
        }

        constexpr tensor( std::initializer_list<unsigned long> shape, std::initializer_list<T> init ) :
        shape_{shape.begin(), shape.end()}, vector_{std::make_shared<vector_type>(init)}
        {
            if (vector_->size() != checked_elements(shape_)) throw std::invalid_argument("initializer count differs from shape");
        }

        ///
        /// @brief Construct a vector with the specified shape. All values initialized to default. With a default constructed allocator
        ///
        template<typename Another_Alloc>
        constexpr tensor( std::vector<unsigned long, Another_Alloc> const& shape ) :
        shape_{shape.begin(), shape.end()},
        vector_{std::make_shared<vector_type>(checked_elements(shape_), T{0})}
        {}

        constexpr tensor( std::initializer_list<unsigned long> shape ) :
        shape_{ shape.begin(), shape.end() },
        vector_{std::make_shared<vector_type>(checked_elements(shape_), T{0})}
        {}

        ///
        /// @brief Construct a vector with the specified shape and all values initialized to `init`. With a default constructed allocator
        ///
        template<typename Another_Alloc>
        constexpr tensor( std::vector<unsigned long, Another_Alloc> const& shape, T init ) :
        shape_{shape.begin(), shape.end()},
        vector_{std::make_shared<vector_type>(checked_elements(shape_), T{0})}
        {
            std::fill( begin(), end(), init );
        }

        constexpr tensor( std::initializer_list<unsigned long> shape,  T init ) :
        shape_{shape.begin(), shape.end()},
        vector_{std::make_shared<vector_type>(checked_elements(shape_), T{0})}
        {
            std::fill( begin(), end(), init );
        }

        ///
        /// @brief Copy-ctor.
        ///
        constexpr tensor( self_type const& other ) : shape_{ other.shape_ }
        {
            vector_ = other.vector_;
            (*this).id_ = other.id_;
        }

        ///
        /// @brief Move-ctor.
        ///
        constexpr tensor( self_type && other ) : shape_{ other.shape_ }
        {
            vector_ = other.vector_;
            (*this).id_ = other.id_;
        }

        ///
        /// @brief Copy-assignment.
        ///
        constexpr self_type& operator = ( self_type const& other )
        {
            shape_ = other.shape_;
            vector_ = other.vector_;
            (*this).id_ = other.id_;
            return *this;
        }

        ///
        /// @brief Move-assignment.
        ///
        constexpr self_type& operator = ( self_type && other )
        {
            shape_ = other.shape_;
            vector_ = other.vector_;
            (*this).id_ = other.id_;
            return *this;
        }

        ///
        /// @brief Iterator to the first element of the tensor.
        ///
        constexpr auto begin()
        {
            return data();
        }

        ///
        /// @brief Iterator to the first element of the tensor.
        ///
        constexpr auto begin() const
        {
            return data();
        }

        ///
        /// @brief Iterator to the first element of the tensor.
        ///
        constexpr auto cbegin() const
        {
            return begin();
        }

        ///
        /// @brief Iterator to the element following the last element of the tensor.
        ///
        constexpr auto end()
        {
            return size() ? begin() + size() : begin();
        }

        ///
        /// @brief Iterator to the element following the last element of the tensor.
        ///
        constexpr auto end() const
        {
            return size() ? begin() + size() : begin();
        }

        ///
        /// @brief Iterator to the element following the last element of the tensor.
        ///
        constexpr auto cend() const
        {
            return  end();
        }


        ///
        /// @brief Reverse iterator to the first element of the tensor.
        ///
        constexpr auto rbegin()
        {
            return std::make_reverse_iterator( end() );
        }

        ///
        /// @brief Reverse iterator to the first element of the tensor.
        ///
        constexpr auto rbegin() const
        {
            return std::make_reverse_iterator( end() );
        }

        ///
        /// @brief Reverse iterator to the first element of the tensor.
        ///
        constexpr auto crbegin() const
        {
            return std::make_reverse_iterator( cend() );
        }

        ///
        /// @brief Reverse iterator to the element following the last element of the tensor.
        ///
        constexpr auto rend()
        {
            return std::make_reverse_iterator( begin() );
        }

        ///
        /// @brief Reverse iterator to the element following the last element of the tensor.
        ///
        constexpr auto rend() const
        {
            return std::make_reverse_iterator( begin() );
        }

        ///
        /// @brief Reverse iterator to the element following the last element of the tensor.
        ///
        constexpr auto crend() const
        {
            return std::make_reverse_iterator( cbegin() );
        }


        ///
        /// @brief Number of elements in the tensor.
        ///
        constexpr unsigned long size() const
        {
            if ( !vector_ ) return 0;
            return (*vector_ ).size();
        }


        ///
        /// @brief Check if the tensor has elements.
        ///
        [[nodiscard]] constexpr bool empty() const
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
        constexpr unsigned long ndim() const
        {
            return shape_.size();
        }

        ///
        /// @brief Shape of the tensor.
        ///
        constexpr std::vector<unsigned long> const shape() const
        {
            return std::vector<unsigned long>{ shape_.begin(), shape_.end() };
        }


        ///
        /// @brief A deep copy of the tensor.
        ///
        constexpr self_type& deep_copy( self_type const& other )
        {
            auto replacement=other.deep_copy();
            *this=std::move(replacement);
            return *this;
        }

        constexpr self_type const deep_copy() const
        {
            if(shape_.empty() && empty()) return self_type{};
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
            if (idx >= size()) throw std::out_of_range("tensor index");
            return *(data()+idx);
        }

        // 1-D view
        constexpr value_type const& operator[]( unsigned long idx ) const
        {
            if (idx >= size()) throw std::out_of_range("tensor index");
            return *(data()+idx);
        }

        ///
        /// @brief Resize the tensor with a new shape.
        ///
        constexpr self_type& resize( std::vector< unsigned long > const& new_shape )
        {
            auto shape = new_shape;
            auto n = checked_elements(shape);
            checked_multiply(n, sizeof(T));
            if (size() != n) {
                auto replacement = std::make_shared<vector_type>(n, T{});
                vector_ = std::move(replacement);
            }
            shape_.swap(shape);
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
            if (!_new_shape.empty() && _new_shape.back() == std::numeric_limits<unsigned long>::max()) {
                _new_shape.back()=1;
                auto known=checked_elements(_new_shape);
                if(!known || size()%known) throw std::invalid_argument("invalid inferred extent");
                _new_shape.back()=size()/known;
            }
            if(checked_elements(_new_shape)!=size()) throw std::invalid_argument("reshape changes element count; use resize");
            shape_.swap(_new_shape);
            return *this;
        }

        ///
        /// @brief Returns pointer to the underlying array serving as element storage.
        ///
        /// The pointer is such that range [data(); data() + size()) is always a valid range,
        /// even if the container is empty (data() is not dereferenceable in that case).
        ///
        constexpr value_type* data()
        {
            return (*vector_).data();
        }

        ///
        /// @brief Returns pointer to the underlying array serving as element storage.
        ///
        /// The pointer is such that range [data(); data() + size()) is always a valid range,
        /// even if the container is empty (data() is not dereferenceable in that case).
        ///
        constexpr const value_type* data() const
        {
            return (*vector_).data();
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
            if(shape()!=other.shape()) throw std::invalid_argument("tensor shape mismatch");
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
            if(shape()!=other.shape()) throw std::invalid_argument("tensor shape mismatch");
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
            if(shape()!=other.shape()) throw std::invalid_argument("tensor shape mismatch");
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
            if(shape()!=other.shape()) throw std::invalid_argument("tensor shape mismatch");
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

        constexpr value_type as_scalar() const
        {
            better_assert( size() == 1, "Expecting tensor has a single value, but got ", size() );
            return *begin();
        }

        template< typename U >
        constexpr auto as_type() const
        {
            tensor<U, typename std::allocator_traits<Allocator>:: template rebind_alloc<U>> ans{ (*this).shape() };
            std::copy( (*this).begin(), (*this).end(), ans.begin() );
            return ans;
        }
    }; // struct tensor

    template <typename T, typename A=default_allocator<T> >
    constexpr tensor<T, A> as_tensor( T val )
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

// All numerical operations defined in tensor.tcc
#include "./tensor.tcc"

#endif//HQKGLAXWWVFBFHQNHBVTQJKGUFTPCQPTPXDVNOSBDJIBHITCEKDISJYNAMCPLJDURURDAISFV

