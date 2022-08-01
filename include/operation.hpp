#ifndef IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM
#define IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM

#include "./includes.hpp"
#include "./place_holder.hpp"
#include "./variable.hpp"
#include "./constant.hpp"
#include "./value.hpp"
#include "./session.hpp"
#include "./utils/range.hpp"
#include "./utils/debug.hpp"
#include "./config.hpp"
#include "./utils/context_cast.hpp"
#include "./utils/for_each.hpp"
#include "./utils/id.hpp"
#include "./utils/enable_shared.hpp"
#include "./utils/fmt.hpp"
#include "./utils/enable_serializer.hpp"

namespace ceras
{

    ///
    /// @brief The default identity output shape calculator for unary/binary operators. Should be overrided for some special operators
    ///
    struct identity_output_shape_calculator
    {
        std::vector<unsigned long> operator()( std::vector<unsigned long> const& input_shape ) const noexcept
        {
            return input_shape;
        }

        std::vector<unsigned long> operator()( std::vector<unsigned long> const& lhs_input_shape, std::vector<unsigned long> const& rhs_input_shape ) const noexcept
        {
            return lhs_input_shape.size() > rhs_input_shape.size() ? lhs_input_shape : rhs_input_shape;
        }

        std::vector<unsigned long> operator()() const noexcept
        {
            return std::vector<unsigned long>{ {-1UL,} };
        }
    }; // struct identity_output_shape_calculator



    ///
    /// @brief A unary operator is composed of a.) an input expression, b.) a forward action and c.) a backward action.
    ///
    template< typename Operator, typename Forward_Action, typename Backward_Action, typename Output_Shape_Calculator, typename Serializer >
    struct unary_operator :
        enable_id<unary_operator<Operator, Forward_Action, Backward_Action, Output_Shape_Calculator, Serializer>, "Unary_Operator">,
        enable_unary_serializer<unary_operator<Operator, Forward_Action, Backward_Action, Output_Shape_Calculator, Serializer> >
    {
        Operator op_;
        Forward_Action forward_action_;
        Backward_Action backward_action_;
        Output_Shape_Calculator output_shape_calculator_;
        Serializer serializer_;

        typedef decltype( std::declval<Forward_Action>()( std::declval<decltype(op_)>().forward() ) ) tensor_type;

        tensor_type input_data_;
        tensor_type output_data_;

        unary_operator( Operator const& op, Forward_Action const& forward_action, Backward_Action const& backward_action, Output_Shape_Calculator const& output_shape_calculator, Serializer const& serializer ) noexcept :
            op_{op}, forward_action_{ forward_action }, backward_action_{ backward_action }, output_shape_calculator_{ output_shape_calculator }, serializer_{ serializer } { }

        auto forward()
        {
            auto& sess = get_default_session<tensor_type>();
            output_data_= sess.query_forward_cache( (*this).id() );

            if ( output_data_.empty() )
            {
                input_data_ = op_.forward();
                output_data_ = forward_action_( input_data_ );
                sess.update_forward_cache( (*this).id(), output_data_ );
            }

            return output_data_;
        }

        void backward( tensor_type const& grad )
        {
            auto const& current_gradient = backward_action_( input_data_, output_data_, grad );
            op_.backward( current_gradient );
        }

        ///
        /// @brief Calculate the output tensor shape.
        ///
        std::vector<unsigned long> shape() const noexcept
        {
            return output_shape_calculator_( op_.shape() );
        }


        Operator const op() const noexcept
        {
            return op_;
        }

        Serializer const serializer() const noexcept
        {
            return serializer_;
        }
    };

    ///
    /// @brief Construct an unary operator by passing the forward/backward actions and output shape calculator
    ///
    template< typename Forward_Action, typename Backward_Action, typename Output_Shape_Calculator= identity_output_shape_calculator, typename Serializer = default_unary_expression_serializer >
    auto constexpr make_unary_operator( Forward_Action const& unary_forward_action,
                                        Backward_Action const& unary_backward_action,
                                        std::string const& name = "Anonymous_Unary_Operator",
                                        Output_Shape_Calculator const& output_shape_calculator = Output_Shape_Calculator{},
                                        Serializer const& serializer = Serializer{}
            ) noexcept
    {
        return [&]( auto const& op ) noexcept
        {
            auto ans = unary_operator{ op, unary_forward_action, unary_backward_action, output_shape_calculator, serializer };
            ans.name_ = name;
            return ans;
        };
    }


    ///
    /// @brief A binary operator is composed of a.) a left-side input expression, b.) a right-side input expression, c.)  a forward action and d.) a backward action.
    ///
    template< typename Lhs_Operator, typename Rhs_Operator, typename Forward_Action, typename Backward_Action, typename Output_Shape_Calculator, typename Serializer >
    struct binary_operator :
        enable_id<binary_operator<Lhs_Operator, Rhs_Operator, Forward_Action, Backward_Action, Output_Shape_Calculator, Serializer>, "Binary Operator">,
        enable_binary_serializer<binary_operator<Lhs_Operator, Rhs_Operator, Forward_Action, Backward_Action, Output_Shape_Calculator, Serializer>>
    {
        Lhs_Operator lhs_op_;
        Rhs_Operator rhs_op_;
        Forward_Action forward_action_;
        Backward_Action backward_action_; // backward action for binary operator produces a tuple of two tensors
        Output_Shape_Calculator output_shape_calculator_;
        Serializer serializer_;


        typedef typename tensor_deduction<Lhs_Operator, Rhs_Operator>::tensor_type tensor_type; // defined in value.hpp

        tensor_type lhs_input_data_;
        tensor_type rhs_input_data_;
        tensor_type output_data_;

        binary_operator( Lhs_Operator const& lhs_op, Rhs_Operator const& rhs_op, Forward_Action const& forward_action, Backward_Action const& backward_action, Output_Shape_Calculator const& output_shape_calculator, Serializer const& serializer) noexcept :
            lhs_op_{lhs_op}, rhs_op_{rhs_op}, forward_action_{ forward_action }, backward_action_{ backward_action }, output_shape_calculator_{ output_shape_calculator }, serializer_{ serializer }  { }

        auto forward()
        {
            auto& sess = get_default_session<tensor_type>();
            output_data_= sess.query_forward_cache( (*this).id() );

            if ( !output_data_.empty() )
                return output_data_;

            static_assert( !(is_value_v<Lhs_Operator> && is_value_v<Rhs_Operator>), "Not valid for two values" );

            if constexpr ( is_value_v<Lhs_Operator> )
            {
                rhs_input_data_ = rhs_op_.forward();
                lhs_input_data_ = lhs_op_.forward( rhs_input_data_ );
            }
            else if constexpr ( is_value_v<Rhs_Operator> )
            {
                lhs_input_data_ = lhs_op_.forward();
                rhs_input_data_ = rhs_op_.forward( lhs_input_data_ );
            }
            else
            {
                lhs_input_data_ = lhs_op_.forward();
                rhs_input_data_ = rhs_op_.forward();
            }

            output_data_ = forward_action_( lhs_input_data_, rhs_input_data_ );
            sess.update_forward_cache( (*this).id(), output_data_ );
            return output_data_;
        }

        ///
        /// @brief Backward action, grad back-propagated.
        ///
        void backward( tensor_type const& grad )
        {
            auto const& [current_gradient_lhs, current_gradient_rhs] = backward_action_( lhs_input_data_, rhs_input_data_, output_data_, grad );
            lhs_op_.backward( current_gradient_lhs );
            rhs_op_.backward( current_gradient_rhs );
        }

        ///
        /// @brief Calculate the output shape.
        ///
        std::vector<unsigned long> shape() const noexcept
        {
            if constexpr ( is_value_v<Lhs_Operator> )
                return rhs_op_.shape();
            else if constexpr ( is_value_v<Rhs_Operator> )
                return lhs_op_.shape();
            else
                return output_shape_calculator_( lhs_op_.shape(), rhs_op_.shape() );
        }

        Lhs_Operator const& lhs_op() const noexcept
        {
            return lhs_op_;
        }

        Rhs_Operator const& rhs_op() const noexcept
        {
            return rhs_op_;
        }

        Serializer const& serializer() const noexcept
        {
            return serializer_;
        }
    };

    template< typename Forward_Action, typename Backward_Action, typename Output_Shape_Calculator= identity_output_shape_calculator, typename Serializer = default_binary_expression_serializer >
    auto make_binary_operator( Forward_Action const& binary_forward_action,
                               Backward_Action const& binary_backward_action,
                               std::string const& name = "Anonymous_Binary_Operator",
                               Output_Shape_Calculator const& output_shape_calculator = Output_Shape_Calculator{},
                               Serializer const& serializer = Serializer{}) noexcept
    {
        return [&]( auto const& lhs_op, auto const& rhs_op ) noexcept
        {
            auto ans = binary_operator{ lhs_op, rhs_op, binary_forward_action, binary_backward_action, output_shape_calculator, serializer };
            ans.name_ = name;
            return ans;
        };
    }


    template< typename T >
    struct is_unary_operator : std::false_type{};

    template< typename Operator, typename Forward_Action, typename Backward_Action, typename Output_Shape_Calculator, typename Serializer >
    struct is_unary_operator< unary_operator<Operator, Forward_Action, Backward_Action, Output_Shape_Calculator, Serializer> > : std::true_type {};

    ///
    /// If T is an instance of a unary_operator, the constant value equals to `true`. `false` otherwise.
    ///
    template< class T >
    inline constexpr bool is_unary_operator_v = is_unary_operator<T>::value;

    ///
    /// @concept Unary_Operator<>
    /// @brief A type that represents an unary operator.
    ///
    template< typename T >
    concept Unary_Operator = is_unary_operator_v<T>;


    template< typename T >
    struct is_binary_operator : std::false_type{};

    template< typename Lhs_Operator, typename Rhs_Operator, typename Forward_Action, typename Backward_Action, typename Output_Shape_Calculator, typename Serializer >
    struct is_binary_operator< binary_operator<Lhs_Operator, Rhs_Operator, Forward_Action, Backward_Action, Output_Shape_Calculator, Serializer> > : std::true_type {};

    ///
    /// If T is an instance of a binary_operator, the constant value equals to `true`. Otherwise this value is `false`.
    ///
    template< class T >
    inline constexpr bool is_binary_operator_v = is_binary_operator<T>::value;

    ///
    /// @concept Binary_Operator<>
    /// @brief A type that represents a binary operator.
    ///
    template< typename T >
    concept Binary_Operator = is_binary_operator_v<T>;

    ///
    /// @concept Operator<>
    /// @brief A type that represents an unary or a binary operator.
    ///
    template< typename T >
    concept Operator = Unary_Operator<T> || Binary_Operator<T>;

    ///
    /// @concept Expression<>
    /// @brief A type that represents a unary operator, a binary operator, a variable, a place_holder, a constant or a value
    ///
    template< typename T >
    concept Expression = Operator<T> || Variable<T> || Place_Holder<T> || Constant<T> || Value<T>;

    template< Expression Ex >
    std::tuple<std::string, std::vector<std::string>> const serialize( Ex const& ex )
    {
        return ex.serialize();
    }


    ///
    /// Generating the computation graph, in [graph description language](https://www.graphviz.org/documentation/).
    /// @param ex An expression.
    /// @return A string describing the computation graph, in graph description language.
    ///
    template< Expression Ex >
    inline std::string computation_graph( Ex const& ex ) noexcept
    {
        auto generate_node_and_label = []<Expression Expr>( Expr const& expr ) noexcept
        {
            std::string const id = std::to_string( expr.id() );
            std::string const name = expr.name();
            std::string node = std::string{"n"} + id;

            std::vector<long long> shape;
            {
                std::vector<unsigned long> _shape = expr.shape();
                shape.resize( _shape.size() );
                std::copy( _shape.begin(), _shape.end(), shape.begin() );
                if ( _shape.size() > 0 && _shape[0] == -1UL )
                    shape[0] = -1;
            }

            std::string label = fmt::format( "{} <shape:{}> [id:{}]", name, shape, id);
            return std::make_tuple( node, label );
        };

        auto generate_dot = [&generate_node_and_label]<Expression Expr>( Expr const& expr, auto const& _generate_dot ) noexcept
        {
            auto const& [node, label] = generate_node_and_label( expr );
            std::string const& expr_dot = node + std::string{" [label=\""} + label + std::string{"\"] ;\n"};

            if constexpr( is_unary_operator_v<Expr> )
            {
                auto const& [n_node, n_label] = generate_node_and_label( expr.op_ );
                std::string const& arrow_relation = n_node + std::string{" -> "} + node + std::string{" ;\n"};
                std::string const& op_dot = _generate_dot( expr.op_, _generate_dot );
                return expr_dot + arrow_relation + op_dot;
            }
            else if constexpr( is_binary_operator_v<Expr> )
            {
                // for LHS operator
                auto const& [n_lhs_node, n_lhs_label] = generate_node_and_label( expr.lhs_op_ );
                std::string const& arrow_lhs_relation = n_lhs_node + std::string{" -> "} + node + std::string{" ;\n"};
                std::string const& op_lhs_dot = _generate_dot( expr.lhs_op_, _generate_dot );

                // for RHS operator
                auto const& [n_rhs_node, n_rhs_label] = generate_node_and_label( expr.rhs_op_ );
                std::string const& arrow_rhs_relation = n_rhs_node + std::string{" -> "} + node + std::string{" ;\n"};
                std::string const& op_rhs_dot = _generate_dot( expr.rhs_op_, _generate_dot );

                return expr_dot + arrow_lhs_relation + arrow_rhs_relation + op_lhs_dot + op_rhs_dot;
            }
            else if constexpr ( is_variable_v<Expr> )
            {
                std::vector<unsigned long> const& shape = expr.shape();
                bool const training_state = expr.trainable();

                // shape
                std::stringstream ss;
                std::copy( shape.begin(), shape.end(), std::ostream_iterator<unsigned long>( ss, " " ) );
                std::string const& str_shape = ss.str() + (training_state ? std::string{"), trainable"} : std::string{"), non-trainable"});
                // trainable state
                std::string const& new_label = label + std::string{"[("} + str_shape + std::string{"]"};

                if (!training_state)
                    return node + std::string{" [shape=box,label=\""} + new_label + std::string{"\"] ;\n"};

                return node + std::string{" [peripheries=3,style=filled,color=\".7 .3 1.0\",shape=box,label=\""} + new_label + std::string{"\"] ;\n"};
            }
            else
            {
                return expr_dot;
            }
        };

        std::string const& head = "\n\ndigraph g {\n";
        std::string const& tail = "}\n\n";
        return head + generate_dot( ex, generate_dot ) + tail;
    }


    ///
    /// @brief Broadcast an expression to produce a new shape.
    ///
    /// \code{.cpp}
    /// auto e = ...; // shape `(1, 64)`
    /// auto f = broadcast( {128, 128, 64} )( e ); // shape `(128, 128, 64)`
    /// \endcode
    ///
    auto inline broadcast( std::vector<unsigned long> const& new_shape ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [new_shape, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [new_shape, forward_cache]<Tensor Tsor>( Tsor const& input )noexcept
                {
                    std::vector<unsigned long> const& old_shape = input.shape();
                    if (new_shape == old_shape) return input;

                    // Note: ceras only considers simple cases such as `Wx+b`, in which `b` is either has shape `(n,)` or shape `(1,n)`, the implementation is simplified accordingly.
                    unsigned long const new_size = std::accumulate( new_shape.begin(), new_shape.end(), 1UL, []( auto x, auto y ){ return x*y; } );
                    unsigned long const old_size = std::accumulate( old_shape.begin(), old_shape.end(), 1UL, []( auto x, auto y ){ return x*y; } );
                    unsigned long const factor = new_size / old_size;

                    Tsor& ans = context_cast<Tsor>( forward_cache );
                    ans.resize( new_shape );
                    for ( auto idx : range( factor ) )
                        for_each( input.begin(), input.end(), ans.begin()+idx*old_size, []( auto x, auto& y ){ y = x; } );
                    return ans;
                },
                [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad) noexcept
                {
                    if ( input.shape() == output.shape() ) return grad;

                    better_assert( output.shape() == grad.shape(), fmt::format( "Error with broadcast: shape mismatch. Output shape is {}, but grad shape is {}", output.shape(), grad.shape() ) );

                    std::vector<unsigned long> const& old_shape = input.shape();
                    unsigned long const old_size = std::accumulate( old_shape.begin(), old_shape.end(), 1UL, []( auto x, auto y ){ return x*y; } );
                    std::vector<unsigned long> const& new_shape = grad.shape();
                    unsigned long const new_size = std::accumulate( new_shape.begin(), new_shape.end(), 1UL, []( auto x, auto y ){ return x*y; } );
                    unsigned long const factor = new_size / old_size;

                    Tsor& ans = context_cast<Tsor>( backward_cache );
                    ans.resize( input.shape() );
                    for_each( ans.begin(), ans.end(), []( auto& x ){ x = 0; } );
                    for ( auto idx : range( factor ) )
                        for_each( ans.begin(), ans.end(), grad.begin()+idx*old_size, []( auto& x, auto y ){ x += y; } );

                    return ans;
                },
                "broadcast",
                [new_shape]( std::vector<unsigned long> const&) noexcept { return new_shape; }
            )(ex);
        };
    }



    namespace
    {
        struct plus_context
        {
            auto make_forward() const noexcept
            {
                return []<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
                {
                    better_assert( !has_nan( lhs_tensor ), "forward propagation for operator plus: lhs_tensor contains Nan!" );
                    better_assert( !has_nan( rhs_tensor ), "forward propagation for operator plus: rhs_tensor contains Nan!" );
                    return add( lhs_tensor, rhs_tensor );
                };
            }

            auto const make_backward() const noexcept
            {
                return []<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& grad ) noexcept
                {
                   better_assert( !has_nan( grad ), "backprop: upcoming gradient for operator + contains NaN!" );

                   auto const& grad_fun = [&grad]( auto const& input )
                   {
                       Tsor ans = grad.deep_copy();
                       while( input.ndim() < ans.ndim() )
                           ans = sum( ans, 0 );
                       auto const& shape = input.shape();
                       for ( auto axis : range( input.ndim() ) )
                           if ( shape[axis] == 1 )
                              ans = sum( ans, axis, true );
                       return ans;
                   };
                   return std::make_tuple( grad_fun( lhs_input), grad_fun( rhs_input ) );
                };
            }
        }; // plus_context
    }//anonymous namespace

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr plus( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        auto const& shape_calculator = []( std::vector<unsigned long> const& l, std::vector<unsigned long> const& r ) noexcept
        {
            return broadcast_shape( l, r );
        };


        return make_binary_operator( plus_context{}.make_forward(), plus_context{}.make_backward(), "plus", shape_calculator )( lhs_ex, rhs_ex );
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr operator + ( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return plus( lhs_ex, rhs_ex );
    }

    template< Expression Ex >
    auto constexpr operator + ( Ex const& ex ) noexcept
    {
        return ex;
    }

    namespace
    {
        struct multiplication_context
        {
            auto make_forward() const noexcept
            {
                return []( std::shared_ptr<std::any> forward_cache ) noexcept
                {
                    return [forward_cache]<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
                    {
                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        multiply( lhs_tensor, rhs_tensor, ans );
                        return ans;
                    };
                };
            }
            auto make_backward() const noexcept
            {
                return []( std::shared_ptr<std::any> backward_cache_lhs, std::shared_ptr<std::any> backward_cache_rhs ) noexcept
                {
                    return [backward_cache_lhs, backward_cache_rhs]<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& grad ) noexcept
                    {
                       // left branch <-- grad * rhs^T
                       auto const& g_shape = grad.shape();
                       auto const[m, n] = std::make_tuple( g_shape[0], g_shape[1] ); // 4, 1
                       auto const k = *(lhs_input.shape().rbegin()); // 13

                       Tsor& lhs_grad = context_cast<Tsor>( backward_cache_lhs );
                       lhs_grad.resize( lhs_input.shape() );

                       gemm( grad.data(), false, rhs_input.data(), true, m, n, k, lhs_grad.data() );

                       // right branch <-- lhs^T * grad
                       Tsor& rhs_grad = context_cast<Tsor>( backward_cache_rhs );
                       rhs_grad.resize( rhs_input.shape() );
                       gemm( lhs_input.data(), true, grad.data(), false, k, m, n, rhs_grad.data() );

                       return std::make_tuple( lhs_grad, rhs_grad );
                    };
                };
            }
        };//multiplication_context
    }//anonymous namespace

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto operator * ( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        // case of Value * Operator and Operator * Value
        if constexpr( is_value_v<Lhs_Expression> || is_value_v<Rhs_Expression> )
        {
            return elementwise_product( lhs_ex, rhs_ex );
        }
        else
        {
            auto const& shape_calculator = []( std::vector<unsigned long> const& l, std::vector<unsigned long> const& r ) noexcept
            {
                better_assert( l.size() == 2, fmt::format( "expecting l size of 2, but got {}", l.size() ) );
                better_assert( r.size() == 2, fmt::format( "expecting r size of 2, but got {}", r.size() ) );
                better_assert( l[1] == r[0], fmt::format( "expecting l[1] == r[0], but l[1]={}, r[0]={}", l[1], r[0] ) ); // TODO: what if unknown dimension???
                return std::vector<unsigned long>{ {l[0], r[1]} };
            };
            std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
            std::shared_ptr<std::any> backward_cache_lhs = std::make_shared<std::any>();
            std::shared_ptr<std::any> backward_cache_rhs = std::make_shared<std::any>();
            return make_binary_operator( multiplication_context{}.make_forward()(forward_cache), multiplication_context{}.make_backward()(backward_cache_lhs, backward_cache_rhs), "multiply", shape_calculator )( lhs_ex, rhs_ex );
        }
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto multiply( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return lhs_ex * rhs_ex;
    }

    ///
    /// @brief Negative operator, elementwise.
    /// @code{.cpp}
    /// auto x = variable{ ... };
    /// auto ix = negative( x );
    /// @endcode
    ///
    template <Expression Ex>
    auto constexpr negative( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& tensor ) noexcept
                                    {
                                        better_assert( !has_nan( tensor ), "forward propagation for operator log: tensor contains Nan!" );
                                        return -tensor;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator negative contains NaN!" );
                                        return -grad;
                                    },
                                    "negative"
                )( ex );
    };

    template <Expression Ex>
    auto constexpr operator - ( Ex const& ex ) noexcept
    {
        return negative( ex );
    }


    ///
    /// @brief Inverse operator, elementwise.
    /// @code{.cpp}
    /// auto x = variable{ ... };
    /// auto ix = inverse( x );
    /// @endcode
    template <Expression Ex>
    auto constexpr inverse( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& tensor ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( tensor.shape() );
                                        for_each( tensor.begin(), tensor.end(), ans.begin(), [](auto const x, auto& y) { y = (x > 0.0) ? (1.0/std::max(eps, x)) : (1.0/std::min(-eps, x)); });
                                        //for_each( tensor.begin(), tensor.end(), ans.begin(), [](auto const x, auto& y) { y = 1.0/x; } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( ans.begin(), ans.end(), grad.begin(), input.begin(), []( auto& x, auto y, auto z ){ x = - y / std::max(z*z, eps); } );
                                        //for_each( ans.begin(), ans.end(), grad.begin(), input.begin(), []( auto& x, auto y, auto z ){ x = - y / (z*z); } );
                                        ans.resize( grad.shape() );
                                        return ans;
                                    },
                                    "inverse"
                )( ex );
    };




    ///
    /// @brief Multiply two input operators, elementwise.
    /// @code{.cpp}
    /// auto x = variable{ tensor<float>{ {2, 3, 5} } };
    /// auto y = variable{ tensor<float>{ {2, 3, 5} } };
    /// auto z = elementwise_product( x, y ); // z = x*y;
    /// @endcode
    ///
    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr elementwise_product( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return make_binary_operator( []<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
                                     {
                                        return elementwise_product( lhs_tensor, rhs_tensor );
                                     },
                                     []<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const grad ) noexcept
                                     {
                                        auto const& grad_fun = [&grad]( auto const& input, auto const& other_input )
                                        {
                                            Tsor ans = elementwise_product( grad, other_input );
                                            while( input.ndim() < ans.ndim() )
                                                ans = sum( ans, 0 );
                                            auto const& shape = input.shape();
                                            for ( auto axis : range( input.ndim() ) )
                                                if ( shape[axis] == 1 )
                                                    ans = sum( ans, axis, true );
                                            return ans;
                                        };
                                        return std::make_tuple( grad_fun( lhs_input, rhs_input ), grad_fun( rhs_input, lhs_input ) );
                                     },
                                     "elementwise_product"
                )( lhs_ex, rhs_ex );
    };

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr elementwise_multiply( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return elementwise_product( lhs_ex, rhs_ex );
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr hadamard_product( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return elementwise_product( lhs_ex, rhs_ex );
    }


    ///
    /// @brief Divide one tensor by the other.
    /// @code{.cpp}
    /// auto x = varialbe{ tensor<float>{ {17, 12} } };
    /// auto y = varialbe{ tensor<float>{ {17, 12} } };
    /// auto z = divide( x, y ); // z = x / y
    /// @endcode
    ///
    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr divide( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return elementwise_product( lhs_ex, inverse( rhs_ex ) );
    }

    ///
    /// @brief Divide one tensor by the other.
    /// @code{.cpp}
    /// auto x = varialbe{ tensor<float>{ {17, 12} } };
    /// auto y = varialbe{ tensor<float>{ {17, 12} } };
    /// auto z = x/y; // same as  `divide( x, y );`
    /// @endcode
    ///
    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr operator / ( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return divide( lhs_ex, rhs_ex );
    }

    ///
    /// @brief Sum up all elements, returns a scalar.
    /// @code{.cpp}
    /// auto x = variable{ ... };
    /// auto y = sum_reduce( x );
    /// @endcode
    ///
    template <Expression Ex>
    auto constexpr sum_reduce( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator sum_reduce: tensor contains Nan!" );
                                        return reduce_sum( tsor );
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator sum_reduce contains NaN!" );
                                        better_assert( grad.size() == 1, "sum_reduce should only output one value" );
                                        Tsor ans = ones_like( input );
                                        ans *= grad[0];
                                        return ans;
                                    },
                                    "sum_reduce",
                                    []( std::vector<unsigned long> const& ) noexcept { return std::vector<unsigned long>{ {1,} }; }
                )( ex );
    }

    template <Expression Ex>
    auto constexpr reduce_sum( Ex const& ex ) noexcept
    {
        return sum_reduce( ex );
    }

    ///
    /// @brief Computes the mean of elements across all dimensions of an expression.
    /// @param ex Incoming expression.
    ///
    /// Example code:
    ///
    /// \code{.cpp}
    /// auto va = place_holder<tensor<float>>{};
    /// auto vb = variable{ random<float>{ 3, 4} };
    /// auto diff = mean_reduce( va, vb );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr mean_reduce( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator mean: tensor contains Nan!" );
                                        return reduce_mean( tsor );
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator mean_reduce contains NaN!" );
                                        better_assert( grad.size() == 1, "mean_reduce should only output one value" );
                                        Tsor ans = ones_like( input );
                                        ans *= grad[0];
                                        unsigned long const batch_size = (input.shape().size() == 1) ? 1 : (*(input.shape().begin()));
                                        ans /= static_cast<typename Tsor::value_type>(batch_size);
                                        return ans;
                                    },
                                    "mean_reduce",
                                    []( std::vector<unsigned long> const& ) noexcept { return std::vector<unsigned long>{ {1,} }; }
                )( ex );
    }

    ///
    /// @brief An alias name of mean_reduce.
    ///
    template <Expression Ex>
    auto constexpr reduce_mean( Ex const& ex ) noexcept
    {
        return mean_reduce( ex );
    }

    ///
    /// @brief An alias name of mean_reduce.
    ///
    template <Expression Ex>
    auto constexpr mean( Ex const& ex ) noexcept
    {
        return mean_reduce( ex );
    }




    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr minus( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        if constexpr (is_value_v<Rhs_Expression>)
        {
            return negative( plus( negative(lhs_ex), rhs_ex ) );
        }
        else
        {
            return plus( lhs_ex, negative(rhs_ex) );
        }
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr operator - ( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return minus( lhs_ex, rhs_ex );
    }


    ///
    /// Returns the square of the input
    ///
    /// @param ex The input operator.
    /// @return An instance of a unary_operator that evaluate the squared value of the input operator.
    ///
    /// Example code:
    /// @code
    /// auto e = variable<tensor<float>>{ /*...*/ };
    /// auto square = square(e);
    /// @endcode
    ///
    template <Expression Ex>
    auto constexpr square( Ex const& ex ) noexcept
    {
        return make_unary_operator( []<Tensor Tsor>( Tsor const& tsor ) noexcept
                                    {
                                        better_assert( !has_nan( tsor ), "forward propagation for operator square: tensor contains Nan!" );
                                        Tsor ans = tsor.deep_copy();
                                        std::for_each( ans.data(), ans.data() + ans.size(), []( auto & v ){ v *= v; } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        better_assert( !has_nan( grad ), "input gradient for operator square contains NaN!" );
                                        Tsor ans = input.deep_copy();
                                        ans *= grad;
                                        ans *= typename Tsor::value_type{2};
                                        return ans;
                                    },
                                    "square"
                )( ex );
    }


    ///
    /// @brief  Computes the square root of the sum of the squares of x and y.
    ///
    /// @param x The first operator.
    /// @param y The second operator.
    ///
    ///
    /// Example code:
    /// @code{.cpp}
    /// auto x = variable<tensor<float>>{ /*...*/ };
    /// auto y = variable<tensor<float>>{ /*...*/ };
    /// auto sqr = hypot( x, y );
    /// @endcode
    ///
    template <Expression Ex, Expression Ey>
    auto constexpr hypot( Ex const& ex, Ey const& ey ) noexcept
    {
        return sqrt( square(ex) + square(ey) );
    }

    template <typename Float> requires std::floating_point<Float>
    auto constexpr clip( Float lower, Float upper=std::numeric_limits<Float>::max() ) noexcept
    {
        return [lower, upper]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator( [lower, upper]<Tensor Tsor>( Tsor const& tsor ) noexcept
                                        {
                                            better_assert( !has_nan( tsor ), "forward propagation for operator clip: tensor contains Nan!" );
                                            Tsor ans = tsor.deep_copy();
                                            clip( ans, lower, upper );
                                            return ans;
                                        },
                                        [lower, upper]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                        {
                                            better_assert( !has_nan( grad ), "input gradient for operator clip contains NaN!" );
                                            const typename Tsor::value_type zero{0};
                                            Tsor ans = grad;
                                            for ( auto idx : range( input.size() ) )
                                                ans[idx] = (input[idx] < lower) ? zero :
                                                           (input[idx] > upper) ? zero :
                                                           ans[idx];
                                            return ans;
                                        },
                                        "clip"
                    )( ex );
        };
    }

    // include_batch_flag:
    //
    //  true: considering the batch size at the first dim
    //      - for an input of (1, 3, 4), expecting an incoming expression of shape like [BS, 12, 1 1]
    //      - expected output of shape [BS, 1, 3, 4]
    //  false: do not consider the batch size
    //      - for an input of (1, 3, 4), expecting an incoming expression of shape like [12, 1]
    //      - expected output of shape [1, 3, 4]
    auto inline reshape( std::vector<unsigned long> const& new_shape, bool include_batch_flag=true ) noexcept
    {
        return [new_shape, include_batch_flag]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [new_shape, include_batch_flag]<Tensor Tsor>( Tsor const& tsor ) noexcept
                {
                    unsigned long const new_size = std::accumulate( new_shape.begin(), new_shape.end(), 1UL, []( auto x, auto y ){ return x*y; } );
                    unsigned long const total_size = tsor.size();
                    unsigned long const batch_size = total_size / new_size;

                    better_assert( batch_size * new_size == total_size, "size mismatch for reshape operator, expect ",  batch_size*new_size, " but total input size is ", total_size, ", where batch_size is ", batch_size );

                    if ( !include_batch_flag )
                    {
                        better_assert( batch_size == 1, "expecting batch size of 1 while not including batch, but got ", batch_size );
                        Tsor ans{tsor};
                        ans.reshape( new_shape );
                        return ans;
                    }

                    std::vector<unsigned long> batched_new_shape;
                    {
                        batched_new_shape.resize( 1 + new_shape.size() );
                        batched_new_shape[0] = batch_size;
                        std::copy( new_shape.begin(), new_shape.end(), batched_new_shape.begin()+1 );
                    }

                    Tsor ans{ tsor };
                    ans.reshape( batched_new_shape );
                    return ans;
                },
                []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                {
                    Tsor ans{ grad };
                    ans.reshape( input.shape() );
                    return ans;
                },
                "reshape",
                [new_shape, include_batch_flag]( std::vector<unsigned long> const& shape ) noexcept
                {
                    if ( include_batch_flag == false )
                        return new_shape;

                    unsigned long const new_size = std::accumulate( new_shape.begin(), new_shape.end(), 1UL, []( auto x, auto y ){ return x*y; } );
                    unsigned long const total_size = std::accumulate( shape.begin(), shape.end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } );
                    unsigned long const batch_size = total_size / new_size;
                    std::vector<unsigned long> batched_new_shape;
                    {
                        batched_new_shape.resize( 1 + new_shape.size() );
                        batched_new_shape[0] = batch_size;
                        std::copy( new_shape.begin(), new_shape.end(), batched_new_shape.begin()+1 );
                    }
                    return batched_new_shape;
                },
                [new_shape, include_batch_flag]<Expression Self_Expression, Expression Input_Expression>( Self_Expression const& self_expression, Input_Expression const& input_expression ) noexcept
                { // serializer
                    auto const& [input_expression_name, input_expression_code] = serialize( input_expression );
                    std::string const& self_expression_identity = fmt::format( "unary_expression_{}_{}", self_expression.name(), self_expression.id() );
                    std::vector<std::string> self_expression_code = input_expression_code;
                    self_expression_code.emplace_back( fmt::format( "auto {} = {}( {}/*new_shape*/, {}/*include_batch_flag*/ )( {} );", self_expression_identity, self_expression.name(), new_shape, include_batch_flag, input_expression_name ) );
                    return std::make_tuple( self_expression_identity, self_expression_code );
                }
            )( ex );
        };
    }

    ///
    /// @brief Flatten input tensor.
    /// @code{.cpp}
    /// auto x = .....; // an operator returns tensor of shape ( 12, 34, 1 2 )
    /// auto f = flatten( x ); // returns tensor of shape (12*34*1*2, )
    /// @endcode
    ///
    template <Expression Ex>
    auto constexpr flatten( Ex const& ex ) noexcept
    {
        return make_unary_operator
        (
            []<Tensor Tsor>( Tsor const& tsor ) noexcept
            {
                better_assert( tsor.ndim() > 1, "Expecting dimension of incoming tensor to be greater than 1, but got ", tsor.ndim() );
                unsigned long const batch_size = *(tsor.shape().begin());
                unsigned long const rem = tsor.size() / batch_size;
                Tsor ans = tsor;
                return ans.reshape( {batch_size, rem} );
            },
            []<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
            {
                Tsor ans = grad;
                return ans.reshape( input.shape() );
            },
            "flatten",
            []( std::vector<unsigned long> const& shape ) noexcept
            {
                unsigned long const total = std::accumulate( shape.begin()+1, shape.end(), 1, []( unsigned long x, unsigned long y ){ return x*y; } );
                return std::vector<unsigned long>{ {shape[0], total,} }; // the 1st dim is batch size
            }
        )( ex );
    }

    ///
    /// @brief Expand input tensor with a length 1 axis inserted at index axis.
    /// @code{.cpp}
    /// auto x = variable<float>{ {2, 3, 4}}
    /// auto x0 = expand_dims(0)( x ); // new shape is ( 1, 2, 3, 4 )
    /// auto x1 = expand_dims(1)( x ); // new shape is ( 2, 1, 3, 4 )
    /// auto x2 = expand_dims(2)( x ); // new shape is ( 2, 3, 1, 4 )
    /// auto x3 = expand_dims(-1)( x ); // new shape is ( 2, 3, 4, 1 )
    /// @endcode
    ///
    constexpr auto inline expand_dims( int axis=-1 ) noexcept
    {
        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [=]<Tensor Tsor>( Tsor const& tsor ) noexcept
                {//forward propagation
                    Tsor ans = tsor;
                    std::vector<unsigned long> shape = ans.shape();
                    int const _axis = (axis == -1) ? shape.size() : axis;
                    shape.insert( shape.begin()+_axis, 1UL );
                    ans.reshape( shape );
                    return ans;
                },
                [=]<Tensor Tsor>( Tsor const& input, Tsor const& /*output*/, Tsor const& grad ) noexcept
                {//backward propagation
                    Tsor ans = grad;
                    ans.reshape( input.shape() );
                    return ans;
                },
                "expand_dims",
                [axis]( std::vector<unsigned long> const& shape ) noexcept
                {//shape calculator
                    std::vector<unsigned long> ans = shape;
                    //int offset = axis;
                    //if ( axis == -1 ) offset = ans.size();
                    int const offset = (axis == -1) ? shape.size() : axis;
                    ans.insert( ans.begin()+offset, 1UL );
                },
                [axis]<Expression Self_Expression, Expression Input_Expression>( Self_Expression const& self_expression, Input_Expression const& input_expression ) noexcept
                { // serializer
                    auto const& [input_expression_name, input_expression_code] = serialize( input_expression );
                    std::string const& self_expression_identity = fmt::format( "unary_expression_{}_{}", self_expression.name(), self_expression.id() );
                    std::vector<std::string> self_expression_code = input_expression_code;
                    self_expression_code.emplace_back( fmt::format( "auto {} = {}( {}/*axis*/ )( {} );", self_expression_identity, self_expression.name(), axis, input_expression_name ) );
                    return std::make_tuple( self_expression_identity, self_expression_code );
                }
            )(ex);
        };
    }


    ///
    /// @brief Returns the index with the largest value across axes of an input tensor.
    /// @code{.cpp}
    /// auto a = variable{ ... };
    /// auto ma = argmax( 1 )( a );
    /// @endcode
    ///
    auto inline argmax( unsigned long axis=0 ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [=]<Tensor Tsor>( Tsor const& input ) noexcept
                {
                    std::vector<unsigned long> const& shape = input.shape();
                    better_assert( axis < shape.size(), fmt::format("axis {} is greater than the dimension of the input tensor shape {}", axis, shape) );

                    // calculate the output tensor shape
                    std::vector<unsigned long> output_shape = shape;
                    std::copy( output_shape.begin()+axis+1, output_shape.end(), output_shape.begin()+axis );
                    output_shape.resize( output_shape.size() - 1 );
                    Tsor& ans = context_cast<Tsor>( forward_cache );
                    ans.resize( output_shape );

                    //  viewing the input tensor as a 3D tensor, and viewing the output tensor as a 2D tensor
                    unsigned long const bs = std::accumulate( shape.begin(), shape.begin()+axis, 1UL, []( unsigned long x, unsigned long y ){ return x*y; } );
                    unsigned long const row = shape[axis];
                    unsigned long const col = std::accumulate( shape.begin()+axis+1, shape.end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } );
                    auto cube_input = view_3d{ input.data(), bs, row, col };
                    auto matrix_output = view_2d{ ans.data(), bs, col };

                    for ( auto _bs : range( bs ) )
                        for ( auto _col : range( col ) )
                        {
                            unsigned long mx_idx = 0;
                            auto mx = cube_input[_bs][0][_col];
                            for ( auto _row : range( row ) )
                            {
                                if ( cube_input[_bs][_row][_col] > mx )
                                {
                                    mx = cube_input[_bs][_row][_col];
                                    mx_idx = _row;
                                }
                            }
                            matrix_output[_bs][_col] = mx_idx;
                        }
                    return ans;
                },
                [=]<Tensor Tsor>( Tsor const& input, Tsor const& /*output*/, Tsor const& /*grad*/ ) noexcept
                {
                    Tsor& back_ans = context_cast<Tsor>( backward_cache );
                    back_ans.resize( input.shape() );
                    for_each( back_ans.begin(), back_ans.end(), []( auto& v ){ v = 0.0; } ); // always return zero
                    return back_ans;
                },
                "argmax",
                [axis]( std::vector<unsigned long> const& shape ) noexcept
                {
                    std::vector<unsigned long> ans = shape;
                    std::copy( ans.begin()+axis+1, ans.end(), ans.begin()+axis );
                    ans.resize( ans.size() - 1 );
                    return ans;
                },
                [axis]<Expression Self_Expression, Expression Input_Expression>( Self_Expression const& self_expression, Input_Expression const& input_expression ) noexcept
                { // serializer
                    auto const& [input_expression_name, input_expression_code] = serialize( input_expression );
                    std::string const& self_expression_identity = fmt::format( "unary_expression_{}_{}", self_expression.name(), self_expression.id() );
                    std::vector<std::string> self_expression_code = input_expression_code;
                    self_expression_code.emplace_back( fmt::format( "auto {} = {}( {}/*axis*/ )( {} );", self_expression_identity, self_expression.name(), axis, input_expression_name ) );
                    return std::make_tuple( self_expression_identity, self_expression_code );
                }
            )(ex);
        };
    }


    ///
    /// @brief Returns the index with the smallest value across axes of an input tensor.
    /// @code{.cpp}
    /// auto a = variable{ ... };
    /// auto ma = argmin( 1 )( a );
    /// @endcode
    ///
    auto inline argmin( unsigned long axis=0 ) noexcept
    {
        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            return argmax(axis)( -ex );
        };
    }



    ///
    /// @brief Identity operation.
    ///
    template <Expression Ex>
    auto constexpr identity( Ex const& ex ) noexcept
    {
        return ex;
    }


    auto constexpr inline flip( int axis ) noexcept
    {
        return [=]<Expression Ex>( Ex const& ex )
        {
            std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
            std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
            return make_unary_operator( [forward_cache, axis]<Tensor Tsor>( Tsor const& input ) noexcept
                                        {
                                            Tsor& ans = context_cast<Tsor>( forward_cache );
                                            flip( input, axis, ans );
                                            return ans;
                                        },
                                        [backward_cache, axis]<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                        {
                                            Tsor& ans = context_cast<Tsor>( backward_cache );
                                            flip( grad, axis, ans );
                                            return ans;
                                        },
                                        "flip",
                                        identity_output_shape_calculator{},
                                        [axis]<Expression Self_Expression, Expression Input_Expression>( Self_Expression const& self_expression, Input_Expression const& input_expression ) noexcept
                                        { // serializer
                                            auto const& [input_expression_name, input_expression_code] = serialize( input_expression );
                                            std::string const& self_expression_identity = fmt::format( "unary_expression_{}_{}", self_expression.name(), self_expression.id() );
                                            std::vector<std::string> self_expression_code = input_expression_code;
                                            self_expression_code.emplace_back( fmt::format( "auto {} = {}( {}/*axis*/ )( {} );", self_expression_identity, self_expression.name(), axis, input_expression_name ) );
                                            return std::make_tuple( self_expression_identity, self_expression_code );
                                        }
                    )( ex );
        };
    }




    template< Expression Ex >
    auto transpose( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator
        (
            [forward_cache]<Tensor Tsor>( Tsor const& tsor ) noexcept
            {
                better_assert( tsor.ndim() == 2, "Expecting 2D tensor, but got dimensions ", tsor.ndim() );

                typedef typename Tsor::value_type value_type;

                std::vector<unsigned long> const shape = tsor.shape();
                auto const[row, col] = std::make_tuple( shape[0], shape[1] );
                view_2d<value_type> v_in{ tsor.data(), row, col };

                Tsor& ans = context_cast<Tsor>( forward_cache );
                ans.resize( {col, row} );
                view_2d<value_type> v_out{ ans.data(), col, row };

                for ( auto r : range( row ) )
                    for ( auto c : range( col ) )
                        v_out[c][r] = v_in[r][c];

                return ans;
            },
            [backward_cache]<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
            {
                typedef typename Tsor::value_type value_type;

                std::vector<unsigned long> const shape = grad.shape();
                auto const[row, col] = std::make_tuple( shape[0], shape[1] );
                view_2d<value_type> v_in{ grad.data(), row, col };

                Tsor& back_ans = context_cast<Tsor>( backward_cache );
                back_ans.resize( {col, row} );

                view_2d<value_type> v_out{ back_ans.data(), col, row };

                for ( auto r : range( row ) )
                    for ( auto c : range( col ) )
                        v_out[c][r] = v_in[r][c];

                return back_ans;
            },
            "transpose",
            []( std::vector<unsigned long> const& shape ) noexcept
            {
                better_assert( shape.size() == 2, fmt::format( "expecting shape size of 2, but got {}", shape.size() ) );
                return std::vector<unsigned long>{ {shape[1], shape[0]} };
            }
        )( ex );
    }

    auto inline img2col( unsigned long const row_kernel, unsigned long col_kernel=-1,
                         unsigned long const row_padding=0, unsigned long col_padding=0,
                         unsigned long const row_stride=1, unsigned long const col_stride=1,
                         unsigned long const row_dilation=1, unsigned long const col_dilation=1 ) noexcept
    {
        if ( col_kernel == (unsigned long)-1 ) col_kernel = row_kernel;

        std::shared_ptr<std::vector<std::uint32_t>> s_index_record = std::make_shared<std::vector<std::uint32_t>>(); // col_img[idx] = img[index_record[idx]]  -- (-1) for zero padding

        auto img2col_forward = [s_index_record]<Tensor Tsor>
        (
            Tsor const& input_img, Tsor& output_col_mat,
            unsigned long kernel_row, unsigned long kernel_col,
            unsigned long padding_row, unsigned long padding_col,
            unsigned long stride_row, unsigned long stride_col,
            unsigned long dilation_row, unsigned long dilation_col
        ) noexcept
        {
            typedef typename Tsor::value_type value_type;
            std::vector<std::uint32_t>& index_record = *s_index_record; //32 bit should be enough for memory address offeset

            std::vector<unsigned long> input_shape = input_img.shape();
            better_assert( input_shape.size() == 4, "Expecting a 4D tensor." );
            auto const [BS, R, C, CH] = std::make_tuple( input_shape[0], input_shape[1], input_shape[2], input_shape[3] );

            unsigned long const output_row = ( R + 2 * padding_row - ( dilation_row * (kernel_row - 1) + 1 ) ) / stride_row + 1;
            unsigned long const output_col = ( C + 2 * padding_col - ( dilation_col * (kernel_col - 1) + 1 ) ) / stride_col + 1;
            unsigned long const output_column_matrix_row = kernel_row * kernel_col * CH;
            unsigned long const output_column_matrix_col = BS * output_row * output_col;

            output_col_mat.resize( {output_column_matrix_row, output_column_matrix_col} );

            if ( index_record.size() != output_column_matrix_row * output_column_matrix_col ) // first-run?
            {
                index_record.resize( output_column_matrix_row * output_column_matrix_col );

                for ( auto bs : range( BS ) )
                {
                    std::int64_t const col_offset = bs * output_row * output_col * kernel_row * kernel_col * CH;
                    std::int64_t const im_offset = bs * R * C * CH;
                    for ( auto c : range( CH * kernel_row * kernel_col ) )
                    {
                        std::int64_t const w_offset = c % kernel_col;
                        std::int64_t const h_offset = ( c / kernel_col ) % kernel_row;
                        std::int64_t const c_im = c / ( kernel_col * kernel_row );

                        for ( auto h : range( output_row ) )
                        {
                            std::int64_t const im_row_idx = h * stride_row - padding_row + h_offset * dilation_row;
                            for ( auto w : range( output_col ) )
                            {
                                std::int64_t const im_col_idx = w * stride_col - padding_col + w_offset * dilation_col;
                                std::int64_t const im_idx = im_offset+( im_row_idx * C + im_col_idx ) * CH + c_im;
                                std::int64_t const col_idx = col_offset+( c * output_row + h ) * output_col + w;
                                index_record[col_idx] = static_cast<std::uint32_t>((im_row_idx<0 || im_row_idx>=static_cast<std::int64_t>(R) || im_col_idx<0 || im_col_idx>=static_cast<std::int64_t>(C)) ? 0xffffffff : im_idx);
                            }
                        }
                    }
                }
                // re-arrange [bs, new_R, new_C] --> [new_R, new_c*bs]
                {
                    std::vector<std::uint32_t> re_arranged_index;
                    re_arranged_index.resize( index_record.size() );

                    view_3d<std::uint32_t> re_arranged_mat{ re_arranged_index.data(), output_column_matrix_row, BS, output_row*output_col };
                    view_3d<std::uint32_t> index_record_mat{ index_record.data(), BS, output_column_matrix_row, output_row*output_col };

                    for ( auto bs : range( BS ) )
                        for ( auto r : range( output_column_matrix_row ) )
                            for ( auto c : range( output_row*output_col ) )
                                re_arranged_mat[r][bs][c] = index_record_mat[bs][r][c];
                    // overwrite index record
                    std::copy( re_arranged_index.begin(), re_arranged_index.end(), index_record.begin() );
                }
            }

            // fill-in
            for ( auto idx : range( output_col_mat.size() ) )
            {
                auto const index = index_record[idx];
                output_col_mat[idx] = (index == 0xffffffff) ? value_type{0} : input_img[index];
            }
        };

        auto img2col_backward = [s_index_record]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad, Tsor& ans ) noexcept
        {
            typedef typename Tsor::value_type value_type;
            ans.resize( input.shape() );
            std::fill( ans.begin(), ans.end(), value_type{0} );

            std::vector<std::uint32_t>& index_record = *s_index_record; //32 bit should be enough for memory address offeset
            for ( auto idx : range( grad.size() ) )
            {
                auto const index = index_record[idx];
                if ( index != 0xffffffff )
                    ans[index] += grad[idx];
            }
        };

        std::shared_ptr<std::any> output_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> back_grad_cache = std::make_shared<std::any>();

        return [row_kernel, col_kernel, row_padding, col_padding, row_stride, col_stride, row_dilation, col_dilation, img2col_forward, img2col_backward, output_cache, back_grad_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [=]<Tensor Tsor>( Tsor const & tsor ) noexcept
                {
                    Tsor& output = context_cast<Tsor>( output_cache );
                    img2col_forward( tsor, output, row_kernel, col_kernel, row_padding, col_padding, row_stride, col_stride, row_dilation, col_dilation );
                    return Tsor{output};
                },
                [=]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                {
                    Tsor& back_grad = context_cast<Tsor>( back_grad_cache );
                    img2col_backward( input, output, grad, back_grad );
                    return Tsor{back_grad};
                },
                "img2col",
                [=]( std::vector<unsigned long> const& shape ) noexcept
                {
                    better_assert( shape.size() == 4, fmt::format("Expecting a 4D tensor, but got {}.", shape.size()) );
                    auto const [BS, R, C, CH] = std::make_tuple( shape[0], shape[1], shape[2], shape[3] );

                    unsigned long const output_row = ( R + 2 * row_padding - ( row_dilation * (row_kernel - 1) + 1 ) ) / row_stride + 1;
                    unsigned long const output_col = ( C + 2 * col_padding - ( col_dilation * (col_kernel - 1) + 1 ) ) / col_stride + 1;
                    unsigned long const output_column_matrix_row = row_kernel * col_kernel * CH;
                    unsigned long const output_column_matrix_col = BS * output_row * output_col;
                    return std::vector<unsigned long>{ {output_column_matrix_row, output_column_matrix_col} };
                }
            )( ex );
        };
    }

    auto inline conv2d
    (
        unsigned long row_input, unsigned long col_input,
        unsigned long const row_stride=1, unsigned long const col_stride=1,
        unsigned long const row_dilation=1, unsigned long const col_dilation=1,
        std::string const& padding="valid"
    ) noexcept
    {
        // lhs_ex is for one 4D tensor of [BS, R, C, CH]
        // rhs_ex is for NC 4D filter of [1, r, c, CH], thus the shape is [NC, r, c, CH]
        // the output tensor is of shape [BS, .., .., NC]
        //
        // Note: the rhs expression is fixed as a variable, as we need to extract the kernel shape from it
        //
        //return [row_input, col_input, row_stride, col_stride, row_dilation, col_dilation, padding ]<Expression Ex, Variable Va>( Ex const& lhs_ex, Va const& rhs_ex ) noexcept
        return [row_input, col_input, row_stride, col_stride, row_dilation, col_dilation, padding ]<Expression Ex, Expression Ey>( Ex const& lhs_ex, Ey const& rhs_ex ) noexcept
        {
            std::vector<unsigned long> const& shape = rhs_ex.shape();
            better_assert( shape.size() == 4 );
            auto const[new_channel, row_kernel, col_kernel, channel] = std::make_tuple( shape[0], shape[1], shape[2], shape[3] );
            //TODO: optimization in case of small kernels of (1, 1), (3, 3)
            unsigned long row_padding = 0;
            unsigned long col_padding = 0;
            if ( padding == "same" )
            {
                unsigned long const row_padding_total = (row_kernel + (row_kernel - 1) * (row_dilation - 1) - row_stride);
                better_assert( !(row_padding_total & 0x1), "Expecting total row padding to be even, but got ", row_padding_total, " With row input ", row_input, " and row_stride ", row_stride );
                unsigned long const col_padding_total = (col_kernel + (col_kernel - 1) * (col_dilation - 1) - col_stride);
                better_assert( !(col_padding_total & 0x1), "Expecting total col padding to be even, but got ", col_padding_total );
                row_padding = ((row_kernel&1)+row_padding_total) >> 1;
                col_padding = ((col_kernel&1)+col_padding_total) >> 1;
            }

            unsigned long const row_output = ( row_input + 2 * row_padding - ( row_dilation * (row_kernel - 1) + 1 ) ) / row_stride + 1;
            unsigned long const col_output = ( col_input + 2 * row_padding - ( col_dilation * (col_kernel - 1) + 1 ) ) / col_stride + 1;

            auto lhs_ex_as_col = img2col(row_kernel, col_kernel, row_padding, col_padding, row_stride, col_stride, row_dilation, col_dilation)( lhs_ex ); // [BS, R, C, CH] ==> [r*c*CH, BS*new_row*new_col]

            auto rhs_ex_flatten = reshape({row_kernel*col_kernel*channel,})( rhs_ex ); // [NC, r, c, CH] ==> [NC, r*c*CH]

            auto flatten_output = rhs_ex_flatten * lhs_ex_as_col; // [NC, BS * new_row * new_col]

            auto tr_output = transpose( flatten_output ); // [BS*new_row*new_col, NC]

            auto ans = reshape({row_output, col_output, new_channel})( tr_output );

            return ans;
        };
    }


    ///
    /// @brief Conv2D not constrained by the input shape.
    ///
    auto inline general_conv2d
    (
        unsigned long const row_stride=1, unsigned long const col_stride=1,
        unsigned long const row_dilation=1, unsigned long const col_dilation=1,
        std::string const& padding="valid"
    ) noexcept
    {
        // lhs_ex is for one 4D tensor of [BS, R, C, CH]
        // rhs_ex is for NC 4D filter of [1, r, c, CH], thus the shape is [NC, r, c, CH]
        // the output tensor is of shape [BS, .., .., NC]
        //
        // Note: the rhs expression is fixed as a variable, as we need to extract the kernel shape from it
        //
        return [ row_stride, col_stride, row_dilation, col_dilation, padding ]<Expression Ex, Expression Ey>( Ex const& lhs_ex, Ey const& rhs_ex ) noexcept
        {
            auto const& lhs_shape = lhs_ex.shape();
            better_assert( lhs_shape.size() == 4, fmt::format( "expecting lhs_shape size of 4, but got {}", lhs_shape.size() ) );
            auto [_bs, row_input, col_input, _ch] = std::make_tuple( lhs_shape[0], lhs_shape[1], lhs_shape[2], lhs_shape[3] );

            std::vector<unsigned long> const& shape = rhs_ex.shape();
            better_assert( shape.size() == 4 );
            auto const[new_channel, row_kernel, col_kernel, channel] = std::make_tuple( shape[0], shape[1], shape[2], shape[3] );
            unsigned long row_padding = 0;
            unsigned long col_padding = 0;
            if ( padding == "same" )
            {
                unsigned long const row_padding_total = (row_kernel + (row_kernel - 1) * (row_dilation - 1) - row_stride);
                unsigned long const col_padding_total = (col_kernel + (col_kernel - 1) * (col_dilation - 1) - col_stride);
                row_padding = ((row_kernel&1)+row_padding_total) >> 1;
                col_padding = ((col_kernel&1)+col_padding_total) >> 1;
            }

            unsigned long const row_output = ( row_input + 2 * row_padding - ( row_dilation * (row_kernel - 1) + 1 ) ) / row_stride + 1;
            unsigned long const col_output = ( col_input + 2 * row_padding - ( col_dilation * (col_kernel - 1) + 1 ) ) / col_stride + 1;

            auto lhs_ex_as_col = img2col(row_kernel, col_kernel, row_padding, col_padding, row_stride, col_stride, row_dilation, col_dilation)( lhs_ex ); // [BS, R, C, CH] ==> [r*c*CH, BS*new_row*new_col]

            auto rhs_ex_flatten = reshape({row_kernel*col_kernel*channel,})( rhs_ex ); // [NC, r, c, CH] ==> [NC, r*c*CH]

            auto flatten_output = rhs_ex_flatten * lhs_ex_as_col; // [NC, BS * new_row * new_col]

            auto tr_output = transpose( flatten_output ); // [BS*new_row*new_col, NC]

            auto ans = reshape({row_output, col_output, new_channel})( tr_output );

            return ans;
        };
    }

    ///
    /// @brief Conv2D Transpose intemediate layer
    ///
    auto inline conv2d_tranpose_intermediate
    (
        unsigned long const row_kernel, unsigned long const col_kernel,
        unsigned long const row_stride=1, unsigned long const col_stride=1,
        std::string const& padding="valid"
    ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        auto shape_calculator = [=]( std::vector<unsigned long> const& old_shape )
        {
            std::vector<unsigned long> new_shape = old_shape;
            new_shape[1] *= row_stride;
            new_shape[2] *= col_stride;
            if (padding == std::string("valid"))
            {
                new_shape[1] += row_kernel * 2 - 2;
                new_shape[2] += col_kernel * 2 - 2;
            }
            return new_shape;
        };

        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [=]<Tensor Tsor>( Tsor const& input ) noexcept
                {
                    typedef typename Tsor::value_type value_type;
                    std::vector<unsigned long> const& old_shape = input.shape();
                    auto [_bs, old_row, old_col, _ch] = std::make_tuple( old_shape[0], old_shape[1], old_shape[2], old_shape[3] );
                    view_4d<value_type const> input_4d{ input.data(), _bs, old_row, old_col, _ch };

                    std::vector<unsigned long> const& new_shape = shape_calculator( old_shape );
                    auto [bs, new_row, new_col, ch] = std::make_tuple( new_shape[0], new_shape[1], new_shape[2], new_shape[3] );

                    auto ans = context_cast<Tsor>( forward_cache );
                    ans.resize( new_shape );
                    std::fill( ans.begin(), ans.end(), value_type{0} ); // just in case not initialized
                    view_4d<value_type> output_4d{ ans.data(), bs, new_row, new_col, ch };

                    unsigned long row_offset = (padding == std::string{"valid"}) ? (row_kernel-1) : 0;
                    unsigned long col_offset = (padding == std::string{"valid"}) ? (col_kernel-1) : 0;

                    for ( auto bs_index : range( bs ) )
                        for ( auto row_index : range( old_row ) )
                            for ( auto col_index : range( old_col ) )
                                for ( auto ch_index : range( ch ) )
                                    output_4d[bs_index][row_offset+row_index*row_stride][col_offset+col_index*col_stride][ch_index] =
                                     input_4d[bs_index][           row_index][                      col_index][           ch_index];

                    return ans;
                },
                [=]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;
                    std::vector<unsigned long> const& input_shape = input.shape();
                    auto [bs, i_row, i_col, ch] = std::make_tuple( input_shape[0], input_shape[1], input_shape[2], input_shape[3] );
                    auto back_ans = context_cast<Tsor>( backward_cache );
                    back_ans.resize( input_shape );
                    view_4d<value_type> b_4d{ back_ans.data(), bs, i_row, i_col, ch };

                    std::vector<unsigned long> const& output_shape = grad.shape();
                    auto [_bs, o_row, o_col, _ch] = std::make_tuple( output_shape[0], output_shape[1], output_shape[2], output_shape[3] );
                    view_4d<value_type> g_4d{ grad.data(), bs, o_row, o_col, ch };

                    unsigned long row_offset = (padding == std::string{"valid"}) ? (row_kernel-1) : 0;
                    unsigned long col_offset = (padding == std::string{"valid"}) ? (col_kernel-1) : 0;

                    for ( auto bs_index : range( bs ) )
                        for ( auto row_index : range( i_row ) )
                            for ( auto col_index : range( i_col ) )
                                for ( auto ch_index : range( ch ) )
                                     b_4d[bs_index][           row_index][                      col_index][           ch_index] =
                                     g_4d[bs_index][row_offset+row_index*row_stride][col_offset+col_index*col_stride][ch_index];

                    return back_ans;
                },
                "conv2d_transpose_intermediate",
                shape_calculator
                // TODO: a serializer
            )(ex);
        };
    }


    ///
    /// @brief Conv2D Transpose
    ///
    auto inline conv2d_transpose
    (
        unsigned long const row_kernel, unsigned long const col_kernel,
        unsigned long const row_stride=1, unsigned long const col_stride=1,
        unsigned long const row_dilation=1, unsigned long const col_dilation=1,
        std::string const& padding="valid"
    ) noexcept
    {
        // lhs_ex is for one 4D tensor of [BS, R, C, CH]
        // rhs_ex is for NC 4D filter of [1, r, c, CH], thus the shape is [NC, r, c, CH]
        // NOTE: here rhs_ex serves as the transposed convolution kernel, its row and column are inversed. Be careful when importing weights from other places.
        // the output tensor is of shape [BS, .., .., NC]
        //
        // Note: the rhs expression is fixed as a variable, as we need to extract the kernel shape from it
        //
        return [ row_kernel, col_kernel, row_stride, col_stride, row_dilation, col_dilation, padding ]<Expression Ex, Expression Ey>( Ex const& lhs_ex, Ey const& rhs_ex ) noexcept
        {
            //auto new_lhs_ex = conv2d_tranpose_intermediate( row_kernel, col_kernel, row_stride, col_stride, padding )( lhs_ex );
            auto new_lhs_ex = conv2d_tranpose_intermediate( row_kernel, col_kernel, row_stride, col_stride, padding )( lhs_ex );
            return general_conv2d( row_stride, col_stride, row_dilation, col_dilation, padding )( new_lhs_ex, flip(1)(flip(2)(rhs_ex)) );
        };
    }





    template< typename T > requires std::floating_point<T>
    inline auto drop_out( T const factor ) noexcept
    {
        better_assert( factor < T{1}, "Expecting drop out rate less than 1, but got factor = ", factor );
        better_assert( factor > T{0}, "Expecting drop out rate greater than 0, but got factor = ", factor );

        std::shared_ptr<std::any> mask = std::make_shared<std::any>();
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [factor, mask, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [factor, mask, forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                {
                    typedef typename Tsor::value_type value_type;

                    if ( learning_phase == 0 ) // defined in 'config.hpp'
                        return input;

                    std::any& mask_ = *mask;
                    // first run, initialize mask
                    if ( !mask_.has_value() )
                    {
                        Tsor const random_tensor = random<value_type>( input.shape() );
                        Tsor mask__{ input.shape() };
                        for ( auto idx : range( input.size() ) )
                            if ( random_tensor[ idx ] > factor )
                                mask__[ idx ] = 1;
                        mask_ = mask__; // initialize
                    }

                    Tsor& mask__ = std::any_cast<Tsor&>( mask_ );

                    Tsor& ans = context_cast<Tsor>( forward_cache );
                    ans.deep_copy( input );

                    for ( auto idx : range( input.size() ) )
                        ans[idx] *= mask__[idx] / (value_type{1} - factor);
                    return ans;
                },
                [mask, backward_cache]<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                {
                    if ( learning_phase == 0 ) // defined in 'config.hpp'
                        return grad;

                    Tsor& mask__ = std::any_cast<Tsor&>( *mask );

                    Tsor& ans = context_cast<Tsor>( backward_cache );
                    ans.deep_copy( grad );

                    for ( auto idx : range( grad.size() ) )
                        ans[idx] *= mask__[idx];
                    return ans;
                },
                "dropout"
                //TODO: serializer
            )( ex );
        };
    }

    ///
    /// @brief dropout is an alias name of drop_out.
    ///
    template< typename T > requires std::floating_point<T>
    inline auto dropout( T const factor ) noexcept
    {
        return drop_out( factor );
    }


    namespace
    {

        struct max_pooling_2d_context
        {

            auto make_forward() const noexcept
            {
                return  []( unsigned long stride, std::shared_ptr<std::any> mask, std::shared_ptr<std::any> forward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;
                        better_assert( input.ndim() == 4, "Expecting a 4D tensor, but got ", input.ndim() );

                        Tsor& mask__ = context_cast<Tsor>( mask );
                        mask__.resize( input.shape() );


                        std::vector<unsigned long> shape = input.shape();
                        auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
                        Tsor input_ = input;
                        view_4d<value_type> ts{ input_.data(), batch_size, row, col, channel };
                        view_4d<value_type> tm{ mask__.data(), batch_size, row, col, channel };

                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        ans.resize( {batch_size, row/stride, col/stride, channel} );

                        view_4d<value_type> t1{ ans.data(), batch_size, row/stride, col/stride, channel };

                        for ( auto bs : range(batch_size) )
                            for ( auto r : range(row/stride) ) // row for t1
                                for ( auto c : range(col/stride) ) // col for t1
                                    for ( auto ch : range(channel) )
                                    {
                                        unsigned long current_row_max = r * stride;
                                        unsigned long current_col_max = c * stride;
                                        for ( auto _r : range( (r*stride), ((r*stride)+stride) ) ) // row for ts
                                            for ( auto _c : range( (c*stride), ((c*stride)+stride) ) ) // col for ts
                                            {
                                                if ( ts[bs][_r][_c][ch] > ts[bs][current_row_max][current_col_max][ch] )
                                                {
                                                    current_row_max = _r;
                                                    current_col_max = _c;
                                                }
                                            }
                                        tm[bs][current_row_max][current_col_max][ch] = 1.0; //mark as max
                                        t1[bs][r][c][ch] = ts[bs][current_row_max][current_col_max][ch]; // update value
                                    }
                        return ans;
                    };
                };
            }

            auto make_backward() const noexcept
            {
                return []( unsigned long stride, std::shared_ptr<std::any> mask, std::shared_ptr<std::any> backward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;
                        std::vector<unsigned long> const& shape = input.shape();
                        auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);

                        Tsor& mask__ = std::any_cast<Tsor&>( *mask );
                        view_4d<value_type> tm{ mask__.data(), batch_size, row, col, channel };

                        Tsor& ans = context_cast<Tsor>( backward_cache );
                        ans.resize( input.shape() );

                        view_4d<value_type> ta{ ans.data(), batch_size, row, col, channel };

                        Tsor grad_ = grad;
                        view_4d<value_type> tg{ grad_.data(), batch_size, row/stride, col/stride, channel };

                        for ( auto bs : range( batch_size ) )
                            for ( auto r : range( row ) )
                                for ( auto c : range( col ) )
                                    for ( auto ch : range( channel ) )
                                        if ( std::abs(tm[bs][r][c][ch] - 1.0) < 1.0e-5 )
                                            ta[bs][r][c][ch] = tg[bs][r/stride][c/stride][ch];
                        return ans;
                    };
                };
            }

        }; // max_pooling_2d_context

    } // anonymous namespace


    // comment: maybe using function 'reduce' to reduce the cod complexity? at a price of performance?
    inline auto max_pooling_2d( unsigned long stride ) noexcept
    {
        better_assert( stride > 1, "Expecting max_pooling_2d stride greater than 1, but got ", stride );

        std::shared_ptr<std::any> mask = std::make_shared<std::any>();
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [stride, mask, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                max_pooling_2d_context{}.make_forward()( stride, mask, forward_cache ),
                max_pooling_2d_context{}.make_backward()( stride, mask, backward_cache ),
                "max_pooling_2d",
                [=]( std::vector<unsigned long> const& shape ) noexcept
                {
                    better_assert( shape.size()==4, fmt::format( "expecting shape size of 4, but got {}", shape.size() ) );
                    return std::vector<unsigned long>{ {shape[0], shape[1]/stride, shape[2]/stride, shape[3]} };
                }
                // TODO: serializer
            )( ex );
        };
    }

    inline auto average_pooling_2d( unsigned long stride ) noexcept
    {
        better_assert( stride > 1, "Expecting average_pooling_2d stride greater than 1, but got ", stride );

        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [stride, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [stride, forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept // [BS, R, C, CH] --> [BS, R/s, C/s, CH]
                {
                    typedef typename Tsor::value_type value_type;
                    better_assert( input.ndim() == 4, "Expecting a 4D tensor, but got ", input.ndim() );

                    std::vector<unsigned long> shape = input.shape();
                    auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
                    Tsor input_ = input;
                    view_4d<value_type> ts{ input_.data(), batch_size, row, col, channel };

                    Tsor& ans = context_cast<Tsor>( forward_cache );
                    ans.resize( {batch_size, row/stride, col/stride, channel} );
                    std::fill( ans.begin(), ans.end(), value_type{0} );

                    view_4d<value_type> t1{ ans.data(), batch_size, row/stride, col/stride, channel };

                    value_type const factor = value_type{1} / static_cast<value_type>(stride*stride);
                    for ( auto bs : range(batch_size) )
                        for ( auto r : range(row/stride) ) // row for t1
                            for ( auto c : range(col/stride) ) // col for t1
                                for ( auto ch : range(channel) )
                                    for ( auto _r : range( (r*stride), ((r*stride)+stride) ) ) // row for ts
                                        for ( auto _c : range( (c*stride), ((c*stride)+stride) ) ) // col for ts
                                            t1[bs][r][c][ch] += ts[bs][_r][_c][ch] * factor;
                    return ans;
                },
                [stride, backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;
                    std::vector<unsigned long> const& shape = input.shape();
                    auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);

                    Tsor& ans = context_cast<Tsor>( backward_cache );
                    ans.resize( input.shape() );

                    view_4d<value_type> ta{ ans.data(), batch_size, row, col, channel };

                    Tsor grad_ = grad;
                    view_4d<value_type> tg{ grad_.data(), batch_size, row/stride, col/stride, channel };

                    value_type const factor = value_type{1} / static_cast<value_type>(stride*stride);
                    for ( auto bs : range( batch_size ) )
                        for ( auto r : range( row ) )
                            for ( auto c : range( col ) )
                                for ( auto ch : range( channel ) )
                                    ta[bs][r][c][ch] = factor * tg[bs][r/stride][c/stride][ch];
                    return ans;
                },
                "average_pooling_2d",
                [=]( std::vector<unsigned long> const& shape ) noexcept
                {
                    better_assert( shape.size()==4, fmt::format( "expecting shape size of 4, but got {}", shape.size() ) );
                    return std::vector<unsigned long>{ {shape[0], shape[1]/stride, shape[2]/stride, shape[3]} };
                }
                // serializer
            )( ex );
        };
    }

    namespace
    {
        struct up_sampling_2d_context
        {
            auto make_forward() const noexcept
            {
                return []( unsigned long stride, std::shared_ptr<std::any> forward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;
                        better_assert( input.ndim() == 4, "Expecting a 4D tensor, but got ", input.ndim() );

                        std::vector<unsigned long> shape = input.shape();
                        auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
                        Tsor input_ = input;
                        view_4d<value_type> ts{ input_.data(), batch_size, row, col, channel };

                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        ans.resize( {batch_size, row*stride, col*stride, channel} );
                        std::fill( ans.begin(), ans.end(), value_type{0} );

                        view_4d<value_type> t1{ ans.data(), batch_size, row*stride, col*stride, channel };

                        for ( auto bs : range(batch_size) )
                            for ( auto r : range(row) ) // row for ts
                                for ( auto c : range(col) ) // col for ts
                                    for ( auto ch : range(channel) )
                                        for ( auto _r : range( (r*stride), ((r*stride)+stride) ) ) // row for t1
                                            for ( auto _c : range( (c*stride), ((c*stride)+stride) ) ) // col for t1
                                                t1[bs][_r][_c][ch] = ts[bs][r][c][ch];
                        return ans;
                    };
                };
            }

            auto make_backward() const noexcept
            {
                return []( unsigned long stride, std::shared_ptr<std::any> backward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;
                        std::vector<unsigned long> const& shape = input.shape();
                        auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);

                        Tsor& ans = context_cast<Tsor>( backward_cache );
                        ans.resize( input.shape() );
                        std::fill( ans.begin(), ans.end(), value_type{0} );

                        view_4d<value_type> ta{ ans.data(), batch_size, row, col, channel };

                        Tsor grad_ = grad;
                        view_4d<value_type> tg{ grad_.data(), batch_size, row*stride, col*stride, channel };

                        for ( auto bs : range( batch_size ) )
                            for ( auto r : range( row ) )
                                for ( auto c : range( col ) )
                                    for ( auto ch : range( channel ) )
                                        for ( auto _r : range( (r*stride), ((r*stride)+stride) ) ) // row for tg
                                            for ( auto _c : range( (c*stride), ((c*stride)+stride) ) ) // col for tg
                                                ta[bs][r][c][ch] += tg[bs][_r][_c][ch];
                        return ans;
                    };
                };
            }
        }; // up_sampling_2d_context

    } // anonymous namespace

    inline auto up_sampling_2d( unsigned long stride ) noexcept
    {
        better_assert( stride > 1, "Expecting up_sampling_pooling_2d stride greater than 1, but got ", stride );

        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [stride, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                up_sampling_2d_context{}.make_forward()( stride, forward_cache ),
                up_sampling_2d_context{}.make_backward()( stride, backward_cache ),
                "up_sampling_2d",
                [=]( std::vector<unsigned long> const& shape ) noexcept
                {
                    better_assert( shape.size()==4, fmt::format( "expecting shape size of 4, but got {}", shape.size() ) );
                    return std::vector<unsigned long>{ {shape[0], shape[1]*stride, shape[2]*stride, shape[3]} };
                }
                //TODO: serializer
            )( ex );
        };
    }

    // an alias name
    inline auto upsampling_2d( unsigned long stride ) noexcept
    {
        return up_sampling_2d( stride );
    }



    template< typename T=double > requires std::floating_point<T>
    inline auto normalization_batch( T const momentum=0.98 ) noexcept
    {
        std::shared_ptr<std::any> global_average_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> global_variance_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> average_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> variance_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [=]<Tensor Tsor>( Tsor const& input ) noexcept
                {
                    better_assert( input.ndim() > 1, "normalization_batch requires input dimension at least 2, got ", input.ndim() );

                    typedef typename Tsor::value_type value_type;
                    //typedef typename Tsor::allocator allocator;

                    std::vector<unsigned long> const& shape = input.shape();
                    unsigned long const channels = *(shape.rbegin());
                    unsigned long const rest_dims = input.size() / channels;

                    view_2d<value_type> input_{ input.data(), rest_dims, channels };

                    // case of prediction phase, in this phase, the batch size could be 1, and it is not possible to calculate the variance
                    if ( learning_phase == 0 ) // defined in 'config.hpp'
                    {
                        // fix for the special case when prediction is executed before the training, typically in a GAN
                        Tsor& global_average_test = context_cast<Tsor>( global_average_cache );
                        if ( global_average_test.empty() )
                            return input;

                        // normal case. i.e., the global_average_cache and global_variance_cache are not empty
                        Tsor& global_average = context_extract<Tsor>( global_average_cache );
                        Tsor& global_variance = context_extract<Tsor>( global_variance_cache );

                        Tsor& ans = context_cast<Tsor>( forward_cache, zeros_like( input ) );
                        ans.resize( input.shape() ); // well, the batch sizes for training and for prediction are not necessarily same

                        view_2d<value_type> ans_{ ans.data(), rest_dims, channels };
                        {
                            for ( auto r : range( rest_dims ) )
                                for ( auto c : range( channels ) )
                                    ans_[r][c] = (input_[r][c] - global_average[c]) / std::sqrt( global_variance[c] + eps );
                        }
                        return ans;
                    }

                    //calculate average along the last channel
                    Tsor& average = context_cast<Tsor>( average_cache );
                    {
                        average.resize( {channels, } );
                        std::fill( average.begin(), average.end(), value_type{0} );

                        for ( auto idx : range( rest_dims ) )
                            for ( auto jdx : range( channels ) )
                                average[jdx] += input_[idx][jdx];

                        average /= static_cast<value_type>(rest_dims);
                    }

                    //calculate Variance along the last channel
                    Tsor& variance = context_cast<Tsor>( variance_cache );
                    {
                        variance.resize( {channels,} );
                        std::fill( variance.begin(), variance.end(), value_type{0} );
                        for ( auto idx : range( rest_dims ) )
                            for ( auto jdx : range( channels ) )
                                variance[jdx] += std::pow( input_[idx][jdx] - average[jdx], 2 );

                        variance /= static_cast<value_type>( rest_dims );
                    }


                    Tsor& ans = context_cast<Tsor>( forward_cache );
                    ans.resize( input.shape() ); // the batch sizes for training and for prediction are not necessarily same
                    view_2d<value_type> ans_{ ans.data(), rest_dims, channels };
                    {
                        for ( auto idx : range( rest_dims ) )
                            for ( auto jdx : range( channels ) )
                                ans_[idx][jdx] = ( input_[idx][jdx] - average[jdx] ) / std::sqrt( variance[jdx] + eps );
                    }

                    // update global average and global variance
                    {
                        Tsor& global_average = context_cast<Tsor>( global_average_cache, zeros_like( average ) );
                        // Note: No obvious different is observed between initializing global_variance to zeros and to ones with MNIST example:
                        //       initializing global_variance to zeros, after 10 epochs mnist gives an error of 0.026
                        //       initializing global_variance to ones, after 10 epochs mnist gives an error of 0.028
                        Tsor& global_variance = context_cast<Tsor>( global_variance_cache, zeros_like( variance ) );
                        //Tsor& global_variance = context_cast<Tsor>( global_variance_cache, ones_like( variance ) );
                        for ( auto idx : range( global_average.size() ) )
                        {
                            global_average[idx] = global_average[idx] * momentum + average[idx] * ( 1.0 - momentum );
                            global_variance[idx] = global_variance[idx] * momentum + variance[idx] * ( 1.0 - momentum );
                        }
                    }

                    return ans;
                },

                [=]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;
                    Tsor& variance = context_extract<Tsor>( variance_cache );

                    std::vector<unsigned long> const& shape = input.shape();
                    unsigned long const channels = *(shape.rbegin());
                    unsigned long const rest_dims = input.size() / channels;

                    Tsor& ans = context_cast<Tsor>( backward_cache, zeros_like( input ) );
                    view_2d<value_type> ans_{ans.data(), rest_dims, channels };
                    view_2d<value_type> grad_{grad.data(), rest_dims, channels };
                    for ( auto r : range( rest_dims ) )
                        for ( auto c : range( channels ) )
                            ans_[r][c] = grad_[r][c] / std::sqrt( variance[c] + eps );
                    return ans;
                },
                "normalization_batch"
                // TODO: normalizer
            )( ex );
        };
    }



    template< typename T > requires std::floating_point<T>
    inline auto batch_normalization( T const momentum=0.98 ) noexcept
    {
        return [=]<Expression Ex, Variable Va>( Ex const& ex, Va const& gamma, Va const& beta ) noexcept
        {
            return elementwise_product( normalization_batch(momentum)(ex), gamma ) + beta; // multiply and sum along the batch: normalization is of shape [BS, R, C, CH], gamma/beta are of shape [R, C, CH]
        };
    }



    //
    //  example:
    //
    //      variable<tensor<float>> a {... };
    //      variable<tensor<float>> b {... };
    //      auto cab = concatenate( a, b )();
    //
    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr concatenate( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return [&]( unsigned long axe = -1 ) noexcept
        {
            return make_binary_operator
            (
                [axe]<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
                {
                    return concatenate( lhs_tensor, rhs_tensor, axe );
                },
                [axe]<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const grad ) noexcept
                {
                    typedef typename Tsor::value_type value_type;

                    Tsor l_ans{ lhs_input.shape() };
                    Tsor r_ans{ rhs_input.shape() };
                    better_assert(  l_ans.size() + r_ans.size() == grad.size(), "size mismatch: lhs size is ", l_ans.size(), " rhs size is ", r_ans.size(), " and grad size is ", grad.size(),
                                    " with lhs dim is ", l_ans.ndim(), "  and rhs dim is ", r_ans.ndim() );

                    // 2D view of grad
                    unsigned long const ax = (axe == (unsigned long)(-1)) ? grad.ndim()-1 : axe;
                    unsigned long const g_col = std::accumulate( grad.shape().begin()+ax, grad.shape().end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } );
                    unsigned long const g_row = grad.size() / g_col;
                    view_2d<value_type> v_g{ grad.data(), g_row, g_col };

                    // 2D view of l_ans
                    unsigned long const lhs_row = g_row;
                    unsigned long const lhs_col = lhs_input.size() / lhs_row;
                    view_2d<value_type> v_l{ l_ans.data(), lhs_row, lhs_col };

                    // 2D view of r_ans
                    unsigned long const rhs_row = g_row;
                    unsigned long const rhs_col = rhs_input.size() / rhs_row;
                    view_2d<value_type> v_r{ r_ans.data(), rhs_row, rhs_col };

                    better_assert( g_col == lhs_col + rhs_col, "last dimension not agree" );

                    for ( unsigned long idx = 0; idx != g_row; ++idx )
                    {
                        std::copy( v_g[idx], v_g[idx]+lhs_col, v_l[idx] );          // fill idx-th row of 'v_l'
                        std::copy( v_g[idx]+lhs_col, v_g[idx]+g_col, v_r[idx] );    // fill idx-th row of 'v_r'
                    }

                    return std::make_tuple( l_ans, r_ans );
                },
                "concatenate",
                [axe]( std::vector<unsigned long> const& l, std::vector<unsigned long> const& r ) noexcept
                {
                    better_assert( l.size() == r.size(), fmt::format( "expecting of same size, but lhs.size is {} and rhs.size is {}.", l.size(), r.size() ) );
                    // more assertion ?
                    std::vector<unsigned long> ans = l;
                    if ( axe > ans.size() ) axe = ans.size() - 1;
                    ans[axe] += r[axe];
                    return ans;
                }
                // TODO serializer
            )( lhs_ex, rhs_ex );
        };
    }

    // just to keep this interface agrees with Keras
    inline auto concatenate( unsigned long axe = -1 )
    {

        return [=]< Expression Lhs_Expression, Expression Rhs_Expression >( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
        {
            return concatenate( lhs_ex, rhs_ex )( axe );
        };
    }

    // alias of 'concatenate'
    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr concat( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return concatenate( lhs_ex, rhs_ex )();
    }

    // alias of 'concatenate'
    inline auto concat( unsigned long axe = -1 )
    {
        return concatenate( axe );
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr maximum( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> mask_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache_lhs = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache_rhs = std::make_shared<std::any>();
        return make_binary_operator
        (
            [=]<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
            {
                better_assert( lhs_tensor.shape() == rhs_tensor.shape(), "tensor shape mismatch." );

                Tsor& ans = context_cast<Tsor>( forward_cache );
                ans.resize( lhs_tensor.shape() );
                Tsor& mask = context_cast<Tsor>( mask_cache ); // 1 if lhs element is larger, 0 if rhs element is larger
                mask.resize( lhs_tensor.shape() );

                for_each( lhs_tensor.begin(), lhs_tensor.end(), rhs_tensor.begin(), ans.begin(), mask.begin(), []( auto const l, auto const r, auto& a, auto& m ) { m = l > r ? 1.0 : 0.0; a = l > r ? l : r; } );

                return ans;
            },
            [=]<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& grad ) noexcept
            {
                Tsor& mask = context_cast<Tsor>( mask_cache ); // 1 if lhs element is larger, 0 if rhs element is larger

                Tsor& l_ans = context_cast<Tsor>( backward_cache_lhs );
                l_ans.resize( lhs_input.shape() );
                Tsor& r_ans = context_cast<Tsor>( backward_cache_rhs );
                r_ans.resize( rhs_input.shape() );

                for_each( grad.begin(), grad.end(), mask.begin(), l_ans.begin(), r_ans.begin(), []( auto const g, auto const m, auto& l, auto& r ) { if ( m > 0.5 ) { l = g; r = 0.0; } else { l = 0.0; r = g; } } );

                return std::make_tuple( l_ans, r_ans );
            },
            "maximum"
        )( lhs_ex, rhs_ex );
    }

    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr minimum( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> mask_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache_lhs = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache_rhs = std::make_shared<std::any>();
        return make_binary_operator
        (
            [=]<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
            {
                better_assert( lhs_tensor.shape() == rhs_tensor.shape(), "tensor shape mismatch." );

                Tsor& ans = context_cast<Tsor>( forward_cache );
                ans.resize( lhs_tensor.shape() );
                Tsor& mask = context_cast<Tsor>( mask_cache ); // 1 if lhs element is larger, 0 if rhs element is larger
                mask.resize( lhs_tensor.shape() );

                for_each( lhs_tensor.begin(), lhs_tensor.end(), rhs_tensor.begin(), ans.begin(), mask.begin(), []( auto const l, auto const r, auto& a, auto& m ) { m = l > r ? 0.0: 1.0 ; a = l > r ? r: l; } );

                return ans;
            },
            [=]<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& grad ) noexcept
            {
                Tsor& mask = context_cast<Tsor>( mask_cache ); // 1 if lhs element is larger, 0 if rhs element is larger

                Tsor& l_ans = context_cast<Tsor>( backward_cache_lhs );
                l_ans.resize( lhs_input.shape() );
                Tsor& r_ans = context_cast<Tsor>( backward_cache_rhs );
                r_ans.resize( rhs_input.shape() );

                for_each( grad.begin(), grad.end(), mask.begin(), l_ans.begin(), r_ans.begin(), []( auto const g, auto const m, auto& l, auto& r ) { if ( m < 0.5 ) { l = g; r = 0.0; } else { l = 0.0; r = g; } } );

                return std::make_tuple( l_ans, r_ans );
            },
            "minmum"
        )( lhs_ex, rhs_ex );
    }

    ///
    /// @brief Computes the arc tangent of y/x using the signs of arguments to determine the correct quadrant.
    ///
    template< Expression Lhs_Expression, Expression Rhs_Expression >
    auto constexpr atan2( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache_lhs = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache_rhs = std::make_shared<std::any>();
        return make_binary_operator
        (
            [=]<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
            {
                better_assert( lhs_tensor.shape() == rhs_tensor.shape(), "tensor shape mismatch." );
                Tsor& ans = context_cast<Tsor>( forward_cache );
                ans.resize( lhs_tensor.shape() );
                for_each( lhs_tensor.begin(), lhs_tensor.end(), rhs_tensor.begin(), ans.begin(), []( auto const l, auto const r, auto& a ) { a = std::atan2(l, r); } );
                return ans;
            },
            [=]<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& grad ) noexcept
            {
                Tsor& l_ans = context_cast<Tsor>( backward_cache_lhs );
                l_ans.resize( lhs_input.shape() );
                Tsor& r_ans = context_cast<Tsor>( backward_cache_rhs );
                r_ans.resize( rhs_input.shape() );
                for_each( grad.begin(), grad.end(), l_ans.begin(), r_ans.begin(), lhs_input.begin(), rhs_input.begin(), []( auto const g, auto& l, auto& r, auto const x, auto const y ) { auto const c = x*x+y*y; l = -g*y/c; r = g*x/c; } );
                return std::make_tuple( l_ans, r_ans );
            },
            "atan2"
        )( lhs_ex, rhs_ex );
    }


    ///
    /// `random_normal_like` produces random tensor from a normal distribution
    /// @param mean Mean of the normal distribution, a scalar.
    /// @param stddev Standard deviation of the normal distribution, a scalar.
    /// @return An unary operator that takes an unary operator, and producing output tensor from a normal distribution. The shape of the output tensor has the same shape corresponding to the input unary operator.
    ///
    /// Example Code
    /// @code
    /// auto va = variable{ ones<float>({3, 3, 3}) };
    /// auto v_rand = random_normal_like( 1.0, 4.0 )( va ); // this expression will produces a tensor of shape (3, 3, 3) from a normal distribution with parameters (1.0, 4.0)
    /// @endcode
    ///
    template< typename T=float > requires std::floating_point<T>
    inline auto random_normal_like( T mean = 0.0, T stddev = 1.0 ) noexcept
    {
        return [=]<Expression Ex>(Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                [=]<Tensor Tsor>( Tsor const& tsor ) noexcept
                {
                    return randn_like( tsor, mean, stddev );
                },
                []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                {
                    return zeros_like( grad );
                },
                "random_normal_like"
            )(ex);
        };
    }

    ///
    /// `ones_like` produces a tensor of the same shape as the input expression, but with every element to be 1.
    /// @return An unary operator that takes an unary operator, and producing an output tensor
    /// Example Code:
    /// @code
    /// auto va = variable{ ones<float>({3, 3, 3}) };
    /// auto v_rand = ones_like( va ); // this expression will produces a tensor of shape (3, 3, 3), with every element to be 1.
    /// @endcode
    ///
    template< Expression Ex>
    auto ones_like( Ex const& ex ) noexcept
    {
        return make_unary_operator
        (
            []<Tensor Tsor>( Tsor const& tsor ) noexcept { return ones_like( tsor ); },
            []<Tensor Tsor>( Tsor const&, Tsor const& , Tsor const& grad ) noexcept { return zeros_like( grad ); },
            "ones_like"
        )(ex);
    }

    ///
    /// `zeros_like` produces a tensor of the same shape as the input expression, but with every element to be 0.
    /// @return An unary operator that takes an unary operator, and producing an output tensor
    /// Example Code:
    /// @code
    /// auto va = variable{ ones<float>({3, 3, 3}) };
    /// auto v_rand = zeros_like( va ); // this expression will produces a tensor of shape (3, 3, 3), with every element to be 0.
    /// @endcode
    ///
    template< Expression Ex>
    auto zeros_like( Ex const& ex ) noexcept
    {
        return make_unary_operator
        (
            []<Tensor Tsor>( Tsor const& tsor ) noexcept { return zeros_like( tsor ); },
            []<Tensor Tsor>( Tsor const&, Tsor const& , Tsor const& grad ) noexcept { return zeros_like( grad ); },
            "zeros_like"
        )(ex);
    }

    ///
    /// Returns the truth value of (lhs == rhs) element-wise. [+1 for true, 0 for false]
    ///
    /// @param lhs_ex The first operator.
    /// @param rhs_ex The second operator.
    /// @return An instance of a binary operator that evaluate the element-wise equality of two input operators.
    ///
    /// Example code:
    /// @code
    /// auto l = variable<tensor<float>>{ /*...*/ };
    /// auto r = place_holder<tensor<float>>{};
    /// auto eq = equal(l, r);
    /// @endcode
    ///
    template< Expression Lhs_Expression, Expression Rhs_Expression, std::floating_point FP >
    auto constexpr equal( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex, FP threshold=0.5 ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_binary_operator
        (
            [=]<Tensor Tsor>( Tsor const& lhs_tensor, Tsor const& rhs_tensor ) noexcept
            {
                typedef typename Tsor::value_type value_type;
                better_assert( lhs_tensor.shape() == rhs_tensor.shape(), "equal: tensor shape mismatch." );

                Tsor& ans = context_cast<Tsor>( forward_cache );
                ans.resize( lhs_tensor.shape() );
                for_each( lhs_tensor.begin(), lhs_tensor.end(), rhs_tensor.begin(), ans.begin(), [threshold]( auto l, auto r, auto& v ){ v = (std::abs(l-r) > threshold) ? value_type{0} : value_type{1}; } );
                return ans;
            },
            [=]<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& grad ) noexcept
            {
                typedef typename Tsor::value_type value_type;
                Tsor& ans = context_cast<Tsor>( backward_cache );
                std::fill( ans.begin(), ans.end(), value_type{0} );
                return std::make_tuple( ans, ans );
            },
            "equal"
        )( lhs_ex, rhs_ex );
    }

    ///
    /// Returns the sign. [1 for positive, 0 for 0 and -1 for negative]
    ///
    /// @param ex The input operator.
    /// @return An instance of a unary_operator that evaluate the sign of the input operator.
    ///
    /// Example code:
    /// @code
    /// auto e = variable<tensor<float>>{ /*...*/ };
    /// auto si = sign(e);
    /// @endcode
    ///
    template <Expression Ex>
    auto constexpr sign( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator
        (
            [=]<Tensor Tsor>( Tsor const& input ) noexcept
            {
                typedef typename Tsor::value_type value_type;
                Tsor& ans = context_cast<Tsor>( forward_cache );
                ans.resize( input.shape() );
                for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ){ v = (value_type{0} < x) - (x < value_type{0}); } );
                return ans;
            },
            [=]<Tensor Tsor>( Tsor const&input, Tsor const&, Tsor const& grad ) noexcept
            {
                typedef typename Tsor::value_type value_type;
                Tsor& ans = context_cast<Tsor>( backward_cache );
                ans.resize( input.shape() );
                std::fill( ans.begin(), ans.end(), value_type{0} ); //TF gives zeros, we follow TF here
                return ans;
            },
            "sign"
        )( ex );
    };



    namespace
    {
        struct zero_padding_2d_context
        {
            auto make_forward() const noexcept
            {
                return []( unsigned long top, unsigned long bottom, unsigned long left, unsigned long right, std::shared_ptr<std::any> forward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;
                        better_assert( input.ndim() == 4, "Expecting a 4D tensor, but got ", input.ndim() );

                        // 4D view of input tensor
                        std::vector<unsigned long> shape = input.shape();
                        auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
                        Tsor input_ = input;
                        view_4d<value_type> ts{ input_.data(), batch_size, row, col, channel };

                        // 4D view of output tensor
                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        ans.resize( {batch_size, top+row+bottom, left+col+right, channel} );
                        view_4d<value_type> ta{ ans.data(), batch_size, top+row+bottom, left+col+right, channel };

                        for ( auto bs : range( batch_size ) )
                            for ( auto r : range( row ) )
                                for ( auto c : range( col ) )
                                    for ( auto ch : range( channel ) )
                                        ta[bs][top+r][left+c][ch] = ts[bs][r][c][ch];

                        return ans;
                    };
                };
            }

            auto make_backward() const noexcept
            {
                return []( unsigned long top, unsigned long bottom, unsigned long left, unsigned long right, std::shared_ptr<std::any> backward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;
                        std::vector<unsigned long> const& shape = input.shape();
                        auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);

                        Tsor& ans = context_cast<Tsor>( backward_cache );
                        ans.resize( input.shape() );
                        std::fill( ans.begin(), ans.end(), value_type{0} );

                        view_4d<value_type> ta{ ans.data(), batch_size, row, col, channel };

                        Tsor grad_ = grad;
                        view_4d<value_type> tg{ grad_.data(), batch_size, top+row+bottom, left+col+right, channel };

                        for ( auto bs : range( batch_size ) )
                            for ( auto r : range( row ) )
                                for ( auto c : range( col ) )
                                    for ( auto ch : range( channel ) )
                                        ta[bs][r][c][ch] = tg[bs][r+top][c+left][ch];
                        return ans;
                    };
                };
            }
        }; // zero_padding_2d_context
    }//anonymouse namespace

    ///
    /// @brief Zero-padding layer for 2D input. The input should have 4-dimensions: `(batch_size, row, col, channel)`. The output has 4-dimensions: `(batch_size, new_row, new_col, channel)`.
    /// @param padding If a single integer, then apply symmetric padding to height and width. If two integers, then first is for height and the second is for width. If four integers, then is intepreted as`(top_pad, bottom_pad, left_pad, right_pad)`.
    ///
    /// Example code:
    ///
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {16, 16, 3} ) };
    /// auto b = zero_padding_2d( {8,} )( a ); // shape for b is (8+16+8, 8+16+8, 3)
    /// auto c = zero_padding_2d( {8, 4} )( a ); // shape for c is (8+16+8, 4+16+4, 3)
    /// auto d = zero_padding_2d( {8, 4, 2, 1} )( a ); // shape for d is (8+16+4, 2+16+1, 3)
    /// \endcode
    ///
    inline auto zero_padding_2d( std::vector<unsigned long> const& padding ) noexcept
    {
        // extracting paddings
        unsigned long top, bottom, left, right;
        if ( padding.size() == 1 )
            std::tie( top, bottom, left, right ) = std::make_tuple( padding[0], padding[0], padding[0], padding[0] );
        else if (padding.size() == 2 )
            std::tie( top, bottom, left, right ) = std::make_tuple( padding[0], padding[0], padding[1], padding[1] );
        else if (padding.size() == 4 )
            std::tie( top, bottom, left, right ) = std::make_tuple( padding[0], padding[1], padding[2], padding[3] );
        else
            better_assert( false, "Expecting padding has size of 1, 2 or 4, but got: ", padding.size() );

        // checking extracted paddings
        better_assert( top >= 1, "Expecting zero_padding_2d top padding no less than 1, but got ", top );
        better_assert( bottom >= 1, "Expecting zero_padding_2d bottom padding no less than 1, but got ", bottom );
        better_assert( left >= 1, "Expecting zero_padding_2d left padding no less than 1, but got ", left );
        better_assert( right >= 1, "Expecting zero_padding_2d right padding no less than 1, but got ", right );

        // to avoid re-allocating memory for tensors
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [top, bottom, left, right, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                zero_padding_2d_context{}.make_forward()( top, bottom, left, right, forward_cache ),
                zero_padding_2d_context{}.make_backward()( top, bottom, left, right, backward_cache ),
                "zero_padding_2d",
                [=]( std::vector<unsigned long> const& shape ) noexcept { return std::vector<unsigned long>{ {shape[0], shape[1]+top+bottom, shape[2]+left+right, shape[3]} }; }
            )( ex );
        };
    }



    namespace
    {
        struct cropping_2d_context
        {
            auto make_forward() const noexcept
            {
                return []( unsigned long top, unsigned long bottom, unsigned long left, unsigned long right, std::shared_ptr<std::any> forward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;
                        better_assert( input.ndim() == 4, "Expecting a 4D tensor, but got ", input.ndim() );
                        // check shape, not too large

                        // 4D view of input tensor
                        std::vector<unsigned long> shape = input.shape();
                        auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
                        Tsor input_ = input;
                        view_4d<value_type> ts{ input_.data(), batch_size, row, col, channel };

                        better_assert( row-top-bottom > 0, fmt::format("Cropping2D: expecting a smaller cropping dimension in row: row:{}, top:{}, bottop:{}", row, top, bottom ) );
                        better_assert( col-left-right > 0, fmt::format("Cropping2D: expecting a smaller cropping dimension in col: col:{}, left:{}, right:{}", col, left, right ) );

                        // 4D view of output tensor
                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        ans.resize( {batch_size, row-top-bottom, col-left-right, channel} );
                        view_4d<value_type> ta{ ans.data(), batch_size, row-top-bottom, col-left-right, channel };

                        for ( auto bs : range( batch_size ) )
                            for ( auto r : range( row-top-bottom ) )
                                for ( auto c : range( col-left-right ) )
                                    for ( auto ch : range( channel ) )
                                        ta[bs][r][c][ch] = ts[bs][top+r][left+c][ch];

                        return ans;
                    };
                };
            }

            auto make_backward() const noexcept
            {
                return []( unsigned long top, unsigned long bottom, unsigned long left, unsigned long right, std::shared_ptr<std::any> backward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;
                        std::vector<unsigned long> const& shape = grad.shape();
                        auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);

                        Tsor& ans = context_cast<Tsor>( backward_cache );
                        ans.resize( input.shape() );
                        std::fill( ans.begin(), ans.end(), value_type{0} );

                        view_4d<value_type> ta{ ans.data(), batch_size, row+top+bottom, col+left+right, channel };

                        Tsor grad_ = grad;
                        view_4d<value_type> tg{ grad_.data(), batch_size, row, col, channel };

                        for ( auto bs : range( batch_size ) )
                            for ( auto r : range( row ) )
                                for ( auto c : range( col ) )
                                    for ( auto ch : range( channel ) )
                                        ta[bs][r+top][c+left][ch] = tg[bs][r][c][ch];
                        return ans;
                    };
                };
            }
        }; // cropping_2d_context
    }//anonymouse namespace

    ///
    /// @brief Cropping layer for 2D input. The input should have 4-dimensions: `(batch_size, row, col, channel)`. The output has 4-dimensions: `(batch_size, new_row, new_col, channel)`.
    /// @param padding If a single integer, then apply symmetric cropping to height and width. If two integers, then first is for height and the second is for width. If four integers, then is intepreted as`(top_crop, bottom_crop, left_crop, right_crop)`.
    ///
    /// Example code:
    ///
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {32, 32, 3} ) };
    /// auto b = cropping_2d( {8,} )( a ); // shape for b is (32-8-8, 32-8-8, 3)
    /// auto c = cropping_2d( {8, 4} )( a ); // shape for c is (32-8-8, 32-4-4, 3)
    /// auto d = cropping_2d( {8, 4, 2, 1} )( a ); // shape for d is (32-8-4, 32-2-1, 3)
    /// \endcode
    ///
    inline auto cropping_2d( std::vector<unsigned long> const& padding ) noexcept
    {
        // extracting paddings
        unsigned long top, bottom, left, right;
        if ( padding.size() == 1 )
            std::tie( top, bottom, left, right ) = std::make_tuple( padding[0], padding[0], padding[0], padding[0] );
        else if (padding.size() == 2 )
            std::tie( top, bottom, left, right ) = std::make_tuple( padding[0], padding[0], padding[1], padding[1] );
        else if (padding.size() == 4 )
            std::tie( top, bottom, left, right ) = std::make_tuple( padding[0], padding[1], padding[2], padding[3] );
        else
            better_assert( false, "Expecting padding has size of 1, 2 or 4, but got: ", padding.size() );

        // checking extracted paddings
        better_assert( top >= 1, "Expecting cropping_2d top padding no less than 1, but got ", top );
        better_assert( bottom >= 1, "Expecting cropping_2d bottom padding no less than 1, but got ", bottom );
        better_assert( left >= 1, "Expecting cropping_2d left padding no less than 1, but got ", left );
        better_assert( right >= 1, "Expecting cropping_2d right padding no less than 1, but got ", right );

        // to avoid re-allocating memory for tensors
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [top, bottom, left, right, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                cropping_2d_context{}.make_forward()( top, bottom, left, right, forward_cache ),
                cropping_2d_context{}.make_backward()( top, bottom, left, right, backward_cache ),
                "cropping_2d",
                [=]( std::vector<unsigned long> const& shape ) noexcept
                {
                    return std::vector<unsigned long>{ {shape[0], shape[1]-top-bottom, shape[2]-left-right, shape[3]} };
                }
                //TODO: serializer
            )( ex );
        };
    }


    //namespace
    //{

        inline auto detailed_sliding_2d( unsigned long const pixels, std::shared_ptr<std::any> shift_cache,
                                         std::shared_ptr<std::any> forward_cache, std::shared_ptr<std::any> backward_cache) noexcept
        {
            return [=]<Expression Ex>( Ex const& ex ) noexcept // <- the output has been zero-padded by n pixels
            {
                return make_unary_operator
                (
                    [=]<Tensor Tsor>( Tsor const& tsor ) noexcept
                    {
                        if (learning_phase != 1)
                            return tsor;

                        typedef typename Tsor::value_type value_type;
                        std::vector<unsigned long> const& shape = tsor.shape();
                        auto const[batch_size, row, col, channel] = std::make_tuple(shape[0], shape[1], shape[2], shape[3]);
                        view_4d vi{tsor.data(), batch_size, row, col, channel};

                        tensor<long> shifts = context_cast<tensor<long>>( shift_cache );
                        shifts.resize( {channel, 2} );
                        {   //generating random shifts
                            std::uniform_int_distribution<long> distribution( -pixels, pixels );
                            for ( auto& v : shifts )
                                v = distribution(random_generator);
                        }
                        view_2d _shifts{shifts.data(), channel, 2};

                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        ans.resize( tsor.shape() );
                        std::fill( ans.begin(), ans.end(), value_type{0} );
                        view_4d vo{ans.data(), batch_size, row, col, channel};

                        for ( auto bs : range(batch_size ) )
                        {
                            for ( auto ch : range( channel ) )
                            {
                                auto [row_shift, col_shift] = std::make_tuple( _shifts[ch][0], _shifts[ch][1]);
                                for ( auto r : range( row ) )
                                {
                                    if (r-row_shift>=0 && r-row_shift<row)
                                    {
                                        for ( auto c : range( col ) )
                                        {
                                            if (c-col_shift>=0 && c-col_shift<col )
                                                vo[bs][r][c][ch] = vi[bs][r-row_shift][c-col_shift][ch];
                                        }
                                    }
                                }
                            }
                        }
                        return ans;
                    },
                    [=]<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;
                        std::vector<unsigned long> const& shape = grad.shape();
                        auto const[batch_size, row, col, channel] = std::make_tuple( shape[0], shape[1], shape[2], shape[3] );
                        view_4d vi{grad.data(), batch_size, row, col, channel};
                        tensor<long> shifts = context_cast<tensor<long>>( shift_cache );
                        view_2d _shifts{shifts.data(), channel, 2};

                        Tsor& ans = context_cast<Tsor>( backward_cache );
                        ans.resize( grad.shape() );
                        std::fill( ans.begin(), ans.end(), value_type{0} );
                        view_4d vo{ans.data(), batch_size, row, col, channel};

                        for ( auto bs : range(batch_size ) )
                        {
                            for ( auto ch : range( channel ) )
                            {
                                auto [row_shift, col_shift] = std::make_tuple( _shifts[ch][0], _shifts[ch][1] );
                                for ( auto r : range( row ) )
                                {
                                    if (r+row_shift>=0 && r+row_shift<row)
                                    {
                                        for ( auto c : range( col ) )
                                        {
                                            if (c+col_shift>=0 && c+col_shift<col )
                                                vo[bs][r][c][ch] = vi[bs][r+row_shift][c+col_shift][ch];
                                        }
                                    }
                                }
                            }
                        }
                        return ans;
                    },
                    "detailed_sliding_2d"
                )( ex );
            };
        }

    //} // anonymous namespace

    inline auto sliding_2d( unsigned long pixels ) noexcept
    {
        std::shared_ptr<std::any> shift_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [=]<Expression Ex>( Ex const& ex ) noexcept
        {
            return cropping_2d( {pixels,} )( detailed_sliding_2d(pixels, shift_cache, forward_cache, backward_cache)( zero_padding_2d( {pixels,} )( ex ) )  );
        };
    }

    namespace
    {
        struct repeat_context
        {
            auto make_forward() const noexcept
            {
                return []( unsigned long repeats, unsigned long axis, std::shared_ptr<std::any> forward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input ) noexcept
                    {
                        if ( 1UL == repeats ) return input;
                        unsigned long const ax = std::min( axis, input.shape().size()-1 );

                        auto const& shape = input.shape();
                        unsigned long const stride = std::accumulate( shape.begin()+ax+1, shape.end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } );
                        unsigned long const iterations = std::accumulate( shape.begin(), shape.begin()+ax+1, 1UL, []( unsigned long x, unsigned long y ){ return x*y; } );

                        // generate output tensor
                        std::vector<unsigned long> output_shape = input.shape();
                        output_shape[ax] *= repeats;

                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        ans.resize( output_shape );

                        // create 2D and 3D view
                        view_2d v2{ input.data(), iterations, stride };
                        view_3d v3{ ans.data(), iterations, repeats, stride };

                        // copy data
                        for ( auto it : range( iterations ) )
                            for ( auto re : range( repeats ) )
                                std::copy_n( v2[it], stride, v3[it][re] );

                        return ans;
                    };
                };
            }

            auto make_backward() const noexcept
            {
                return []( unsigned long repeats, unsigned long axis, std::shared_ptr<std::any> backward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                    {
                        if ( 1UL == repeats ) return grad;
                        unsigned long const ax = std::min( axis, input.shape().size()-1 );

                        auto const& shape = input.shape();
                        unsigned long const stride = std::accumulate( shape.begin()+ax+1, shape.end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } );
                        unsigned long const iterations = std::accumulate( shape.begin(), shape.begin()+ax+1, 1UL, []( unsigned long x, unsigned long y ){ return x*y; } );

                        Tsor& ans = context_cast<Tsor>( backward_cache );
                        ans.resize( input.shape() );
                        ans.reset();

                        view_2d v2{ans.data(), iterations, stride };
                        view_3d v3{ grad.data(), iterations, repeats, stride };

                        for ( auto id : range( iterations ) )
                            for ( auto re : range( repeats ) )
                                for ( auto st : range( stride ) )
                                    v2[id][st] += v3[id][re][st];

                        return ans;
                    };
                };
            }
        };//struct repeat_context
    }//anonymous namespace


    ///
    /// @brief Repeats elements along an axis.
    /// @param repeats The number of repetitions for each element.
    /// @param axis The axis along which to repeat values. Defaults to the last axis.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} };
    /// auto b0 = repeat( 2, 0 )( a ); // <- output shape is ( 4, 3, 5 )
    /// auto b1 = repeat( 2, 1 )( a ); // <- output shape is ( 2, 6, 5 )
    /// auto b2 = repeat( 2, 2 )( a ); // <- output shape is ( 2, 3, 10 )
    /// auto bx = repeat( 2 )( a ); // <- output shape is ( 2, 3, 10 )
    /// \endcode
    ///
    inline auto repeat( unsigned long repeats, unsigned long axis=-1 ) noexcept
    {
        better_assert( repeats > 0, "repeat: repeats can not be zero." );

        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [repeats, axis, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                repeat_context{}.make_forward()( repeats, axis, forward_cache ),
                repeat_context{}.make_backward()( repeats, axis, backward_cache ),
                "repeat",
                [=]( std::vector<unsigned long> const& shape ) noexcept
                {
                    std::vector<unsigned long> ans = shape;
                    if ( axis >= ans.size() ) axis = ans.size()-1;
                    ans[axis] *= repeats;
                    return ans;
                }
                //TODO: serializer
            )
            ( ex );
        };
    }


    namespace
    {
        struct reduce_min_context
        {
            auto make_forward() const noexcept
            {
                return []( unsigned long axis, std::shared_ptr<std::any> forward_cache, std::shared_ptr<std::any> index_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input ) noexcept
                    {
                        unsigned long const ax = std::min( axis, input.shape().size()-1 );

                        // example: for an input tensor of shape ( 2, 3, 4, 5 ), and axis is 1
                        auto const& shape = input.shape(); // example: the shape is ( 2, 3, 4, 5 )
                        unsigned long const stride = std::accumulate( shape.begin()+ax+1, shape.end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the stride is 20
                        unsigned long const iterations = std::accumulate( shape.begin(), shape.begin()+ax, 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the iterations is 2
                        unsigned long const scales = shape[ax]; // the elements in the dimenstion to reduce. example: scales is 3

                        // generate output tensor
                        std::vector<unsigned long> output_shape = input.shape(); // example: temporately being ( 2, 3, 4, 5 )
                        std::copy( output_shape.begin()+ax+1, output_shape.end(), output_shape.begin()+ax ); // example: temporately being ( 2, 4, 5, 5 )
                        output_shape.resize( output_shape.size() - 1 ); // example: output_shape is ( 2, 4, 5 )

                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        ans.resize( output_shape ); // example: ans shape is ( 2, 4, 5 )

                        tensor<unsigned long>& index = context_cast<tensor<unsigned long>>( index_cache );
                        index.resize( output_shape ); // example: index shape is ( 2, 4, 5 )

                        // create 2D and 3D view
                        view_2d v2{ ans.data(), iterations, stride }; // example: viewing as a matrix of shape ( 2, 20 )
                        view_2d v_index{ index.data(), iterations, stride }; // example: viewing as a matrix of ( 2, 20 )
                        view_3d v3{ input.data(), iterations, scales, stride }; // example: viewing as a tube of ( 2, 3, 20 )

                        // reduce minimal elements along the selected axis
                        for ( auto it : range( iterations ) ) // example: range (2)
                            for ( auto st : range( stride ) ) // example: range (20)
                            {
                                // reduce the minimal elements along the column of st
                                auto min_itor = std::min_element( v3[it].col_begin(st), v3[it].col_end(st) );
                                v2[it][st] = *min_itor;

                                // record the minimal position offset with respect to the head of the column
                                unsigned long const offset = std::distance( v3[it].col_begin(st), min_itor );
                                v_index[it][st] = offset;
                            }

                        return ans;
                    };
                };
            }

            auto make_backward() const noexcept
            {
                return []( unsigned long axis, std::shared_ptr<std::any> backward_cache, std::shared_ptr<std::any> index_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                    {
                        unsigned long const ax = std::min( axis, input.shape().size()-1 );

                        // example: for an input tensor of shape ( 2, 3, 4, 5 ), and axis is 1
                        auto const& shape = input.shape(); // example: the shape is ( 2, 3, 4, 5 )
                        unsigned long const stride = std::accumulate( shape.begin()+ax+1, shape.end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the stride is 20
                        unsigned long const iterations = std::accumulate( shape.begin(), shape.begin()+ax, 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the iterations is 2
                        unsigned long const scales = shape[ax]; // the elements in the dimenstion to reduce. example: scales is 3

                        std::vector<unsigned long> const& output_shape = grad.shape(); // example: output shape of ( 2, 4, 5 )
                        tensor<unsigned long>& index = context_cast<tensor<unsigned long>>( index_cache );
                        index.resize( output_shape ); // example: index shape is ( 2, 4, 5 )

                        Tsor& ans = context_cast<Tsor>( backward_cache );
                        ans.resize( shape ); // example: ans shape is ( 2, 3, 4, 5 )
                        ans.reset();

                        view_2d v_index{ index.data(), iterations, stride }; // example: viewing as a matrix of ( 2, 20 )
                        view_3d v3{ ans.data(), iterations, scales, stride }; // example: view as a cube of ( 2, 3, 20 )
                        view_2d v2{ grad.data(), iterations, stride }; // example: viewing as a matrix of ( 2, 20 )

                        for ( auto it : range( iterations ) ) // example: range( 2 )
                            for ( auto st : range( stride ) ) // example: range( 20 )
                            {
                                unsigned long const offset = v_index[it][st]; // get the offset from record
                                v3[it][offset][st] = v2[it][st]; // only the element at the minimal position has gradient back-propagated
                            }

                        return ans;
                    };
                };
            }
        };//struct reduce_min_context
    }//anonymous namespace


    ///
    /// @brief Reduce minimal elements along an axis.
    /// @param axis The axis along which to reduce minimal values. Defaults to the last axis.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = reduce_min( 0 )( a ); // <- output shape is ( 3, 5 )
    /// auto b = reduce_min( 1 )( a ); // <- output shape is ( 2, 5 )
    /// auto b = reduce_min( 2 )( a ); // <- output shape is ( 2, 3 )
    /// auto b = reduce_min( )( a ); // <- output shape is ( 2, 3 )
    /// \endcode
    ///
    inline auto reduce_min( unsigned long axis=-1 ) noexcept
    {
        std::shared_ptr<std::any> index_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [axis, index_cache, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                reduce_min_context{}.make_forward()( axis, forward_cache, index_cache ),
                reduce_min_context{}.make_backward()( axis, backward_cache, index_cache ),
                "reduce_min",
                [=]( std::vector<unsigned long> const& shape ) noexcept
                {
                    std::vector<unsigned long> ans = shape;
                    if ( axis >= shape.size() ) axis = shape.size() - 1;
                    std::copy( ans.begin()+axis+1, ans.end(), ans.begin()+axis );
                    ans.resize( ans.size() - 1 );
                    return ans;
                }
                //TODO: serializer
            )
            ( ex );
        };
    }



    namespace
    {
        struct reduce_max_context
        {
            auto make_forward() const noexcept
            {
                return []( unsigned long axis, std::shared_ptr<std::any> forward_cache, std::shared_ptr<std::any> index_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input ) noexcept
                    {
                        unsigned long const ax = std::min( axis, input.shape().size()-1 );

                        // example: for an input tensor of shape ( 2, 3, 4, 5 ), and axis is 1
                        auto const& shape = input.shape(); // example: the shape is ( 2, 3, 4, 5 )
                        unsigned long const stride = std::accumulate( shape.begin()+ax+1, shape.end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the stride is 20
                        unsigned long const iterations = std::accumulate( shape.begin(), shape.begin()+ax, 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the iterations is 2
                        unsigned long const scales = shape[ax]; // the elements in the dimenstion to reduce. example: scales is 3

                        // generate output tensor
                        std::vector<unsigned long> output_shape = input.shape(); // example: temporately being ( 2, 3, 4, 5 )
                        std::copy( output_shape.begin()+ax+1, output_shape.end(), output_shape.begin()+ax ); // example: temporately being ( 2, 4, 5, 5 )
                        output_shape.resize( output_shape.size() - 1 ); // example: output_shape is ( 2, 4, 5 )

                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        ans.resize( output_shape ); // example: ans shape is ( 2, 4, 5 )

                        tensor<unsigned long>& index = context_cast<tensor<unsigned long>>( index_cache );
                        index.resize( output_shape ); // example: index shape is ( 2, 4, 5 )

                        // create 2D and 3D view
                        view_2d v2{ ans.data(), iterations, stride }; // example: viewing as a matrix of shape ( 2, 20 )
                        view_2d v_index{ index.data(), iterations, stride }; // example: viewing as a matrix of ( 2, 20 )
                        view_3d v3{ input.data(), iterations, scales, stride }; // example: viewing as a tube of ( 2, 3, 20 )

                        // reduce maximal elements along the selected axis
                        for ( auto it : range( iterations ) ) // example: range (2)
                            for ( auto st : range( stride ) ) // example: range (20)
                            {
                                // reduce the maximal elements along the column of st
                                auto max_itor = std::max_element( v3[it].col_begin(st), v3[it].col_end(st) );
                                v2[it][st] = *max_itor;

                                // record the maximal position offset with respect to the head of the column
                                unsigned long const offset = std::distance( v3[it].col_begin(st), max_itor );
                                v_index[it][st] = offset;
                            }

                        return ans;
                    };
                };
            }

            auto make_backward() const noexcept
            {
                return []( unsigned long axis, std::shared_ptr<std::any> backward_cache, std::shared_ptr<std::any> index_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input, Tsor const& , Tsor const& grad ) noexcept
                    {
                        unsigned long const ax = std::min( axis, input.shape().size()-1 );

                        // example: for an input tensor of shape ( 2, 3, 4, 5 ), and axis is 1
                        auto const& shape = input.shape(); // example: the shape is ( 2, 3, 4, 5 )
                        unsigned long const stride = std::accumulate( shape.begin()+ax+1, shape.end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the stride is 20
                        unsigned long const iterations = std::accumulate( shape.begin(), shape.begin()+ax, 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the iterations is 2
                        unsigned long const scales = shape[ax]; // the elements in the dimenstion to reduce. example: scales is 3

                        std::vector<unsigned long> const& output_shape = grad.shape(); // example: output shape of ( 2, 4, 5 )
                        tensor<unsigned long>& index = context_cast<tensor<unsigned long>>( index_cache );
                        index.resize( output_shape ); // example: index shape is ( 2, 4, 5 )

                        Tsor& ans = context_cast<Tsor>( backward_cache );
                        ans.resize( shape ); // example: ans shape is ( 2, 3, 4, 5 )
                        ans.reset();

                        view_2d v_index{ index.data(), iterations, stride }; // example: viewing as a matrix of ( 2, 20 )
                        view_3d v3{ ans.data(), iterations, scales, stride }; // example: view as a cube of ( 2, 3, 20 )
                        view_2d v2{ grad.data(), iterations, stride }; // example: viewing as a matrix of ( 2, 20 )

                        for ( auto it : range( iterations ) ) // example: range( 2 )
                            for ( auto st : range( stride ) ) // example: range( 20 )
                            {
                                unsigned long const offset = v_index[it][st]; // get the offset from record
                                v3[it][offset][st] = v2[it][st]; // only the element at the maximal position has gradient back-propagated
                            }

                        return ans;
                    };
                };
            }
        };//struct reduce_max_context
    }//anonymous namespace


    ///
    /// @brief Reduce maximum elements along an axis.
    /// @param axis The axis along which to reduce maximum values. Defaults to the last axis.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = reduce_max( 0 )( a ); // <- output shape is ( 3, 5 )
    /// auto b = reduce_max( 1 )( a ); // <- output shape is ( 2, 5 )
    /// auto b = reduce_max( 2 )( a ); // <- output shape is ( 2, 3 )
    /// auto b = reduce_max( )( a ); // <- output shape is ( 2, 3 )
    /// \endcode
    ///
    inline auto reduce_max( unsigned long axis=-1 ) noexcept
    {
        std::shared_ptr<std::any> index_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [axis, index_cache, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                reduce_max_context{}.make_forward()( axis, forward_cache, index_cache ),
                reduce_max_context{}.make_backward()( axis, backward_cache, index_cache ),
                "reduce_max",
                [=]( std::vector<unsigned long> const& shape ) noexcept
                {
                    std::vector<unsigned long> ans = shape;
                    if ( axis >= shape.size() ) axis = shape.size() - 1;
                    std::copy( ans.begin()+axis+1, ans.end(), ans.begin()+axis );
                    ans.resize( ans.size() - 1 );
                    return ans;
                }
                //TODO: serializer
            )
            ( ex );
        };
    }



    namespace
    {
        struct reduce_sum_context
        {
            auto make_forward() const noexcept
            {
                return []( unsigned long axis, std::shared_ptr<std::any> forward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input ) noexcept
                    {
                        typedef typename Tsor::value_type value_type;

                        unsigned long const ax = std::min( axis, input.shape().size()-1 );

                        // example: for an input tensor of shape ( 2, 3, 4, 5 ), and axis is 1
                        auto const& shape = input.shape(); // example: the shape is ( 2, 3, 4, 5 )
                        unsigned long const stride = std::accumulate( shape.begin()+ax+1, shape.end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the stride is 20
                        unsigned long const iterations = std::accumulate( shape.begin(), shape.begin()+ax, 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the iterations is 2
                        unsigned long const scales = shape[ax]; // the elements in the dimenstion to reduce. example: scales is 3

                        // generate output tensor
                        std::vector<unsigned long> output_shape = input.shape(); // example: temporately being ( 2, 3, 4, 5 )
                        std::copy( output_shape.begin()+ax+1, output_shape.end(), output_shape.begin()+ax ); // example: temporately being ( 2, 4, 5, 5 )
                        output_shape.resize( output_shape.size() - 1 ); // example: output_shape is ( 2, 4, 5 )

                        Tsor& ans = context_cast<Tsor>( forward_cache );
                        ans.resize( output_shape ); // example: ans shape is ( 2, 4, 5 )

                        // create 2D and 3D view
                        view_2d v2{ ans.data(), iterations, stride }; // example: viewing as a matrix of shape ( 2, 20 )
                        view_3d v3{ input.data(), iterations, scales, stride }; // example: viewing as a tube of ( 2, 3, 20 )

                        // reduce sum along the selected axis
                        for ( auto it : range( iterations ) ) // example: range (2)
                            for ( auto st : range( stride ) ) // example: range (20)
                                v2[it][st] = std::accumulate( v3[it].col_begin(st), v3[it].col_end(st), value_type{0} );

                        return ans;
                    };
                };
            }

            auto make_backward() const noexcept
            {
                return []( unsigned long axis, std::shared_ptr<std::any> backward_cache ) noexcept
                {
                    return [=]<Tensor Tsor>( Tsor const& input, Tsor const& , Tsor const& grad ) noexcept
                    {
                        unsigned long const ax = std::min( axis, input.shape().size()-1 );

                        // example: for an input tensor of shape ( 2, 3, 4, 5 ), and axis is 1
                        auto const& shape = input.shape(); // example: the shape is ( 2, 3, 4, 5 )
                        unsigned long const stride = std::accumulate( shape.begin()+ax+1, shape.end(), 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the stride is 20
                        unsigned long const iterations = std::accumulate( shape.begin(), shape.begin()+ax, 1UL, []( unsigned long x, unsigned long y ){ return x*y; } ); // example: the iterations is 2
                        unsigned long const scales = shape[ax]; // the elements in the dimenstion to reduce. example: scales is 3

                        Tsor& ans = context_cast<Tsor>( backward_cache );
                        ans.resize( shape ); // example: ans shape is ( 2, 3, 4, 5 )
                        ans.reset();

                        view_3d v3{ ans.data(), iterations, scales, stride }; // example: view as a cube of ( 2, 3, 20 )
                        view_2d v2{ grad.data(), iterations, stride }; // example: viewing as a matrix of ( 2, 20 )

                        for ( auto it : range( iterations ) ) // example: range( 2 )
                            for ( auto st : range( stride ) ) // example: range( 20 )
                                std::fill( v3[it].col_begin( st ), v3[it].col_end( st ), v2[it][st] );

                        return ans;
                    };
                };
            }
        };//struct reduce_sum_context
    }//anonymous namespace


    ///
    /// @brief Reduce sum elements along an axis.
    /// @param axis The axis along which to reduce sum.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = reduce_sum( 0 )( a ); // <- output shape is ( 3, 5 )
    /// auto b = reduce_sum( 1 )( a ); // <- output shape is ( 2, 5 )
    /// auto b = reduce_sum( 2 )( a ); // <- output shape is ( 2, 3 )
    /// auto b = reduce_sum( -1 )( a ); // <- output shape is ( 2, 3 )
    /// \endcode
    ///
    inline auto reduce_sum( unsigned long axis ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();

        return [axis, forward_cache, backward_cache]<Expression Ex>( Ex const& ex ) noexcept
        {
            return make_unary_operator
            (
                reduce_sum_context{}.make_forward()( axis, forward_cache ),
                reduce_sum_context{}.make_backward()( axis, backward_cache ),
                "reduce_sum",
                [=]( std::vector<unsigned long> const& shape ) noexcept
                {
                    std::vector<unsigned long> ans = shape;
                    if ( axis >= shape.size() ) axis = shape.size() - 1;
                    std::copy( ans.begin()+axis+1, ans.end(), ans.begin()+axis );
                    ans.resize( ans.size() - 1 );
                    return ans;
                }
                //TODO: serializer
            )
            ( ex );
        };
    }




    ///
    /// @brief Computes Abs of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = abs( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr abs( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::abs(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g * ((x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0)); } );
                                        return ans;
                                    },
                                    "abs"
                )( ex );
    };






    ///
    /// @brief Computes Acos of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = acos( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr acos( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::acos(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = - g / std::sqrt(1.0-x*x); } );
                                        return ans;
                                    },
                                    "acos"
                )( ex );
    };






    ///
    /// @brief Computes Acosh of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = acosh( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr acosh( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::acosh(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g / std::sqrt(x*x-1.0); } );
                                        return ans;
                                    },
                                    "acosh"
                )( ex );
    };






    ///
    /// @brief Computes Asin of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = asin( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr asin( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::asin(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g / std::sqrt(1.0-x*x); } );
                                        return ans;
                                    },
                                    "asin"
                )( ex );
    };






    ///
    /// @brief Computes Asinh of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = asinh( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr asinh( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::asinh(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g / std::sqrt(1.0+x*x); } );
                                        return ans;
                                    },
                                    "asinh"
                )( ex );
    };






    ///
    /// @brief Computes Atan of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = atan( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr atan( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::atan(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g / (1.0+x*x); } );
                                        return ans;
                                    },
                                    "atan"
                )( ex );
    };






    ///
    /// @brief Computes Atanh of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = atanh( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr atanh( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::atanh(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g / (1-x*x); } );
                                        return ans;
                                    },
                                    "atanh"
                )( ex );
    };






    ///
    /// @brief Computes Cbert of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = cbrt( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr cbrt( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::cbrt(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), output.begin(), grad.begin(), ans.begin(), []( auto, auto o, auto g, auto& v ) noexcept { v = g / (3.0*o*o); } );
                                        return ans;
                                    },
                                    "cbert"
                )( ex );
    };






    ///
    /// @brief Computes Ceil of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = ceil( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr ceil( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::ceil(x); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        return grad;
                                    },
                                    "ceil"
                )( ex );
    };






    ///
    /// @brief Computes Cos of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = cos( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr cos( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::cos(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = - g * std::sin(x); } );
                                        return ans;
                                    },
                                    "cos"
                )( ex );
    };






    ///
    /// @brief Computes Cosh of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = cosh( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr cosh( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::cosh(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g * std::sinh(x); } );
                                        return ans;
                                    },
                                    "cosh"
                )( ex );
    };






    ///
    /// @brief Computes Erf of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = erf( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr erf( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::erf(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = typename Tsor::value_type{1.12837916709551257389} * g * std::exp(-x*x); } );
                                        return ans;
                                    },
                                    "erf"
                )( ex );
    };






    ///
    /// @brief Computes Erfc of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = erfc( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr erfc( Ex const& ex ) noexcept
    {

        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::erfc(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = typename Tsor::value_type{-1.12837916709551257389} * g * std::exp(-x*x); } );
                                        return ans;
                                    },
                                    "erfc"
                )( ex );
    };






    ///
    /// @brief Computes Exp of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = exp( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr exp( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::exp(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), output.begin(), grad.begin(), ans.begin(), []( auto, auto o, auto g, auto& v ) noexcept { v = g * o; } );
                                        return ans;
                                    },
                                    "exp"
                )( ex );
    };






    ///
    /// @brief Computes Exp2 of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = exp2( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr exp2( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::exp2(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), output.begin(), grad.begin(), ans.begin(), []( auto, auto o, auto g, auto& v ) noexcept { v = std::log(2.0) * g * o; } );
                                        return ans;
                                    },
                                    "exp2"
                )( ex );
    };






    ///
    /// @brief Computes Expm1 of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = expm1( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr expm1( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::expm1(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), output.begin(), grad.begin(), ans.begin(), []( auto, auto o, auto g, auto& v ) noexcept { v = g * (o+1.0); } );
                                        return ans;
                                    },
                                    "expm1"
                )( ex );
    };






    ///
    /// @brief Computes Fabs of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = fabs( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr fabs( Ex const& ex ) noexcept
    {
        return abs( ex );
    };






    ///
    /// @brief Computes Floor of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = floor( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr floor( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::floor(x); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        return grad;
                                    },
                                    "floor"
                )( ex );
    };










    ///
    /// @brief Computes Llrint of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = llrint( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr llrint( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::llrint(x); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        return grad;
                                    },
                                    "llrint"
                )( ex );
    };






    ///
    /// @brief Computes Llround of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = llround( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr llround( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::llround(x); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        return grad;
                                    },
                                    "llround"
                )( ex );
    };






    ///
    /// @brief Computes Log of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = log( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr log( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::log(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g / x; } );
                                        return ans;
                                    },
                                    "log"
                )( ex );
    };






    ///
    /// @brief Computes Log10 of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = log10( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr log10( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::log10(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g / (2.30258509299404568402*x); } );
                                        return ans;
                                    },
                                    "log10"
                )( ex );
    };






    ///
    /// @brief Computes Log1p of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = log1p( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr log1p( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::log1p(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g / x; } );
                                        return ans;
                                    },
                                    "log1p"
                )( ex );
    };






    ///
    /// @brief Computes Log2 of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = log2( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr log2( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::log2(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g / (0.69314718055994530942*x); } );
                                        return ans;
                                    },
                                    "log2"
                )( ex );
    };



    ///
    /// @brief Computes Lrint of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = lrint( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr lrint( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::lrint(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        return grad;
                                    },
                                    "lrint"
                )( ex );
    };






    ///
    /// @brief Computes Lround of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = lround( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr lround( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::lround(x); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        return grad;
                                    },
                                    "lround"
                )( ex );
    };






    ///
    /// @brief Computes Nearbyint of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = nearbyint( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr nearbyint( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::nearbyint(x); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        return grad;
                                    },
                                    "nearbyint"
                )( ex );
    };






    ///
    /// @brief Computes Rint of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = rint( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr rint( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::rint(x); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        return grad;
                                    },
                                    "rint"
                )( ex );
    };






    ///
    /// @brief Computes Round of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = round( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr round( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::round(x); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        return grad;
                                    },
                                    "round"
                )( ex );
    };






    ///
    /// @brief Computes Sin of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = sin( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr sin( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::sin(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g * std::cos(x); } );
                                        return ans;
                                    },
                                    "sin"
                )( ex );
    };






    ///
    /// @brief Computes Sinh of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = sinh( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr sinh( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::sinh(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), []( auto x, auto g, auto& v ) noexcept { v = g * std::cosh(x); } );
                                        return ans;
                                    },
                                    "sinh"
                )( ex );
    };






    ///
    /// @brief Computes Sqrt of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = sqrt( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr sqrt( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::sqrt(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), output.begin(), grad.begin(), ans.begin(), []( auto, auto o, auto g, auto& v ) noexcept { v = g / (o+o); } );
                                        return ans;
                                    },
                                    "sqrt"
                )( ex );
    };






    ///
    /// @brief Computes Tan of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = tan( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr tan( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::tan(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), output.begin(), grad.begin(), ans.begin(), []( auto x, auto o, auto g, auto& v ) noexcept { v = g * (1.0+o*o); } );
                                        return ans;
                                    },
                                    "tan"
                )( ex );
    };






    ///
    /// @brief Computes Tanh of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = tanh( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr tanh( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::tanh(x); } );
                                        return ans;
                                    },
                                    [backward_cache]<Tensor Tsor>( Tsor const& input, Tsor const& output, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), output.begin(), grad.begin(), ans.begin(), []( auto, auto o, auto g, auto& v ) noexcept { v = g * (1.0-o*o); } );
                                        return ans;
                                    },
                                    "tanh"
                )( ex );
    };


    ///
    /// @brief Computes Trunc of the given expression.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = trunc( a );
    /// \endcode
    ///
    template <Expression Ex>
    auto constexpr trunc( Ex const& ex ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), []( auto x, auto& v ) noexcept { v = std::trunc(x); } );
                                        return ans;
                                    },
                                    []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        return grad;
                                    },
                                    "trunc"
                )( ex );
    };



    ///
    /// @breif Updating the second expression's value by assining the first one to it. The second expression should be a 'variable'.
    /// @param lhs_ex A mutable value.
    /// @param rhs_ex An expression to be assigned to lhs_ex.
    /// TODO: Fixme, this implementation is wrong
    ///
    /// \code{.cpp}
    /// auto x = constant{ ... } * constant{ ... };
    /// auto v = variable{ ... };
    /// assgin( x, v );
    /// \endcode
    ///
    template< Expression Lhs_Expression, Variable Rhs_Expression >
    auto constexpr assign( Lhs_Expression const& lhs_ex, Rhs_Expression const& rhs_ex ) noexcept
    {
        return make_binary_operator( []<Tensor Tsor>( Tsor const& lhs_tensor, Tsor& rhs_tensor ) noexcept
                                     {
                                        rhs_tensor.reshape( lhs_tensor.shape() );
                                        std::copy( lhs_tensor.begin(), lhs_tensor.end(), rhs_tensor.begin() );
                                        return lhs_tensor;
                                     },
                                     []<Tensor Tsor>( Tsor const& lhs_input, Tsor const& rhs_input, Tsor const&, Tsor const& ) noexcept
                                     {
                                        return std::make_tuple( zeros_like( lhs_input ), zeros_like( rhs_input ) );
                                     },
                                     "assign"
                )( lhs_ex, rhs_ex );
    };



    ///
    /// `poisson` produces random tensor from a normal distribution
    /// @return An unary operator that takes an unary operator, and producing output tensor subjects to a Poisson distribution.
    ///         The shape of the output tensor has the same shape corresponding to the input unary operator.
    ///
    /// Example Code
    /// @code
    /// auto va = variable{ ones<float>({3, 3, 3}) };
    /// auto v_rand = poisson( va ); // this expression will produces a tensor of shape (3, 3, 3) subjects to a Poisson distribution
    /// @endcode
    ///
    template< Expression Ex>
    auto poisson(Ex const& ex ) noexcept
    {
        return make_unary_operator
        (
            [=]<Tensor Tsor>( Tsor const& tsor ) noexcept
            {
                return poisson( tsor );
            },
            []<Tensor Tsor>( Tsor const&, Tsor const&, Tsor const& grad ) noexcept
            {
                return grad;
            },
            "poisson"
        )(ex);
    }

    ///
    /// @brief Computes expression raised to the power exponent.
    ///
    /// Example code:
    /// \code{.cpp}
    /// auto a = variable{ random<float>( {2, 3, 5} ) };
    /// auto b = pow( a, -1.0 );
    /// \endcode
    ///
    template<Expression Ex, typename T>
    auto constexpr pow( Ex const& ex, T exponent ) noexcept
    {
        std::shared_ptr<std::any> forward_cache = std::make_shared<std::any>();
        std::shared_ptr<std::any> backward_cache = std::make_shared<std::any>();
        return make_unary_operator( [forward_cache, exponent]<Tensor Tsor>( Tsor const& input ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( forward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), ans.begin(), [exponent]( auto x, auto& v ) noexcept { v = std::pow(x, exponent); } );
                                        return ans;
                                    },
                                    [backward_cache, exponent]<Tensor Tsor>( Tsor const& input, Tsor const&, Tsor const& grad ) noexcept
                                    {
                                        Tsor& ans = context_cast<Tsor>( backward_cache );
                                        ans.resize( input.shape() );
                                        for_each( input.begin(), input.end(), grad.begin(), ans.begin(), [exponent]( auto x, auto g, auto& v ) noexcept { v = exponent * g * std::pow(x, exponent-1.0); } );
                                        return ans;
                                    },
                                    "pow"
                                    //TODO: Serializer
                )( ex );
    };



}//namespace ceras

#endif//IPKVWSJOCMGGVRASCBLPYHFBCHRIVEXYBOMMDAKFAUDFYVYOOOISLRXJNUJKPJEVMLDPRDSNM

