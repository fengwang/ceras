//
// credit goes to Miguel Raggi https://raw.githubusercontent.com/mraggi/dm-cpp/master/dm.hpp
//
#ifndef TQDM_HPP_INCLUDED_SD239IOUAFJKLASF89U34IKJLAFSLKJ349JIAFLSKJSALKJAS98IUJ
#define TQDM_HPP_INCLUDED_SD239IOUAFJKLASF89U34IKJLAFSLKJ349JIAFLSKJSALKJAS98IUJ

#include "../config.hpp"
#include "../includes.hpp"
#include "./color.hpp"

// -------------------- chrono stuff --------------------

//
extern "C"
{
    int ioctl (int __fd, unsigned long int __request, ...);

    struct winsize
    {
        unsigned short  ws_row;         /* rows, in characters */
        unsigned short  ws_col;         /* columns, in characters */
        unsigned short  ws_xpixel;      /* horizontal size, pixels - not used */
        unsigned short  ws_ypixel;      /* vertical size, pixels - not used */
    };
}

namespace dm_details
{
    //
    // Warning: only works with Linux, not portable. Links to libc by default.
    //
    unsigned short get_terminal_width()
    {
        if constexpr( ceras::is_windows_platform )
        {
            return 80;
        }
        else
        {
            winsize w;
            ioctl(0, 0x5413, &w); // 0x5413 is translated from macro 'TIOCGWINSZ'
            return w.ws_col;
        }
    }
}

namespace tq
{
    using index = std::ptrdiff_t; // maybe std::size_t, but I hate unsigned types.
    using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

    inline double elapsed_seconds( time_point_t from, time_point_t to )
    {
        using seconds = std::chrono::duration<double>;
        return std::chrono::duration_cast<seconds>( to - from ).count();
    }

    struct Chronometer
    {

        Chronometer() : start_( std::chrono::steady_clock::now() ) {}

        double reset()
        {
            auto previous = start_;
            start_ = std::chrono::steady_clock::now();
            return elapsed_seconds( previous, start_ );
        }

        [[nodiscard]] double peek() const
        {
            auto now = std::chrono::steady_clock::now();
            return elapsed_seconds( start_, now );
        }

        [[nodiscard]] time_point_t get_start() const
        {
            return start_;
        }


        time_point_t start_;
    };

    // -------------------- progress_bar --------------------
    inline void clamp( double& x, double a, double b )
    {
        if ( x < a ) x = a;

        if ( x > b ) x = b;
    }

    struct progress_bar
    {

        void restart()
        {
            chronometer_.reset();
            refresh_.reset();
        }

        void update( double progress )
        {
            clamp( progress, 0, 1 );

            if ( time_since_refresh() > min_time_per_update_ || progress == 0 ||
                    progress == 1 )
            {
                reset_refresh_timer();
                display( progress );
            }

            suffix_.str( "" );
        }

        void set_ostream( std::ostream& os )
        {
            os_ = &os;
        }
        void set_prefix( std::string s )
        {
            prefix_ = std::move( s );
        }
        void set_bar_size( int size )
        {
            bar_size_ = size;
        }
        void set_min_update_time( double time )
        {
            min_time_per_update_ = time;
        }

        template <typename T>
        progress_bar& operator<<( const T& t )
        {
            suffix_ << t;
            return *this;
        }

        double elapsed_time() const
        {
            return chronometer_.peek();
        }


        void display( double progress )
        {
            auto flags = os_->flags();
            double t = chronometer_.peek();
            double eta = t / progress - t;
            std::stringstream bar;
            bar << '\r' << prefix_ << '{' << std::fixed << std::setprecision( 1 )
                << std::setw( 5 ) << 100 * progress << "%} ";
                //<< std::setw( 5 ) << color::rize( 100 * progress, "Red" ) << "%} ";
            print_bar( bar, progress );
            bar << " (" << color::rize(t, "Magenta") << "s < " << color::rize(eta, "Blue") << "s) ";
            std::string sbar = bar.str();
            std::string suffix = suffix_.str();
            index out_size = sbar.size() + suffix.size();
            term_cols_ = std::max( term_cols_, out_size );
            index num_blank = term_cols_ - out_size;
            ( *os_ ) << sbar << suffix << std::string( num_blank, ' ' ) << std::flush;
            os_->flags( flags );
        }

        void print_bar( std::stringstream& ss, double filled ) const
        {
            auto num_filled = static_cast<index>( std::floor( filled * bar_size_ ) );
            ss << "[";
            for ( int idx = 0; idx != num_filled; ++idx )
                ss << color::rize( "█", "Green" );
            ss << color::rize( std::string( bar_size_ - num_filled, ' ' ), "Default", "Light Green" ) << ']';
        }

        double time_since_refresh() const
        {
            return refresh_.peek();
        }
        void reset_refresh_timer()
        {
            refresh_.reset();
        }

        Chronometer chronometer_{};
        Chronometer refresh_{};
        double min_time_per_update_{0.15}; // found experimentally

        std::ostream* os_{&std::cerr};

        //index bar_size_{100};
        index bar_size_{ dm_details::get_terminal_width()-40 };
        //index bar_size_{ dm_details::get_terminal_width() };
        index term_cols_{1};

        std::string prefix_{};
        std::stringstream suffix_{};
    };

    // -------------------- iter_wrapper --------------------

    template <typename ForwardIter, typename Parent>
    struct iter_wrapper
    {

        using iterator_category = typename ForwardIter::iterator_category;
        using value_type = typename ForwardIter::value_type;
        using difference_type = typename ForwardIter::difference_type;
        using pointer = typename ForwardIter::pointer;
        using reference = typename ForwardIter::reference;

        iter_wrapper( ForwardIter it, Parent* parent ) : current_( it ), parent_( parent )
        {}

        auto operator*()
        {
            return *current_;
        }

        void operator++()
        {
            ++current_;
        }

        template <typename Other>
        bool operator!=( const Other& other ) const
        {
            parent_->update(); // here and not in ++ because I need to run update
            // before first advancement!
            return current_ != other;
        }

        bool operator!=( const iter_wrapper& other ) const
        {
            parent_->update(); // here and not in ++ because I need to run update
            // before first advancement!
            return current_ != other.current_;
        }

        [[nodiscard]] const ForwardIter& get() const
        {
            return current_;
        }


        friend Parent;
        ForwardIter current_;
        Parent* parent_;
    };

    // -------------------- dm_for_lvalues --------------------

    template <typename ForwardIter, typename EndIter = ForwardIter>
    struct dm_for_lvalues
    {

        using this_t = dm_for_lvalues<ForwardIter, EndIter>;
        using iterator = iter_wrapper<ForwardIter, this_t>;
        using value_type = typename ForwardIter::value_type;
        using size_type = index;
        using difference_type = index;

        dm_for_lvalues( ForwardIter begin, EndIter end )
            : first_( begin, this ), last_( end ), num_iters_( std::distance( begin, end ) )
        {}

        dm_for_lvalues( ForwardIter begin, EndIter end, index total )
            : first_( begin, this ), last_( end ), num_iters_( total )
        {}

        template <typename Container>
        explicit dm_for_lvalues( Container& C )
            : first_( C.begin(), this ), last_( C.end() ), num_iters_( C.size() )
        {}

        template <typename Container>
        explicit dm_for_lvalues( const Container& C )
            : first_( C.begin(), this ), last_( C.end() ), num_iters_( C.size() )
        {}

        dm_for_lvalues( const dm_for_lvalues& ) = delete;
        dm_for_lvalues( dm_for_lvalues&& ) = delete;
        dm_for_lvalues& operator=( dm_for_lvalues&& ) = delete;
        dm_for_lvalues& operator=( const dm_for_lvalues& ) = delete;
        ~dm_for_lvalues() = default;

        template <typename Container> dm_for_lvalues( Container&& ) = delete; // prevent misuse!

        iterator begin()
        {
            bar_.restart();
            iters_done_ = 0;
            return first_;
        }

        EndIter end() const
        {
            return last_;
        }

        void update()
        {
            ++iters_done_;
            bar_.update( calc_progress() );
        }

        void set_ostream( std::ostream& os )
        {
            bar_.set_ostream( os );
        }
        void set_prefix( std::string s )
        {
            bar_.set_prefix( std::move( s ) );
        }
        void set_bar_size( int size )
        {
            bar_.set_bar_size( size );
        }
        void set_min_update_time( double time )
        {
            bar_.set_min_update_time( time );
        }

        template <typename T>
        dm_for_lvalues& operator<<( const T& t )
        {
            bar_ << t;
            return *this;
        }

        void manually_set_progress( double to )
        {
            clamp( to, 0, 1 );
            iters_done_ = std::round( to * num_iters_ );
        }


        double calc_progress() const
        {
            double denominator = num_iters_;

            if ( num_iters_ == 0 ) denominator += 1e-9;

            return iters_done_ / denominator;
        }

        iterator first_;
        EndIter last_;
        index num_iters_{0};
        index iters_done_{0};
        progress_bar bar_;
    };

    template <typename Container> dm_for_lvalues( Container& ) -> dm_for_lvalues<typename Container::iterator>;

    template <typename Container> dm_for_lvalues( const Container& )
    -> dm_for_lvalues<typename Container::const_iterator>;

    // -------------------- dm_for_rvalues --------------------

    template <typename Container>
    struct dm_for_rvalues
    {

        using iterator = typename Container::iterator;
        using const_iterator = typename Container::const_iterator;
        using value_type = typename Container::value_type;

        explicit dm_for_rvalues( Container&& C )
            : C_( std::forward<Container>( C ) ), dm_( C_ )
        {}

        auto begin()
        {
            return dm_.begin();
        }

        auto end()
        {
            return dm_.end();
        }

        void update()
        {
            return dm_.update();
        }

        void set_ostream( std::ostream& os )
        {
            dm_.set_ostream( os );
        }
        void set_prefix( std::string s )
        {
            dm_.set_prefix( std::move( s ) );
        }
        void set_bar_size( int size )
        {
            dm_.set_bar_size( size );
        }
        void set_min_update_time( double time )
        {
            dm_.set_min_update_time( time );
        }

        template <typename T>
        auto& operator<<( const T& t )
        {
            return dm_ << t;
        }

        void advance( index amount )
        {
            dm_.advance( amount );
        }

        void manually_set_progress( double to )
        {
            dm_.manually_set_progress( to );
        }


        Container C_;
        dm_for_lvalues<iterator> dm_;
    };

    template <typename Container> dm_for_rvalues( Container&& ) -> dm_for_rvalues<Container>;

    // -------------------- dm --------------------
    template <typename ForwardIter>
    auto dm( const ForwardIter& first, const ForwardIter& last )
    {
        return dm_for_lvalues( first, last );
    }

    template <typename ForwardIter>
    auto dm( const ForwardIter& first, const ForwardIter& last, index total )
    {
        return dm_for_lvalues( first, last, total );
    }

    template <typename Container>
    auto dm( const Container& C )
    {
        return dm_for_lvalues( C );
    }

    template <typename Container>
    auto dm( Container& C )
    {
        return dm_for_lvalues( C );
    }

    template <typename Container>
    auto dm( Container&& C )
    {
        return dm_for_rvalues( std::forward<Container>( C ) );
    }

    // -------------------- int_iterator --------------------

    template <typename IntType>
    struct int_iterator
    {

        using iterator_category = std::random_access_iterator_tag;
        using value_type = IntType;
        using difference_type = IntType;
        using pointer = IntType*;
        using reference = IntType&;

        explicit int_iterator( IntType val ) : value_( val ) {}

        IntType& operator*()
        {
            return value_;
        }

        int_iterator& operator++()
        {
            ++value_;
            return *this;
        }
        int_iterator& operator--()
        {
            --value_;
            return *this;
        }

        int_iterator& operator+=( difference_type d )
        {
            value_ += d;
            return *this;
        }

        difference_type operator-( const int_iterator& other ) const
        {
            return value_ - other.value_;
        }

        bool operator!=( const int_iterator& other ) const
        {
            return value_ != other.value_;
        }


        IntType value_;
    };

    // -------------------- range --------------------
    template <typename IntType>
    struct range
    {

        using iterator = int_iterator<IntType>;
        using const_iterator = iterator;
        using value_type = IntType;

        range( IntType first, IntType last ) : first_( first ), last_( last ) {}
        explicit range( IntType last ) : first_( 0 ), last_( last ) {}

        [[nodiscard]] iterator begin() const
        {
            return first_;
        }
        [[nodiscard]] iterator end() const
        {
            return last_;
        }
        [[nodiscard]] index size() const
        {
            return last_ - first_;
        }


        iterator first_;
        iterator last_;
    };

    template <typename IntType>
    auto trange( IntType first, IntType last )
    {
        return dm( range( first, last ) );
    }

    template <typename IntType>
    auto trange( IntType last )
    {
        return dm( range( last ) );
    }

    // -------------------- timing_iterator --------------------

    struct timing_iterator_end_sentinel
    {

        explicit timing_iterator_end_sentinel( double num_seconds )
            : num_seconds_( num_seconds )
        {}

        [[nodiscard]] double num_seconds() const
        {
            return num_seconds_;
        }


        double num_seconds_;
    };

    struct timing_iterator
    {

        using iterator_category = std::forward_iterator_tag;
        using value_type = double;
        using difference_type = double;
        using pointer = double*;
        using reference = double&;

        double operator*() const
        {
            return chrono_.peek();
        }

        timing_iterator& operator++()
        {
            return *this;
        }

        bool operator!=( const timing_iterator_end_sentinel& other ) const
        {
            return chrono_.peek() < other.num_seconds();
        }


        tq::Chronometer chrono_;
    };

    // -------------------- timer -------------------
    struct timer
    {

        using iterator = timing_iterator;
        using end_iterator = timing_iterator_end_sentinel;
        using const_iterator = iterator;
        using value_type = double;

        explicit timer( double num_seconds ) : num_seconds_( num_seconds ) {}

        [[nodiscard]] static iterator begin()
        {
            return iterator();
        }
        [[nodiscard]] end_iterator end() const
        {
            return end_iterator( num_seconds_ );
        }

        [[nodiscard]] double num_seconds() const
        {
            return num_seconds_;
        }


        double num_seconds_;
    };

    struct dm_timer
    {

        using iterator = iter_wrapper<timing_iterator, dm_timer>;
        using end_iterator = timer::end_iterator;
        using value_type = typename timing_iterator::value_type;
        using size_type = index;
        using difference_type = index;

        explicit dm_timer( double num_seconds ) : num_seconds_( num_seconds ) {}

        dm_timer( const dm_timer& ) = delete;
        dm_timer( dm_timer&& ) = delete;
        dm_timer& operator=( dm_timer&& ) = delete;
        dm_timer& operator=( const dm_timer& ) = delete;
        ~dm_timer() = default;

        template <typename Container> dm_timer( Container&& ) = delete; // prevent misuse!

        iterator begin()
        {
            bar_.restart();
            return iterator( timing_iterator(), this );
        }

        end_iterator end() const
        {
            return end_iterator( num_seconds_ );
        }

        void update()
        {
            double t = bar_.elapsed_time();
            bar_.update( t / num_seconds_ );
        }

        void set_ostream( std::ostream& os )
        {
            bar_.set_ostream( os );
        }
        void set_prefix( std::string s )
        {
            bar_.set_prefix( std::move( s ) );
        }
        void set_bar_size( int size )
        {
            bar_.set_bar_size( size );
        }
        void set_min_update_time( double time )
        {
            bar_.set_min_update_time( time );
        }

        template <typename T>
        dm_timer& operator<<( const T& t )
        {
            bar_ << t;
            return *this;
        }


        double num_seconds_;
        progress_bar bar_;
    };

    inline auto dm( timer t )
    {
        return dm_timer( t.num_seconds() );
    }

} // namespace tq

#endif

