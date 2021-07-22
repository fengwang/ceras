#ifndef _SINGLETON_HPP_INCLUDED_ODISFJ948ILDFJOIUIRFGDUISOIURKLJFLKJASLDKJOIUSDLKJSALKFJEOIUJSODIFUEROIUSFDLKJROIUSFDLKJF
#define _SINGLETON_HPP_INCLUDED_ODISFJ948ILDFJOIUIRFGDUISOIURKLJFLKJASLDKJOIUSDLKJSALKFJEOIUJSODIFUEROIUSFDLKJROIUSFDLKJF

#if 0
template<typename T>
class Singleton
{
public:
    static T& getInstance()
    {
        static T value;
        return value;
    }

private:
    Singleton();
    ~Singleton();
};
#endif

namespace ceras
{
    template< typename T >
    struct singleton
    {
        typedef T value_type;
        typedef singleton self_type;

        static value_type& instance()
        {
            static value_type instance_;
            constuctor_.null_action();
            return instance_;
        }

    private:

        singleton( const self_type& );
        self_type& operator = ( const self_type& );
        singleton();

        struct constuctor
        {
            constuctor()
            {
                self_type::instance();
            }
            inline void null_action() const { }
        };

        static constuctor constuctor_;
    };

    template<typename T>
    typename singleton<T>::constuctor singleton<T>::constuctor_;

}//namespace ceras

#endif//_SINGLETON_HPP_INCLUDED_ODISFJ948ILDFJOIUIRFGDUISOIURKLJFLKJASLDKJOIUSDLKJSALKFJEOIUJSODIFUEROIUSFDLKJROIUSFDLKJF

