#ifndef YYPQRPGVSMBWLOYYOUQRRMKPHOKBRYXBAIPWLIFODEYEVSNXXHMRSTNXXGBUDXQEIWWHSFKQV
#define YYPQRPGVSMBWLOYYOUQRRMKPHOKBRYXBAIPWLIFODEYEVSNXXHMRSTNXXGBUDXQEIWWHSFKQV

#include "../includes.hpp"

namespace ceras
{
    ///
    /// @brief Convert types to strings. Useful for computation graph serialization.
    ///
    template< typename T >
    constexpr std::string type2string() noexcept
    {   // TODO: regenerate this file using script
        if constexpr ( std::is_same_v<T, signed char> )
            return std::string{ "signed char" };
        else if constexpr ( std::is_same_v<T, char> )
            return std::string{ "char" };
        else if constexpr ( std::is_same_v<T, unsigned char> )
            return std::string{ "unsigned char" };
        else if constexpr ( std::is_same_v<T, wchar_t> )
            return std::string{ "wchar_t" };
        else if constexpr ( std::is_same_v<T, char16_t> )
            return std::string{ "char16_t" };
        else if constexpr ( std::is_same_v<T, char32_t> )
            return std::string{ "char32_t" };
        else if constexpr ( std::is_same_v<T, short int> )
            return std::string{ "short int" };
        else if constexpr ( std::is_same_v<T, unsigned short int> )
            return std::string{ "unsigned short int" };
        else if constexpr ( std::is_same_v<T, int> )
            return std::string{ "int" };
        else if constexpr ( std::is_same_v<T, unsigned int> )
            return std::string{ "unsigned int" };
        else if constexpr ( std::is_same_v<T, long int> )
            return std::string{ "long int" };
        else if constexpr ( std::is_same_v<T, unsigned long int> )
            return std::string{ "unsigned long int" };
        else if constexpr ( std::is_same_v<T, long long int> )
            return std::string{ "long long int" };
        else if constexpr ( std::is_same_v<T, unsigned long long int> )
            return std::string{ "unsigned long long int" };
        else if constexpr ( std::is_same_v<T, float> )
            return std::string{ "float" };
        else if constexpr ( std::is_same_v<T, double> )
            return std::string{ "double" };
        else if constexpr ( std::is_same_v<T, long double> )
            return std::string{ "long double" };
        else
            return std::string{ "Unkonwn type" };
    }

}//namespace ceras

#endif//YYPQRPGVSMBWLOYYOUQRRMKPHOKBRYXBAIPWLIFODEYEVSNXXHMRSTNXXGBUDXQEIWWHSFKQV

