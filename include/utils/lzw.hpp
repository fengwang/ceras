#ifndef LZW_HPP_INCLUDED_DPIHJASFLDKJAS8U4LAFJLASKJSALKJDFASLFKJDSFALKJASLKFJAAD
#define LZW_HPP_INCLUDED_DPIHJASFLDKJAS8U4LAFJLASKJSALKJDFASLFKJDSFALKJASLKFJAAD


#include "../includes.hpp"
#include "./better_assert.hpp"


namespace lzw
{
    // interfaces:
    // void compress(std::istream& is, std::ostream& os);
    // int decompress(std::istream& is, std::ostream& os);

    namespace details
    {

        // adapted from http://www.cplusplus.com/articles/iL18T05o/#Version6

        /// Type used to store and retrieve codes.
        using CodeType = std::uint32_t;

        namespace globals
        {

            /// Dictionary Maximum Size (when reached, the dictionary will be reset)
            const CodeType dms {512 * 1024};

        } // namespace globals

        ///
        /// @brief Special codes used by the encoder to control the decoder.
        /// @todo Metacodes should not be hardcoded to match their index.
        ///
        enum class MetaCode : CodeType
        {
            Eof = 1u << CHAR_BIT,   ///< End-of-file.
        };

        ///
        /// @brief Encoder's custom dictionary type.
        ///
        struct EncoderDictionary
        {

            ///
            /// @brief Binary search tree node.
            ///
            struct Node
            {

                ///
                /// @brief Default constructor.
                /// @param c    byte that the Node will contain
                ///
                explicit Node(char c): first(globals::dms), c(c), left(globals::dms), right(globals::dms)
                {
                }

                CodeType    first;  ///< Code of first child string.
                char        c;      ///< Byte.
                CodeType    left;   ///< Code of child node with byte < `c`.
                CodeType    right;  ///< Code of child node with byte > `c`.
            };

            ///
            /// @brief Default constructor.
            /// @details It builds the `initials` cheat sheet.
            ///
            EncoderDictionary()
            {
                const long int minc = std::numeric_limits<char>::min();
                const long int maxc = std::numeric_limits<char>::max();
                CodeType k {0};

                for (long int c = minc; c <= maxc; ++c)
                {
                    initials[static_cast<unsigned char> (c)] = k++;
                }

                vn.reserve(globals::dms);
                reset();
            }

            ///
            /// @brief Resets dictionary to its initial contents.
            /// @note Adds dummy nodes to account for the metacodes.
            ///
            void reset()
            {
                vn.clear();
                const long int minc = std::numeric_limits<char>::min();
                const long int maxc = std::numeric_limits<char>::max();

                for (long int c = minc; c <= maxc; ++c)
                {
                    vn.push_back(Node(c));
                }

                // add dummy nodes for the metacodes
                vn.push_back(Node('\x00')); // MetaCode::Eof
            }

            ///
            /// @brief Searches for a pair (`i`, `c`) and inserts the pair if it wasn't found.
            /// @param i                code to search for
            /// @param c                attached byte to search for
            /// @return The index of the pair, if it was found.
            /// @retval globals::dms    if the pair wasn't found
            ///
            CodeType search_and_insert(CodeType i, char c)
            {
                if (i == globals::dms)
                {
                    return search_initials(c);
                }

                const CodeType vn_size = vn.size();
                CodeType ci {vn[i].first}; // Current Index

                if (ci != globals::dms)
                {
                    while (true)
                        if (c < vn[ci].c)
                        {
                            if (vn[ci].left == globals::dms)
                            {
                                vn[ci].left = vn_size;
                                break;
                            }
                            else
                            {
                                ci = vn[ci].left;
                            }
                        }
                        else if (c > vn[ci].c)
                        {
                            if (vn[ci].right == globals::dms)
                            {
                                vn[ci].right = vn_size;
                                break;
                            }
                            else
                            {
                                ci = vn[ci].right;
                            }
                        }
                        else // c == vn[ci].c
                        {
                            return ci;
                        }
                }
                else
                {
                    vn[i].first = vn_size;
                }

                vn.push_back(Node(c));
                return globals::dms;
            }

            ///
            /// @brief Fakes a search for byte `c` in the one-byte area of the dictionary.
            /// @param c    byte to search for
            /// @return The code associated to the searched byte.
            ///
            CodeType search_initials(char c) const
            {
                return initials[static_cast<unsigned char> (c)];
            }

            ///
            /// @brief Returns the number of dictionary entries.
            ///
            std::vector<Node>::size_type size() const
            {
                return vn.size();
            }


            /// Vector of nodes on top of which the binary search tree is implemented.
            std::vector<Node> vn;

            /// Cheat sheet for mapping one-byte strings to their codes.
            std::array < CodeType, 1u << CHAR_BIT > initials;
        };

        ///
        /// @brief Helper structure for use in `CodeWriter` and `CodeReader`.
        ///
        struct ByteCache
        {

            ///
            /// @brief Default constructor.
            ///
            ByteCache(): used(0), data(0x00)
            {
            }

            std::size_t     used;   ///< Bits currently in use.
            unsigned char   data;   ///< The bits of the cached byte.
        };

        ///
        /// @brief Variable binary width code writer.
        ///
        struct CodeWriter
        {
            ///
            /// @brief Default constructor.
            /// @param [out] os     Output Stream to write codes to
            ///
            explicit CodeWriter(std::ostream& os): os(os), bits(CHAR_BIT + 1)
            {
            }

            ///
            /// @brief Destructor.
            /// @note Writes `MetaCode::Eof` and flushes the last byte to the stream.
            ///
            ~CodeWriter()
            {
                write(static_cast<CodeType>(MetaCode::Eof));

                // write the incomplete leftover byte as-is
                if (lo.used != 0)
                {
                    os.put(static_cast<char>(lo.data));
                }
            }

            ///
            /// @brief Getter for `CodeWriter::bits`.
            ///
            std::size_t get_bits() const
            {
                return bits;
            }

            ///
            /// @brief Resets internal binary width.
            /// @note Default value is `CHAR_BIT + 1`.
            ///
            void reset_bits()
            {
                bits = CHAR_BIT + 1;
            }

            ///
            /// @brief Increases internal binary width by one.
            /// @throws std::overflow_error     internal binary width cannot be increased
            /// @remarks The exception should never be thrown, under normal circumstances.
            ///
            void increase_bits()
            {
                ++bits;
            }

            ///
            /// @brief Writes the code `k` with a binary width of `CodeWriter::bits`.
            /// @param k            code to be written
            /// @return Whether or not the stream can be used for output.
            /// @retval true        the output stream can still be used
            /// @retval false       the output stream can no longer be used
            ///
            bool write(CodeType k)
            {
                std::size_t remaining_bits {bits};

                if (lo.used != 0)
                {
                    lo.data |= k << lo.used;
                    os.put(static_cast<char>(lo.data));
                    k >>= CHAR_BIT - lo.used;
                    remaining_bits -= CHAR_BIT - lo.used;
                    lo.used = 0;
                    lo.data = 0x00;
                }

                while (remaining_bits != 0)
                    if (remaining_bits >= CHAR_BIT)
                    {
                        os.put(static_cast<char>(k));
                        k >>= CHAR_BIT;
                        remaining_bits -= CHAR_BIT;
                    }
                    else
                    {
                        lo.used = remaining_bits;
                        lo.data = k;
                        break;
                    }

                return os.good();
            }


            std::ostream&    os;    ///< Output Stream.
            std::size_t     bits;   ///< Binary width of codes.
            ByteCache       lo;     ///< LeftOvers.
        };

        ///
        /// @brief Variable binary width code reader.
        ///
        struct CodeReader
        {
            ///
            /// @brief Default constructor.
            /// @param [in] is      Input Stream to read codes from
            ///
            explicit CodeReader(std::istream& is): is(is), bits(CHAR_BIT + 1), feofmc(false)
            {
            }

            ///
            /// @brief Getter for `CodeReader::bits`.
            ///
            std::size_t get_bits() const
            {
                return bits;
            }

            ///
            /// @brief Resets internal binary width.
            /// @note Default value is `CHAR_BIT + 1`.
            ///
            void reset_bits()
            {
                bits = CHAR_BIT + 1;
            }

            ///
            /// @brief Increases internal binary width by one.
            /// @throws std::overflow_error     if internal binary width cannot be increased
            /// @remarks The exception should never be thrown, under normal circumstances.
            ///
            void increase_bits()
            {
                ++bits;
            }

            ///
            /// @brief Reads the code `k` with a binary width of `CodeReader::bits`.
            /// @param [out] k      code to be read
            /// @return Whether or not the stream can be used for input.
            /// @retval true        the input stream can still be used
            /// @retval false       the input stream can no longer be used
            ///
            bool read(CodeType& k)
            {
                // ready-made bit masks
                static const std::array<unsigned long int, 9> masks { {0x00, 0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF} };
                std::size_t remaining_bits {bits};
                std::size_t offset {lo.used};
                unsigned char temp;
                k = lo.data;
                remaining_bits -= lo.used;
                lo.used = 0;
                lo.data = 0x00;

                while (remaining_bits != 0 && is.get(reinterpret_cast<char&>(temp)))
                    if (remaining_bits >= CHAR_BIT)
                    {
                        k |= static_cast<CodeType>(temp) << offset;
                        offset += CHAR_BIT;
                        remaining_bits -= CHAR_BIT;
                    }
                    else
                    {
                        k |= static_cast<CodeType>(temp & masks[remaining_bits]) << offset;
                        lo.used = CHAR_BIT - remaining_bits;
                        lo.data = temp >> remaining_bits;
                        break;
                    }

                if (k == static_cast<CodeType>(MetaCode::Eof))
                {
                    feofmc = true;
                    return false;
                }

                return is.good();
            }

            ///
            /// @brief Returns if EF is considered corrupted.
            /// @retval true    didn't find end-of-file metacode
            /// @retval false   found end-of-file metacode
            ///
            bool corrupted() const
            {
                return !feofmc;
            }

            std::istream&    is;    ///< Input Stream.
            std::size_t     bits;   ///< Binary width of codes.
            bool            feofmc; ///< Found End-Of-File MetaCode.
            ByteCache       lo;     ///< LeftOvers.
        };

        ///
        /// @brief Computes the minimum number of bits required to store the value of `n`.
        /// @param n    number to be evaluated
        /// @return Number of required bits.
        ///
        inline std::size_t required_bits(unsigned long int n)
        {
            std::size_t r {1};

            while ((n >>= 1) != 0)
            {
                ++r;
            }

            return r;
        }

    }//namespace details

    ///
    /// @brief Compresses the contents of `is` and writes the result to `os`.
    /// @param [in] is      input stream
    /// @param [out] os     output stream
    ///
    inline void compress(std::istream& is, std::ostream& os)
    {
        using namespace details;
        EncoderDictionary ed;
        CodeWriter cw(os);
        CodeType i {globals::dms}; // Index
        char c;
        bool rbwf {false}; // Reset Bit Width Flag

        while (is.get(c))
        {
            // dictionary's maximum size was reached
            if (ed.size() == globals::dms)
            {
                ed.reset();
                rbwf = true;
            }

            const CodeType temp {i};

            if ((i = ed.search_and_insert(temp, c)) == globals::dms)
            {
                cw.write(temp);
                i = ed.search_initials(c);

                if (required_bits(ed.size() - 1) > cw.get_bits())
                {
                    cw.increase_bits();
                }
            }

            if (rbwf)
            {
                cw.reset_bits();
                rbwf = false;
            }
        }

        if (i != globals::dms)
        {
            cw.write(i);
        }
    }

    ///
    /// @brief Decompresses the contents of `is` and writes the result to `os`.
    /// @param [in] is      input stream
    /// @param [out] os     output stream
    /// @return 0 for success, -1 for failure
    ///
    inline int decompress(std::istream& is, std::ostream& os)
    {
        using namespace details;
        std::vector<std::pair<CodeType, char>> dictionary;
        // "named" lambda function, used to reset the dictionary to its initial contents
        const auto reset_dictionary = [&dictionary]
        {
            dictionary.clear();
            dictionary.reserve(globals::dms);

            const long int minc = std::numeric_limits<char>::min();
            const long int maxc = std::numeric_limits<char>::max();

            for (long int c = minc; c <= maxc; ++c)
                dictionary.push_back({globals::dms, static_cast<char>(c)});

            // add dummy elements for the metacodes
            dictionary.push_back({0, '\x00'}); // MetaCode::Eof
        };
        const auto rebuild_string = [&dictionary](CodeType k) -> const std::vector<char>*
        {
            static std::vector<char> s; // String

            s.clear();

            // the length of a string cannot exceed the dictionary's number of entries
            s.reserve(globals::dms);

            while (k != globals::dms)
            {
                s.push_back(dictionary[k].second);
                k = dictionary[k].first;
            }

            std::reverse(s.begin(), s.end());
            return &s;
        };
        reset_dictionary();
        CodeReader cr(is);
        CodeType i {globals::dms}; // Index
        CodeType k; // Key

        while (true)
        {
            // dictionary's maximum size was reached
            if (dictionary.size() == globals::dms)
            {
                reset_dictionary();
                cr.reset_bits();
            }

            if (required_bits(dictionary.size()) > cr.get_bits())
            {
                cr.increase_bits();
            }

            if (!cr.read(k))
            {
                break;
            }

            if (k > dictionary.size())
            {
                better_assert(false, "lzw::invalid compression code with k = ", k, " but dictionary size ", dictionary.size());
                return -1;
            }

            const std::vector<char>* s; // String

            if (k == dictionary.size())
            {
                dictionary.push_back({i, rebuild_string(i)->front()});
                s = rebuild_string(k);
            }
            else
            {
                s = rebuild_string(k);

                if (i != globals::dms)
                    dictionary.push_back({i, s->front()});
            }

            os.write(&s->front(), s->size());
            i = k;
        }

        if (cr.corrupted())
        {
            better_assert(false, "lzw::corrupted comressed file.");
            return -1;
        }

        return 0;
    }

}//namespace lzw

#endif

