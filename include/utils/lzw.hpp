#ifndef LZW_HPP_INCLUDED_ADSFLKSALKJSALKJASLKJSALSAKJSALKJLKJASLKFSDJLKASJFDFFFFFFFFF
#define LZW_HPP_INCLUDED_ADSFLKSALKJSALKJASLKJSALSAKJSALKJLKJASLKFSDJLKASJFDFFFFFFFFF

// credit goes to https://github.com/glampert/compression-algorithms/blob/master/lzw.hpp

namespace lzw
{

    namespace lzw_details
    {
        inline
        void fatalError( const char* const message )
        {
            std::fprintf( stderr, "LZW encoder/decoder error: %s\n", message );
            std::abort();
        }

        struct BitStreamWriter final
        {
                BitStreamWriter( const BitStreamWriter& ) = delete;
                BitStreamWriter& operator = ( const BitStreamWriter& ) = delete;

                BitStreamWriter();
                explicit BitStreamWriter( std::int64_t initialSizeInBits, std::int64_t growthGranularity = 2 );

                void allocate( std::int64_t bitsWanted );
                void setGranularity( std::int64_t growthGranularity );
                std::uint8_t* release();

                void appendBit( std::int64_t bit );
                void appendBitsU64( std::uint64_t num, std::int64_t bitCount );

                std::string toBitString() const; // Useful for debugging.
                void appendBitString( const std::string& bitStr );

                std::int64_t getByteCount() const;
                std::int64_t getBitCount()  const;
                const std::uint8_t* getBitStream() const;

                ~BitStreamWriter();

            private:

                void internalInit();
                static std::uint8_t* allocBytes( std::int64_t bytesWanted, std::uint8_t* oldPtr, std::int64_t oldSize );

                std::uint8_t* stream;  // Growable buffer to store our bits. Heap allocated & owned by the class instance.
                std::int64_t bytesAllocated;    // Current size of heap-allocated stream buffer *in bytes*.
                std::int64_t granularity;       // Amount bytesAllocated multiplies by when auto-resizing in appendBit().
                std::int64_t currBytePos;       // Current byte being written to, from 0 to bytesAllocated-1.
                std::int64_t nextBitPos;        // Bit position within the current byte to access next. 0 to 7.
                std::int64_t numBitsWritten;    // Number of bits in use from the stream buffer, not including byte-rounding padding.
        };


        struct BitStreamReader final
        {
                BitStreamReader( const BitStreamReader& ) = delete;
                BitStreamReader& operator = ( const BitStreamReader& ) = delete;

                BitStreamReader( const BitStreamWriter& bitStreamWriter );
                BitStreamReader( const std::uint8_t* bitStream, std::int64_t byteCount, std::int64_t bitCount );

                bool isEndOfStream() const;
                bool readNextBit( std::int64_t& bitOut );
                std::uint64_t readBitsU64( std::int64_t bitCount );
                void reset();

            private:

                const std::uint8_t* stream;  // Pointer to the external bit stream. Not owned by the reader.
                const std::int64_t sizeInBytes;       // Size of the stream *in bytes*. Might include padding.
                const std::int64_t sizeInBits;        // Size of the stream *in bits*, padding *not* include.
                std::int64_t currBytePos;             // Current byte being read in the stream.
                std::int64_t nextBitPos;              // Bit position within the current byte to access next. 0 to 7.
                std::int64_t numBitsRead;             // Total bits read from the stream so far. Never includes byte-rounding padding.
        };

        constexpr std::int64_t Nil            = -1;
        constexpr std::int64_t MaxDictBits    = 12;
        constexpr std::int64_t StartBits      = 9;
        constexpr std::int64_t FirstCode      = ( 1 << ( StartBits - 1 ) ); // 256
        constexpr std::int64_t MaxDictEntries = ( 1 << MaxDictBits );   // 4096

        struct Dictionary final
        {
            struct Entry
            {
                std::int64_t code;
                std::int64_t value;
            };

            // Dictionary entries 0-255 are always reserved to the byte/ASCII range.
            std::int64_t size;
            Entry entries[MaxDictEntries];

            Dictionary();
            std::int64_t findIndex( std::int64_t code, std::int64_t value ) const;
            bool add( std::int64_t code, std::int64_t value );
            bool flush( std::int64_t& codeBitsWidth );
        };


        // Round up to the next power-of-two number, e.g. 37 => 64
        static inline std::int64_t nextPowerOfTwo( std::int64_t num )
        {
            --num;

            for ( std::size_t i = 1; i < sizeof( num ) * 8; i <<= 1 )
            {
                num = num | num >> i;
            }

            return ++num;
        }

        inline
        BitStreamWriter::BitStreamWriter()
        {
            // 8192 bits for a start (1024 bytes). It will resize if needed.
            // Default granularity is 2.
            internalInit();
            allocate( 8192 );
        }

        inline
        BitStreamWriter::BitStreamWriter( const std::int64_t initialSizeInBits, const std::int64_t growthGranularity )
        {
            internalInit();
            setGranularity( growthGranularity );
            allocate( initialSizeInBits );
        }

        inline
        BitStreamWriter::~BitStreamWriter()
        {
            if ( stream != nullptr )
            {
                std::free( stream );
            }
        }


        inline
        void BitStreamWriter::internalInit()
        {
            stream         = nullptr;
            bytesAllocated = 0;
            granularity    = 2;
            currBytePos    = 0;
            nextBitPos     = 0;
            numBitsWritten = 0;
        }

        inline
        void BitStreamWriter::allocate( std::int64_t bitsWanted )
        {
            // Require at least a byte.
            if ( bitsWanted <= 0 )
            {
                bitsWanted = 8;
            }

            // Round upwards if needed:
            if ( ( bitsWanted % 8 ) != 0 )
            {
                bitsWanted = nextPowerOfTwo( bitsWanted );
            }

            // We might already have the required count.
            const std::int64_t sizeInBytes = bitsWanted / 8;

            if ( sizeInBytes <= bytesAllocated )
            {
                return;
            }

            stream = allocBytes( sizeInBytes, stream, bytesAllocated );
            bytesAllocated = sizeInBytes;
        }

        inline
        void BitStreamWriter::appendBit( const std::int64_t bit )
        {
            const std::uint32_t mask = std::uint32_t( 1 ) << nextBitPos;
            stream[currBytePos] = ( stream[currBytePos] & ~mask ) | ( -bit & mask );
            ++numBitsWritten;

            if ( ++nextBitPos == 8 )
            {
                nextBitPos = 0;

                if ( ++currBytePos == bytesAllocated )
                {
                    allocate( bytesAllocated * granularity * 8 );
                }
            }
        }

        inline
        void BitStreamWriter::appendBitsU64( const std::uint64_t num, const std::int64_t bitCount )
        {
            assert( bitCount <= 64 );

            for ( std::int64_t b = 0; b < bitCount; ++b )
            {
                const std::uint64_t mask = std::uint64_t( 1 ) << b;
                const std::int64_t bit = !!( num & mask );
                appendBit( bit );
            }
        }

        inline
        void BitStreamWriter::appendBitString( const std::string& bitStr )
        {
            for ( std::size_t i = 0; i < bitStr.length(); ++i )
            {
                appendBit( bitStr[i] == '0' ? 0 : 1 );
            }
        }

        inline
        std::string BitStreamWriter::toBitString() const
        {
            std::string bitString;
            std::int64_t usedBytes = numBitsWritten / 8;
            std::int64_t leftovers = numBitsWritten % 8;

            if ( leftovers != 0 )
            {
                ++usedBytes;
            }

            assert( usedBytes <= bytesAllocated );

            for ( std::int64_t i = 0; i < usedBytes; ++i )
            {
                const std::int64_t nBits = ( leftovers == 0 ) ? 8 : ( i == usedBytes - 1 ) ? leftovers : 8;

                for ( std::int64_t j = 0; j < nBits; ++j )
                {
                    bitString += ( stream[i] & ( 1 << j ) ? '1' : '0' );
                }
            }

            return bitString;
        }


        inline
        std::uint8_t* BitStreamWriter::release()
        {
            std::uint8_t* oldPtr = stream;
            internalInit();
            return oldPtr;
        }

        inline
        void BitStreamWriter::setGranularity( const std::int64_t growthGranularity )
        {
            granularity = ( growthGranularity >= 2 ) ? growthGranularity : 2;
        }

        inline
        std::int64_t BitStreamWriter::getByteCount() const
        {
            std::int64_t usedBytes = numBitsWritten / 8;
            std::int64_t leftovers = numBitsWritten % 8;

            if ( leftovers != 0 )
            {
                ++usedBytes;
            }

            assert( usedBytes <= bytesAllocated );
            return usedBytes;
        }

        inline
        std::int64_t BitStreamWriter::getBitCount() const
        {
            return numBitsWritten;
        }

        inline
        const std::uint8_t* BitStreamWriter::getBitStream() const
        {
            return stream;
        }

        inline
        std::uint8_t* BitStreamWriter::allocBytes( const std::int64_t bytesWanted, std::uint8_t* oldPtr, const std::int64_t oldSize )
        {
            std::uint8_t* newMemory = static_cast<std::uint8_t*>( std::malloc( bytesWanted ) );
            std::memset( newMemory, 0, bytesWanted );

            if ( oldPtr != nullptr )
            {
                std::memcpy( newMemory, oldPtr, oldSize );
                std::free( oldPtr );
            }

            return newMemory;
        }


        inline
        BitStreamReader::BitStreamReader( const BitStreamWriter& bitStreamWriter ) : stream( bitStreamWriter.getBitStream() ), sizeInBytes( bitStreamWriter.getByteCount() ),
            sizeInBits( bitStreamWriter.getBitCount() )
        {
            reset();
        }

        inline
        BitStreamReader::BitStreamReader( const std::uint8_t* bitStream, const std::int64_t byteCount, const std::int64_t bitCount ) : stream( bitStream ), sizeInBytes( byteCount ), sizeInBits( bitCount )
        {
            reset();
        }

        inline
        bool BitStreamReader::readNextBit( std::int64_t& bitOut )
        {
            if ( numBitsRead >= sizeInBits )
            {
                return false; // We are done.
            }

            const std::uint32_t mask = std::uint32_t( 1 ) << nextBitPos;
            bitOut = !!( stream[currBytePos] & mask );
            ++numBitsRead;

            if ( ++nextBitPos == 8 )
            {
                nextBitPos = 0;
                ++currBytePos;
            }

            return true;
        }

        inline
        std::uint64_t BitStreamReader::readBitsU64( const std::int64_t bitCount )
        {
            assert( bitCount <= 64 );
            std::uint64_t num = 0;

            for ( std::int64_t b = 0; b < bitCount; ++b )
            {
                std::int64_t bit;

                if ( !readNextBit( bit ) )
                {
                    fatalError( "Failed to read bits from stream! Unexpected end." );
                    break;
                }

                // Based on a "Stanford bit-hack":
                // http://graphics.stanford.edu/~seander/bithacks.html#ConditionalSetOrClearBitsWithoutBranching
                const std::uint64_t mask = std::uint64_t( 1 ) << b;
                num = ( num & ~mask ) | ( -bit & mask );
            }

            return num;
        }

        inline
        void BitStreamReader::reset()
        {
            currBytePos = 0;
            nextBitPos  = 0;
            numBitsRead = 0;
        }

        inline
        bool BitStreamReader::isEndOfStream() const
        {
            return numBitsRead >= sizeInBits;
        }

        // ========================================================
        // class Dictionary:
        // ========================================================

        inline
        Dictionary::Dictionary()
        {
            // First 256 dictionary entries are reserved to the byte/ASCII
            // range. Additional entries follow for the character sequences
            // found in the input. Up to 4096 - 256 (MaxDictEntries - FirstCode).
            size = FirstCode;

            for ( std::int64_t i = 0; i < size; ++i )
            {
                entries[i].code  = Nil;
                entries[i].value = i;
            }
        }

        inline
        std::int64_t Dictionary::findIndex( const std::int64_t code, const std::int64_t value ) const
        {
            if ( code == Nil )
            {
                return value;
            }

            for ( std::int64_t i = 0; i < size; ++i )
            {
                if ( entries[i].code == code && entries[i].value == value )
                {
                    return i;
                }
            }

            return Nil;
        }

        inline
        bool Dictionary::add( const std::int64_t code, const std::int64_t value )
        {
            if ( size == MaxDictEntries )
            {
                fatalError( "Dictionary overflowed!" );
                return false;
            }

            entries[size].code  = code;
            entries[size].value = value;
            ++size;
            return true;
        }

        inline
        bool Dictionary::flush( std::int64_t& codeBitsWidth )
        {
            if ( size == ( 1 << codeBitsWidth ) )
            {
                ++codeBitsWidth;

                if ( codeBitsWidth > MaxDictBits )
                {
                    // Clear the dictionary (except the first 256 byte entries).
                    codeBitsWidth = StartBits;
                    size = FirstCode;
                    return true;
                }
            }

            return false;
        }

        // ========================================================
        // easyEncode() implementation:
        // ========================================================

        inline
        void easyEncode( const std::uint8_t* uncompressed, std::int64_t uncompressedSizeBytes, std::uint8_t** compressed, std::int64_t* compressedSizeBytes, std::int64_t* compressedSizeBits )
        {
            if ( uncompressed == nullptr || compressed == nullptr )
            {
                fatalError( "lzw_details::easyEncode(): Null data pointer(s)!" );
                return;
            }

            if ( uncompressedSizeBytes <= 0 || compressedSizeBytes == nullptr || compressedSizeBits == nullptr )
            {
                fatalError( "lzw_details::easyEncode(): Bad in/out sizes!" );
                return;
            }

            // LZW encoding context:
            std::int64_t code = Nil;
            std::int64_t codeBitsWidth = StartBits;
            Dictionary dictionary;
            // Output bit stream we write to. This will allocate
            // memory as needed to accommodate the encoded data.
            BitStreamWriter bitStream;

            for ( ; uncompressedSizeBytes > 0; --uncompressedSizeBytes, ++uncompressed )
            {
                const std::int64_t value = *uncompressed;
                const std::int64_t index = dictionary.findIndex( code, value );

                if ( index != Nil )
                {
                    code = index;
                    continue;
                }

                // Write the dictionary code using the minimum bit-with:
                bitStream.appendBitsU64( code, codeBitsWidth );

                // Flush it when full so we can restart the sequences.
                if ( !dictionary.flush( codeBitsWidth ) )
                {
                    // There's still space for this sequence.
                    dictionary.add( code, value );
                }

                code = value;
            }

            // Residual code at the end:
            if ( code != Nil )
            {
                bitStream.appendBitsU64( code, codeBitsWidth );
            }

            // Pass ownership of the compressed data buffer to the user pointer:
            *compressedSizeBytes = bitStream.getByteCount();
            *compressedSizeBits  = bitStream.getBitCount();
            *compressed          = bitStream.release();
        }

        // ========================================================
        // easyDecode() and helpers:
        // ========================================================

        inline
        static bool outputByte( std::int64_t code, std::uint8_t*& output, std::int64_t outputSizeBytes, std::int64_t& bytesDecodedSoFar )
        {
            if ( bytesDecodedSoFar >= outputSizeBytes )
            {
                fatalError( "Decoder output buffer too small!" );
                return false;
            }

            assert( code >= 0 && code < 256 );
            *output++ = static_cast<std::uint8_t>( code );
            ++bytesDecodedSoFar;
            return true;
        }

        inline
        static bool outputSequence( const Dictionary& dict, std::int64_t code, std::uint8_t*& output, std::int64_t outputSizeBytes, std::int64_t& bytesDecodedSoFar, std::int64_t& firstByte )
        {
            // A sequence is stored backwards, so we have to write
            // it to a temp then output the buffer in reverse.
            std::int64_t i = 0;
            std::uint8_t sequence[MaxDictEntries];

            do
            {
                assert( i < MaxDictEntries - 1 && code >= 0 );
                sequence[i++] = dict.entries[code].value;
                code = dict.entries[code].code;
            }
            while ( code >= 0 );

            firstByte = sequence[--i];

            for ( ; i >= 0; --i )
            {
                if ( !outputByte( sequence[i], output, outputSizeBytes, bytesDecodedSoFar ) )
                {
                    return false;
                }
            }

            return true;
        }

        inline
        std::int64_t easyDecode( const std::uint8_t* compressed, const std::int64_t compressedSizeBytes, const std::int64_t compressedSizeBits, std::uint8_t* uncompressed, const std::int64_t uncompressedSizeBytes )
        {
            if ( compressed == nullptr || uncompressed == nullptr )
            {
                fatalError( "lzw_details::easyDecode(): Null data pointer(s)!" );
                return 0;
            }

            if ( compressedSizeBytes <= 0 || compressedSizeBits <= 0 || uncompressedSizeBytes <= 0 )
            {
                fatalError( "lzw_details::easyDecode(): Bad in/out sizes!" );
                return 0;
            }

            std::int64_t code          = Nil;
            std::int64_t prevCode      = Nil;
            std::int64_t firstByte     = 0;
            std::int64_t bytesDecoded  = 0;
            std::int64_t codeBitsWidth = StartBits;
            // We'll reconstruct the dictionary based on the
            // bit stream codes. Unlike Huffman encoding, we
            // don't store the dictionary as a prefix to the data.
            Dictionary dictionary;
            BitStreamReader bitStream( compressed, compressedSizeBytes, compressedSizeBits );

            // We check to avoid an overflow of the user buffer.
            // If the buffer is smaller than the decompressed size,
            // fatalError() is called. If that doesn't throw or
            // terminate we break the loop and return the current
            // decompression count.
            while ( !bitStream.isEndOfStream() )
            {
                assert( codeBitsWidth <= MaxDictBits );
                code = static_cast<int>( bitStream.readBitsU64( codeBitsWidth ) );

                if ( prevCode == Nil )
                {
                    if ( !outputByte( code, uncompressed, uncompressedSizeBytes, bytesDecoded ) )
                    {
                        break;
                    }

                    firstByte = code;
                    prevCode  = code;
                    continue;
                }

                if ( code >= dictionary.size )
                {
                    if ( !outputSequence( dictionary, prevCode, uncompressed, uncompressedSizeBytes, bytesDecoded, firstByte ) )
                    {
                        break;
                    }

                    if ( !outputByte( firstByte, uncompressed, uncompressedSizeBytes, bytesDecoded ) )
                    {
                        break;
                    }
                }
                else
                {
                    if ( !outputSequence( dictionary, code, uncompressed, uncompressedSizeBytes, bytesDecoded, firstByte ) )
                    {
                        break;
                    }
                }

                dictionary.add( prevCode, firstByte );

                if ( dictionary.flush( codeBitsWidth ) )
                {
                    prevCode = Nil;
                }
                else
                {
                    prevCode = code;
                }
            }

            return bytesDecoded;
        }

    }//namespace lzw_details


    inline auto compress() noexcept
    {
        return []<typename OutputIterator>( std::uint8_t const* begin, std::uint8_t const* end, OutputIterator output ) noexcept
        {
            std::int64_t compressed_size_bytes = 0;
            std::int64_t compressed_size_bits = 0;
            std::uint8_t* compressed_data = nullptr;
            lzw_details::easyEncode( begin, end-begin, &compressed_data, &compressed_size_bytes, &compressed_size_bits );
            std::copy_n( compressed_data, compressed_size_bytes, output );
            std::free( compressed_data );
            return compressed_size_bits;
        };
    }

    inline auto uncompress( std::int64_t uncompressed_size_bytes, std::int64_t compressed_size_bits ) noexcept
    {
        return [=]<typename OutputIterator>( std::uint8_t const* begin, std::uint8_t const* end, OutputIterator output ) noexcept
        {
            std::vector<std::uint8_t> uncompressed( uncompressed_size_bytes, 0 );
            lzw_details::easyDecode( begin, end-begin, compressed_size_bits, uncompressed.data(), uncompressed_size_bytes );
            std::copy(uncompressed.begin(), uncompressed.end(), output);
        };
    }

}//namespace lzw

#endif//LZW_HPP_INCLUDED_ADSFLKSALKJSALKJASLKJSALSAKJSALKJLKJASLKFSDJLKASJFDFFFFFFFFF

