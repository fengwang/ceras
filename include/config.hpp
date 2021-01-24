#ifndef FBXYAXRPGSNHIXESHOGNYHPVEWWVSRSRJLQPIRIFENBGNMGFLJNMWXDNQLHKOAGBNYGBJRLBD
#define FBXYAXRPGSNHIXESHOGNYHPVEWWVSRSRJLQPIRIFENBGNMGFLJNMWXDNQLHKOAGBNYGBJRLBD

namespace ceras
{
    inline constexpr unsigned long version = 20210124UL;

    #ifdef NDEBUG
        inline constexpr unsigned long debug_mode = 0;
    #else
        inline constexpr unsigned long debug_mode = 1;
    #endif

    #ifdef CUDA
        inline constexpr unsigned long cuda_mode = 1;
    #else
        inline constexpr unsigned long cuda_mode = 0;
    #endif
    inline int visible_device = 0; // using GPU 0 by default
    inline unsigned long cuda_gemm_threshold = 0UL; // will be updated if in CUDA mode, always assume float multiplications as double is rearly used

    inline constexpr double eps = 1.0e-8;

    //
    // some layers, such as batch normalization and drop out, behave differently during the training and the testing time
    //
    // 1 for learning
    // 0 for prediction/test
    //
    inline int learning_phase = 1;
}

#endif//FBXYAXRPGSNHIXESHOGNYHPVEWWVSRSRJLQPIRIFENBGNMGFLJNMWXDNQLHKOAGBNYGBJRLBD

