#ifndef FBXYAXRPGSNHIXESHOGNYHPVEWWVSRSRJLQPIRIFENBGNMGFLJNMWXDNQLHKOAGBNYGBJRLBD
#define FBXYAXRPGSNHIXESHOGNYHPVEWWVSRSRJLQPIRIFENBGNMGFLJNMWXDNQLHKOAGBNYGBJRLBD

namespace ceras
{
    #ifdef NDEBUG
        inline constexpr unsigned long debug_mode = 0;
    #else
        inline constexpr unsigned long debug_mode = 1;
    #endif

    #ifdef CUDA
        inline constexpr unsigned long cuda_mode = 1;
        inline int visible_device = 0; // using GPU 0 by default
    #else
        inline constexpr unsigned long cuda_mode = 0;
    #endif

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

