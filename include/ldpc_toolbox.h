#ifndef _LDPC_TOOLBOX_H
#define _LDPC_TOOLBOX_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

void *ldpc_toolbox_decoder_ctor(const char *alist, const char *implementation,
                                const char *puncturing);
void ldpc_toolbox_decoder_dtor(void *decoder);
int32_t ldpc_toolbox_decoder_decode_f64(void *decoder,
                                        uint8_t *output, size_t output_len,
                                        const double *llrs, size_t llrs_len,
                                        uint32_t max_iterations);
int32_t ldpc_toolbox_decoder_decode_f32(void *decoder,
                                        uint8_t *output, size_t output_len,
                                        const float *llrs, size_t llrs_len,
                                        uint32_t max_iterations);

void *ldpc_toolbox_encoder_ctor(const char *alist, const char *puncturing);
void ldpc_toolbox_encoder_dtor(void *encoder);
void ldpc_toolbox_encoder_encode(void *encoder,
                                 uint8_t *output, size_t output_len,
                                 const uint8_t *input, size_t input_len);
#ifdef __cplusplus
}
#endif

#endif /* _LDPC_TOOLBOX_H */
