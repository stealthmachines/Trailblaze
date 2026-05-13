/*
 * Minimal SHA-256 Implementation (~2 KB compiled)
 * Ported from stealthmachines/AnalogContainer1
 * https://github.com/stealthmachines/AnalogContainer1
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#ifndef SHA256_MINIMAL_H
#define SHA256_MINIMAL_H

#include <stdint.h>
#include <stddef.h>

void sha256_hash(const void *data, size_t len, uint8_t hash_out[32]);

#endif /* SHA256_MINIMAL_H */
