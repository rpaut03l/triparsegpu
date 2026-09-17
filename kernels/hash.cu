#include "common.cuh"

// One thread per token.
//  - FNV-1a 64-bit hash over the token's bytes
//  - classify the token: if it looks like a number / IP / hex, it is a
//    wildcard (<*>) and gets hash 0 so all wildcards compare equal.
extern "C" __global__ void hash_tokens(
        const unsigned char* __restrict__ buf,
        const long long* __restrict__ tstart,
        const long long* __restrict__ tend,
        long long ntok,
        unsigned long long* __restrict__ out_hash,
        unsigned char* __restrict__ out_wild)
{
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= ntok) return;

    const long long a = tstart[t], b = tend[t];
    const int len = (int)(b - a);

    unsigned long long h = 1469598103934665603ULL;   // FNV offset basis
    int digits = 0, dots = 0, alpha = 0;

    for (long long i = a; i < b; ++i) {
        unsigned char c = buf[i];
        h ^= (unsigned long long)c;
        h *= 1099511628211ULL;                       // FNV prime
        digits += (c >= '0' && c <= '9');
        dots   += (c == '.');
        alpha  += ((c|32) >= 'a' && (c|32) <= 'z');
    }

    // a token that already contains the wildcard marker is templated text:
    // hash it literally so blk_<*> != <*> != <*>:<*>
    bool has_marker = false;
    for (long long i = a; i + 2 < b; ++i)
        if (buf[i] == '<' && buf[i+1] == '*' && buf[i+2] == '>') { has_marker = true; break; }

    // ignore leading/trailing punctuation when deciding if this is a number
    long long ca = a, cb = b;
    while (ca < cb && !((buf[ca] >= '0' && buf[ca] <= '9') ||
                        ((buf[ca]|32) >= 'a' && (buf[ca]|32) <= 'z'))) ++ca;
    while (cb > ca && !((buf[cb-1] >= '0' && buf[cb-1] <= '9') ||
                        ((buf[cb-1]|32) >= 'a' && (buf[cb-1]|32) <= 'z'))) --cb;
    int cd = 0, cdots = 0, calpha = 0;
    for (long long i = ca; i < cb; ++i) {
        unsigned char c = buf[i];
        cd     += (c >= '0' && c <= '9');
        cdots  += (c == '.');
        calpha += (((c|32) >= 'a' && (c|32) <= 'z'));
    }
    const int clen = (int)(cb - ca);
    const bool hexpre = (clen > 2 && buf[ca] == '0' && (buf[ca+1]|32) == 'x');
    const bool all_digits = (clen > 0 && cd == clen);
    const bool ipv4 = (cdots == 3 && cd + cdots == clen);
    const bool mostly_digits = (cd > 0 && calpha == 0 && cd * 2 >= clen);

    const bool wild = !has_marker && (all_digits || ipv4 || hexpre || mostly_digits);
    (void)digits; (void)dots; (void)alpha; (void)len;

    out_wild[t] = wild ? 1 : 0;
    out_hash[t] = wild ? 0ULL : h;                   // 0 == <*>
}
