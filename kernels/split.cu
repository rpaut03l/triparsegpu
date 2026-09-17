#include "common.cuh"

// Mark newline positions: one thread per byte.
extern "C" __global__ void mark_newlines(
        const unsigned char* __restrict__ buf, long long n,
        unsigned char* __restrict__ flag)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) flag[i] = (buf[i] == '\n') ? 1 : 0;
}

// Per line, skip K space-delimited fields; Content starts at the byte after
// the K-th space. Fixed-column LogHub formats (BGL, HDFS, OpenStack) work
// this way. Runs of spaces count as one delimiter.
extern "C" __global__ void content_start(
        const unsigned char* __restrict__ buf,
        const long long* __restrict__ line_start,
        const long long* __restrict__ line_end,     // exclusive, excludes '\n'
        long long nlines, int k,
        long long* __restrict__ out_start)
{
    long long L = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (L >= nlines) return;
    const long long a = line_start[L], b = line_end[L];
    int seen = 0;
    long long i = a;
    while (i < b && seen < k) {
        while (i < b && buf[i] != ' ') ++i;         // to end of field
        while (i < b && buf[i] == ' ') ++i;         // skip the run of spaces
        ++seen;
    }
    out_start[L] = (seen == k) ? i : a;             // fall back to whole line
}


// OpenStack: <Logrecord> <Date> <Time> <Pid> <Level> <Component> [<ADDR>] <Content>
// The bracketed field contains spaces, so key off the first ']' instead.
extern "C" __global__ void content_start_bracket(
        const unsigned char* __restrict__ buf,
        const long long* __restrict__ line_start,
        const long long* __restrict__ line_end,
        long long nlines,
        long long* __restrict__ out_start)
{
    long long L = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (L >= nlines) return;
    const long long a = line_start[L], b = line_end[L];
    long long i = a;
    while (i < b && buf[i] != ']') ++i;
    if (i >= b) { out_start[L] = a; return; }       // no bracket: whole line
    ++i;                                            // past ']'
    while (i < b && buf[i] == ' ') ++i;
    out_start[L] = i;
}
