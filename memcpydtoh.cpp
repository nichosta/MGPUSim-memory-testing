#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <random>
#include <hip/hip_runtime.h>

using namespace std;

#define WARMUP_FLOATS 48

#define FLOAT_MIN 0.0
#define FLOAT_MAX 1.0

int main() {
    // random float generators
    random_device dev;
    default_random_engine eng(dev());
    uniform_real_distribution<> dis(FLOAT_MIN, FLOAT_MAX);

    // allocate memory for warmup float transfers
    float* warmupDeviceMem;
    int err = hipMalloc((void**)&warmupDeviceMem, sizeof(float)*WARMUP_FLOATS);
    if (err != 0) {
        cout << "error in warmup device malloc: " << err << endl;
        exit(err);
    }
    float* warmupHostMem;
    err = hipHostMalloc((void**)&warmupHostMem, sizeof(float)*WARMUP_FLOATS);
    if (err != 0) {
        cout << "error in warmup host malloc: " << err << endl;
        exit(err);
    }

    // generate warmup floats
    for (int i = 0; i < WARMUP_FLOATS; i++) {
        warmupDeviceMem[i] = dis(eng);
    }

    // warmup transfer
    err = hipMemcpyDtoH(warmupHostMem, warmupDeviceMem, sizeof(float)*WARMUP_FLOATS);
    if (err != 0) {
        cout << "error in warmup transfer: " << err << endl;
        exit(err);
    }

    // compare (ensure transfer works)
    for (int i = 0; i < WARMUP_FLOATS; i++) {
        if (warmupDeviceMem[i] != warmupHostMem[i]) {
            cout << "copied data did not match at index " << i;
            exit(1);
        }
    }

    // free memory
    err = hipFree(warmupDeviceMem);
    if (err != 0) {
        cout << "error in warmup device free: " << err << endl;
        exit(err);
    }
    err = hipHostFree(warmupHostMem);
    if (err != 0) {
        cout << "error in warmup host free: " << err << endl;
        exit(err);
    }

    // main D2H testing

    int float_count = 48;

    while (true) {

        cout << "current cycle: " << float_count << " floats" << endl;
        // allocate memory for testing
        float* deviceMem;
        err = hipMalloc((void**)&deviceMem, sizeof(float)*float_count);
        if (err != 0) {
            cout << "error in primary device malloc: " << err << endl;
            exit(err);
        }
        float* hostMem;
        err = hipHostMalloc((void**)&hostMem, sizeof(float)*float_count);
        if (err != 0) {
            cout << "error in primary host malloc: " << err << endl;
            exit(err);
        }

        // generate primary floats
        for (int i = 0; i < float_count; i++) {
            deviceMem[i] = dis(eng);
        }

        // get before realtime
        struct timeval before_time;
        gettimeofday(&before_time, NULL);

        // get before clock time
        // clock_t before_clocktime = clock();

        // primary transfer
        err = hipMemcpyDtoH(hostMem, deviceMem, sizeof(float)*float_count);
        if (err != 0) {
            cout << "error in primary transfer: " << err << endl;
            exit(err);
        }

        // get after realtime
        struct timeval after_time;
        gettimeofday(&after_time, NULL);

        // get after clock time
        // clock_t after_clocktime = clock();

        cout << float_count << " floats realtime taken: " << (after_time.tv_sec * 1000000 + after_time.tv_usec) - (before_time.tv_sec * 1000000 + before_time.tv_usec) << " microseconds" << endl;

        // cout << float_count << " floats system time taken: " << after_clocktime - before_clocktime << endl;

        // free memory
        err = hipFree(deviceMem);
        if (err != 0) {
            cout << "error in primary device free: " << err << endl;
            exit(err); 
        }
        err = hipHostFree(hostMem);
        if (err != 0) {
            cout << "error in primary host free: " << err << endl;
            exit(err);
        }

        float_count *= 2;
    }


}