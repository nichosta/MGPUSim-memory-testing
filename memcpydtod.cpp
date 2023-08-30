#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <hip/hip_runtime.h>

using namespace std;

#define WARMUP_FLOATS 48

#define FLOAT_MIN 0.0
#define FLOAT_MAX 1.0

void handleHipError(hipError_t err, const char* sourceName) {
    if (err != 0) {
        cout << "error in " << sourceName << " : " << hipGetErrorName(err);
        exit(err);
    }
}

int main() {
    // random float generators
    random_device dev;
    default_random_engine eng(dev());
    uniform_real_distribution<> dis(FLOAT_MIN, FLOAT_MAX);

    // allocate memory for warmup float transfers
    float* warmupDeviceFirstMem;
    hipError_t err = hipMalloc((void**)&warmupDeviceFirstMem, sizeof(float)*WARMUP_FLOATS);
    handleHipError(err, "warmup device first malloc");
    float* warmupDeviceSecondMem;
    err = hipMalloc((void**)&warmupDeviceSecondMem, sizeof(float)*WARMUP_FLOATS);
    handleHipError(err, "warmup device second malloc");
    float* warmupHostSourceMem;
    err = hipHostMalloc((void**)&warmupHostSourceMem, sizeof(float)*WARMUP_FLOATS);
    handleHipError(err, "warmup host source malloc");
    float* warmupHostDestMem;
    err = hipHostMalloc((void**)&warmupHostDestMem, sizeof(float)*WARMUP_FLOATS);
    handleHipError(err, "warmup host dest malloc");

    // generate warmup floats
    for (int i = 0; i < WARMUP_FLOATS; i++) {
        warmupHostSourceMem[i] = dis(eng);
    }

    // warmup transfers
    err = hipMemcpyHtoD(warmupDeviceFirstMem, warmupHostSourceMem, sizeof(float)*WARMUP_FLOATS);
    handleHipError(err, "warmup HtD transfer");
    err = hipMemcpyDtoD(warmupDeviceSecondMem, warmupDeviceFirstMem, sizeof(float)*WARMUP_FLOATS);
    handleHipError(err, "warmup DtD transfer");
    err = hipMemcpyDtoH(warmupHostDestMem, warmupDeviceSecondMem, sizeof(float)*WARMUP_FLOATS);
    handleHipError(err, "warmup DtH transfer");

    // compare (ensure transfer works)
    for (int i = 0; i < WARMUP_FLOATS; i++) {
        if (warmupHostDestMem[i] != warmupHostSourceMem[i]) {
            cout << "copied data did not match at index " << i;
            exit(1);
        }
    }

    // free memory
    err = hipFree(warmupDeviceFirstMem);
    handleHipError(err, "warmup device first data free");
    err = hipFree(warmupDeviceSecondMem);
    handleHipError(err, "warmup device second data free");
    err = hipHostFree(warmupHostSourceMem);
    handleHipError(err, "warmup host source data free");
    err = hipHostFree(warmupHostDestMem);
    handleHipError(err, "warmup host dest data free");

    // main D2D testing

    unsigned int float_count = 48;

    while (true) {

        cout << "current cycle: " << float_count << " floats" << endl;
        // allocate memory for testing
        float* deviceFirstMem;
        err = hipMalloc((void**)&deviceFirstMem, sizeof(float)*float_count);
        handleHipError(err, "device first mem malloc");
        float* deviceSecondMem;
        err = hipMalloc((void**)&deviceSecondMem, sizeof(float)*float_count);
        handleHipError(err, "device second mem malloc");
        float* hostSourceMem;
        err = hipHostMalloc((void**)&hostSourceMem, sizeof(float)*float_count);
        handleHipError(err, "host source mem malloc");
        float* hostDestMem;
        err = hipHostMalloc((void**)&hostDestMem, sizeof(float)*float_count);
        handleHipError(err, "host dest mem malloc");

        // generate primary floats
        for (int i = 0; i < float_count; i++) {
            hostSourceMem[i] = dis(eng);
        }

        // DEVICE to DEVICE test

        // host to device transfer
        err = hipMemcpyHtoD(deviceFirstMem, hostSourceMem, sizeof(float)*float_count);
        handleHipError(err, "host to device transfer");

        // hipEvent setup
        hipEvent_t startDtD, stopDtD;
        hipEventCreate(&startDtD);
        hipEventCreate(&stopDtD);
        float durationDtD;

        // get before time
        hipEventRecord(startDtD, NULL);

        // primary transfer
        err = hipMemcpyDtoD(deviceSecondMem, deviceFirstMem, sizeof(float)*float_count);
        handleHipError(err, "device to device transfer");

        // get after time
        hipEventRecord(stopDtD, NULL);
        hipEventSynchronize(stopDtD);
        hipEventElapsedTime(&durationDtD, startDtD, stopDtD);

        cout << "device to device time taken: " << durationDtD << "ms" << endl;

        // device to host transfer
        err = hipMemcpyDtoH(hostDestMem, deviceSecondMem, sizeof(float)*float_count);
        handleHipError(err, "device to host transfer");

        // free memory
        err = hipFree(deviceFirstMem);
        handleHipError(err, "free first device memory");
        err = hipFree(deviceSecondMem);
        handleHipError(err, "free second device memory");
        err = hipHostFree(hostSourceMem);
        handleHipError(err, "free host source memory");
        err = hipHostFree(hostDestMem);
        handleHipError(err, "free host dest memory");

        float_count *= 2;
    }


}