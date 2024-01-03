#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <hip/hip_runtime.h>
// #include <rocm_smi/rocm_smi.h>

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
    // rsmi_status_t ret;
    // ret = rsmi_init(0);

    // random float generators
    random_device dev;
    default_random_engine eng(dev());
    uniform_real_distribution<> dis(FLOAT_MIN, FLOAT_MAX);

    // allocate memory for warmup float transfers (a destination on the device, and a source and a destination on the host)
    float* warmupDeviceMem;
    hipError_t err = hipMalloc((void**)&warmupDeviceMem, sizeof(float)*WARMUP_FLOATS);
    handleHipError(err, "warmup device malloc");
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

    rsmi_pcie_bandwidth_t* bandwidth;
    ret = rsmi_dev_pci_bandwidth_get(0, bandwidth);
    cout << bandwidth->transfer_rate.current << endl;

    // warmup transfers
    err = hipMemcpyHtoD(warmupDeviceMem, warmupHostSourceMem, sizeof(float)*WARMUP_FLOATS);
    handleHipError(err, "warmup HtD transfer");
    err = hipMemcpyDtoH(warmupHostDestMem, warmupDeviceMem, sizeof(float)*WARMUP_FLOATS);
    handleHipError(err, "warmup DtH transfer");

    // compare (ensure transfer works)
    for (int i = 0; i < WARMUP_FLOATS; i++) {
        if (warmupHostDestMem[i] != warmupHostSourceMem[i]) {
            cout << "copied data did not match at index " << i;
            exit(1);
        }
    }

    // free memory
    err = hipFree(warmupDeviceMem);
    handleHipError(err, "warmup device data free");
    err = hipHostFree(warmupHostSourceMem);
    handleHipError(err, "warmup host source data free");
    err = hipHostFree(warmupHostDestMem);
    handleHipError(err, "warmup host dest data free");

    // main D2H / H2D testing

    unsigned int float_count = 48;

    while (true) {

        cout << "current cycle: " << float_count << " floats" << endl;
        // allocate memory for testing
        float* deviceMem;
        err = hipMalloc((void**)&deviceMem, sizeof(float)*float_count);
        handleHipError(err, "device mem malloc");
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

        // HOST to DEVICE test

        // hipEvent setup
        hipEvent_t startHtD, stopHtD;
        hipEventCreate(&startHtD);
        hipEventCreate(&stopHtD);
        float durationHtD;

        // get before time
        hipEventRecord(startHtD, NULL);

        // primary transfer
        err = hipMemcpyHtoD(deviceMem, hostSourceMem, sizeof(float)*float_count);
        handleHipError(err, "host to device transfer");

        // get after time
        hipEventRecord(stopHtD, NULL);
        hipEventSynchronize(stopHtD);
        hipEventElapsedTime(&durationHtD, startHtD, stopHtD);

        cout << "host to device time taken: " << durationHtD << "ms" << endl;

        // DEVICE to HOST test

        hipEvent_t startDtH, stopDtH;
        hipEventCreate(&startDtH);
        hipEventCreate(&stopDtH);
        float durationDtH;

        // get before time
        hipEventRecord(startDtH, NULL);

        // primary transfer
        err = hipMemcpyDtoH(hostDestMem, deviceMem, sizeof(float)*float_count);
        handleHipError(err, "device to host transfer");

        // get after time
        hipEventRecord(stopDtH, NULL);
        hipEventSynchronize(stopDtH);
        hipEventElapsedTime(&durationDtH, startDtH, stopDtH);

        cout << "device to host time taken: " << durationDtH << "ms" << endl;


        // free memory
        err = hipFree(deviceMem);
        handleHipError(err, "free device memory");
        err = hipHostFree(hostSourceMem);
        handleHipError(err, "free host source memory");
        err = hipHostFree(hostDestMem);
        handleHipError(err, "free host dest memory");

        float_count *= 2;
    }


}