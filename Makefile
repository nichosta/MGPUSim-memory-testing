HIP_PATH=/opt/rocm-5.6.0/include
HIPCC=/opt/rocm-5.6.0/hip/bin/hipcc

devicehost: memcpydtoh.cpp
	$(HIPCC) $^ -o $@

devicedevice: memcpydtod.cpp
	$(HIPCC) $^ -o $@

clean:
	rm -f devicehost
	rm -f devicedevice
