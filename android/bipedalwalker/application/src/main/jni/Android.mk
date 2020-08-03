LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := libc10
LOCAL_SRC_FILES := \
../../../../../libtorch/lib/libc10.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libclog
LOCAL_SRC_FILES := \
../../../../../libtorch/lib/libclog.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libcpuinfo
LOCAL_SRC_FILES := \
../../../../../libtorch/lib/libcpuinfo.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libeigen_blas
LOCAL_SRC_FILES := \
../../../../../libtorch/lib/libeigen_blas.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libnnpack
LOCAL_SRC_FILES := \
../../../../../libtorch/lib/libnnpack.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libpthreadpool
LOCAL_SRC_FILES := \
../../../../../libtorch/lib/libpthreadpool.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libpytorch_qnnpack
LOCAL_SRC_FILES := \
../../../../../libtorch/lib/libpytorch_qnnpack.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libtorch
LOCAL_SRC_FILES := \
../../../../../libtorch/lib/libtorch.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libtorch_cpu
LOCAL_SRC_FILES := \
../../../../../libtorch/lib/libtorch_cpu.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libXNNPACK
LOCAL_SRC_FILES := \
../../../../../libtorch/lib/libXNNPACK.a
include $(PREBUILT_STATIC_LIBRARY)
##############################
########## libppo.so #########
##############################
include $(CLEAR_VARS)
LOCAL_MODULE := libppo
LOCAL_CPP_FEATURES := rtti
LOCAL_CFLAGS := -pthread -std=gnu++14 -DANDROID -DUSE_PTHREADPOOL -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_VULKAN_WRAPPER
LOCAL_CPPFLAGS := -pthread -std=gnu++14 -DANDROID -DUSE_PTHREADPOOL -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_VULKAN_WRAPPER
	
APP_DEBUG := $(strip $(NDK_DEBUG))
ifeq ($(APP_DEBUG),1)
	LOCAL_STRIP_MODULE := false
endif

LOCAL_C_INCLUDES := \
$(LOCAL_PATH)/../../../../../../bipedalwalker_cpp \
$(LOCAL_PATH)/../../../../../libtorch/include \
$(LOCAL_PATH)/../../../../../libtorch/include/torch/csrc/api/include
                
LOCAL_SRC_FILES := \
../../../../../../bipedalwalker_cpp/jniapi.cpp \
../../../../../../bipedalwalker_cpp/link_list.cpp \
../../../../../../bipedalwalker_cpp/servermessage.cpp

LOCAL_WHOLE_STATIC_LIBRARIES := \
libc10 \
libclog \
libcpuinfo \
libeigen_blas \
libnnpack \
libpthreadpool \
libpytorch_qnnpack \
libtorch \
libtorch_cpu \
libXNNPACK

LOCAL_LDLIBS := -ldl -llog -lz

include $(BUILD_SHARED_LIBRARY)
