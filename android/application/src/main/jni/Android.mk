LOCAL_PATH := $(call my-dir)
##############################
########## libppo.so #########
##############################
include $(CLEAR_VARS)
LOCAL_MODULE := libppo
LOCAL_CFLAGS := -std=c++14
LOCAL_CPP_FEATURES := rtti

LOCAL_C_INCLUDES := \
$(LOCAL_PATH)/../../../../../cpp_src \
$(LOCAL_PATH)/../../../../../cpp_src/libtorch/include \
$(LOCAL_PATH)/../../../../../cpp_src/libtorch/include/torch/csrc/api/include
                
LOCAL_SRC_FILES := \
../../../../../cpp_src/jniapi.cpp

#LOCAL_WHOLE_STATIC_LIBRARIES := \
#LOCAL_STATIC_LIBRARIES := \
#liblog

LOCAL_LDLIBS := -ldl -llog -lz

include $(BUILD_SHARED_LIBRARY)
