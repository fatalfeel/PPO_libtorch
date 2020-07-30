LOCAL_PATH := $(call my-dir)
##############################
########## libppo.so #########
##############################
include $(CLEAR_VARS)
LOCAL_MODULE := libppo
LOCAL_CFLAGS := -std=c++14
LOCAL_CPP_FEATURES := rtti

LOCAL_C_INCLUDES := \
$(LOCAL_PATH)/../../cp_src \

LOCAL_SRC_FILES := \
../../cp_src/actorcritic.cpp \
../../cp_src/categorical.cpp \
../../cp_src/flowcontrol.cpp \
../../cp_src/gamecontent.cpp \
../../cp_src/link_list.cpp \
../../cp_src/servermessage.cpp

LOCAL_LDLIBS := \
-lz -ldl -llog -pthread

#LOCAL_SHARED_LIBRARIES := \
#liblog

LOCAL_CFLAGS += -O0 -g
LOCAL_CPPFLAGS += -O0 -g
LOCAL_STRIP_MODULE := false
include $(BUILD_SHARED_LIBRARY)
