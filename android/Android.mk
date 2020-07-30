LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)
LOCAL_PACKAGE_NAME			:= PPOServerApp
LOCAL_MODULE_TAGS			:= optional
LOCAL_SDK_VERSION			:= current
LOCAL_CERTIFICATE			:= platform
LOCAL_PROGUARD_ENABLED		:= disabled
LOCAL_SRC_FILES 			:= $(call all-subdir-java-files)
LOCAL_JNI_SHARED_LIBRARIES	:= libppo
include $(BUILD_PACKAGE)

include $(call all-makefiles-under,$(LOCAL_PATH))
