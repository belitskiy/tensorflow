alias(
    name = "nccl",
    actual = "@cuda_nccl//:nccl",
    visibility = ["//visibility:public"],
)

alias(
    name = "nccl_headers",
    actual = "@cuda_nccl//:headers",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl_config",
    hdrs = ["nccl_config.h"],
    include_prefix = "third_party/nccl",
    visibility = ["//visibility:public"],
)
