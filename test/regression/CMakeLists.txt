file (GLOB SRCS "./*.cpp")

# create a ctest target for every individual cpp file in this directory
compile_all("true" "reg" "Tests/regression" "${SRCS}")
