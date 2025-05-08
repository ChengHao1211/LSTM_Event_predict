FROM ubuntu:22.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libssl-dev \
    nlohmann-json3-dev \
    openssl

# 设置工作目录
WORKDIR /app

# 复制文件
COPY Source.cpp /app/
COPY httplib.h /app/
COPY user_behavior_model.onnx /app/
COPY scaler_params.json /app/
COPY model_metadata.json /app/

# 编译代码
RUN g++ -std=c++17 -O2 Source.cpp -o api_server -lssl -lcrypto -pthread

# 运行应用
CMD ["./api_server"]