# 使用包含Python的基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，防止Python缓冲输出
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 安装系统依赖（如果需要）
# RUN apt-get update && apt-get install -y --no-install-recommends gcc

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
# 如果需要使用GPU，请确保基础镜像和torch版本兼容
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码到容器中
COPY ./app /app/app
COPY .env /app/

# 暴露端口
EXPOSE 8000

# 运行应用的命令
# 使用 uvicorn 启动服务器
# --host 0.0.0.0 使其可以从外部访问
# --port 8000 监听的端口
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]