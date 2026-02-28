import streamlit as st
from PIL import Image
import time

# 页面配置 - 必须在最前面
st.set_page_config(
    page_title="Streamlit组件演示",
    page_icon="🎈",
    layout="centered"
)

# App title
st.title("🎈 Streamlit组件演示")

# 添加一些说明文字
st.markdown("""
这个App演示了Streamlit的基本组件：
- 📝 `st.write` - 显示文本
- 🖼️ `st.file_uploader` - 上传文件
- ⏳ `st.spinner` - 加载动画
- 🎯 `st.button` - 按钮交互
- 📸 `st.image` - 显示图片
""")

st.divider()  # 添加分割线

# Write some text
st.write("### 欢迎使用这个演示App 👋")
st.write("上传一张图片试试看！")

# File uploader for image and audio
uploaded_image = st.file_uploader(
    "上传一张图片", 
    type=["jpg", "jpeg", "png"],
    help="支持JPG、JPEG、PNG格式"
)

# Display image with spinner
if uploaded_image is not None:
    with st.spinner("正在加载图片..."):
        time.sleep(1)  # Simulate a delay
        image = Image.open(uploaded_image)
        
        # 显示图片信息
        st.success("✅ 图片上传成功！")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**图片格式:** {image.format}")
        with col2:
            st.write(f"**图片尺寸:** {image.size[0]} x {image.size[1]}")
        
        # 显示图片
        st.image(image, caption="你上传的图片", use_column_width=True)

st.divider()

# Button interaction
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🎯 点击我试试", type="primary", use_container_width=True):
        st.balloons()  # 添加气球动画
        st.write("🎉 太棒了！你点击了按钮！")
        st.snow()  # 添加雪花动画
