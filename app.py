import streamlit as st
from transformers import pipeline
from PIL import Image

# Streamlit页面配置
st.set_page_config(
    page_title="年龄分类器",
    page_icon="👤",
    layout="centered"
)

# 标题
st.title("👤 ViT年龄分类器")
st.markdown("上传一张人脸照片，AI会判断年龄范围！")

# 加载模型
@st.cache_resource
def load_classifier():
    return pipeline("image-classification",
                   model="nateraw/vit-age-classifier")

# 文件上传
uploaded_file = st.file_uploader(
    "选择一张图片",
    type=["jpg", "jpeg", "png"],
    help="上传包含人脸的图片"
)

if uploaded_file is not None:
    # 打开用户上传的图片
    image = Image.open(uploaded_file).convert("RGB")
    
    # 显示图片
    st.image(image, caption="上传的图片", use_column_width=True)
    
    # 分析按钮
    if st.button("🔍 分析年龄"):
        with st.spinner("AI正在分析..."):
            # 加载模型
            classifier = load_classifier()
            
            # 预测
            predictions = classifier(image)
            
            # 排序
            predictions = sorted(predictions, 
                               key=lambda x: x['score'], 
                               reverse=True)
            
            # 显示结果
            st.success("✅ 分析完成！")
            st.metric("预测年龄范围", 
                     predictions[0]['label'],
                     f"{predictions[0]['score']:.2%}")
            
            # 显示详细结果
            st.subheader("📊 详细置信度")
            for pred in predictions:
                st.progress(pred['score'], 
                          text=f"{pred['label']}: {pred['score']:.2%}")
