import streamlit as st
from PIL import Image
import time
from transformers import pipeline
import requests
from io import BytesIO

# 页面配置 - 必须在最前面
st.set_page_config(
    page_title="图片故事生成器",
    page_icon="📖",
    layout="wide"
)

# 应用标题和说明
st.title("📖 AI图片故事生成器")
st.markdown("""
上传一张图片，让AI帮你创作一个有趣的故事！
AI会先分析图片内容，然后根据图片中的元素生成一个独特的故事。
""")

# 侧边栏：模型设置
with st.sidebar:
    st.header("⚙️ 设置")
    
    # 故事风格选择
    story_style = st.selectbox(
        "选择故事风格",
        ["奇幻冒险", "温馨治愈", "悬疑惊悚", "科幻未来", "童话寓言"],
        index=0
    )
    
    # 故事长度
    story_length = st.slider(
        "故事长度",
        min_value=50,
        max_value=300,
        value=150,
        step=50,
        help="生成故事的大致字数"
    )
    
    st.divider()
    
    # 关于
    st.markdown("""
    ### ℹ️ 关于
    这个App使用AI模型：
    1. **图片分析**: BLIP模型
    2. **故事生成**: GPT-2
    """)

# 加载模型（使用缓存）
@st.cache_resource
def load_models():
    """加载AI模型"""
    try:
        with st.spinner("正在加载图片分析模型..."):
            # 图片描述模型
            image_to_text = pipeline("image-to-text", 
                                   model="Salesforce/blip-image-captioning-base")
        
        with st.spinner("正在加载故事生成模型..."):
            # 故事生成模型
            story_generator = pipeline("text-generation",
                                     model="gpt2",
                                     max_new_tokens=300)
        
        return image_to_text, story_generator
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None, None

# 生成故事的函数
def generate_story(image_description, style, length):
    """根据图片描述生成故事"""
    
    # 根据风格设置故事开头
    style_prompts = {
        "奇幻冒险": f"在一个神奇的世界里，{image_description}。勇敢的冒险者发现了这个景象，一段奇幻的旅程就此开始。",
        "温馨治愈": f"这是一个温暖的故事。{image_description}，让每个人的心中都充满了感动。",
        "悬疑惊悚": f"夜幕降临，{image_description}。一个神秘的故事正在悄然展开。",
        "科幻未来": f"在未来的某一天，{image_description}。这个发现将改变人类的命运。",
        "童话寓言": f"从前有一个地方，{image_description}。这里住着一个关于勇气和智慧的故事。"
    }
    
    prompt = style_prompts.get(style, f"让我告诉你一个关于{image_description}的故事。")
    
    try:
        # 生成故事
        result = story_generator(
            prompt,
            max_length=length,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True
        )
        return result[0]['generated_text']
    except Exception as e:
        return f"生成故事时出错: {str(e)}"

# 主界面布局
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 上传图片")
    
    # 图片上传方式选择
    upload_option = st.radio(
        "选择图片来源",
        ["📁 本地上传", "🔗 图片URL"]
    )
    
    image = None
    image_source = None
    
    if upload_option == "📁 本地上传":
        uploaded_file = st.file_uploader(
            "选择一张图片",
            type=["jpg", "jpeg", "png", "webp"],
            help="支持JPG、PNG、WEBP格式"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_source = "upload"
    
    else:  # URL上传
        url = st.text_input("输入图片URL", placeholder="https://example.com/image.jpg")
        if url:
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                image_source = "url"
            except:
                st.error("无法加载图片，请检查URL是否有效")
    
    # 显示图片
    if image is not None:
        st.image(image, caption="你上传的图片", use_column_width=True)
        
        # 保存图片到session状态
        st.session_state['current_image'] = image

with col2:
    st.subheader("📖 生成的故事")
    
    # 加载模型
    if 'models_loaded' not in st.session_state:
        with st.status("正在加载AI模型...", expanded=True) as status:
            st.write("⏳ 加载图片分析模型...")
            st.write("⏳ 加载故事生成模型...")
            image_model, story_model = load_models()
            if image_model and story_model:
                st.session_state['image_model'] = image_model
                st.session_state['story_model'] = story_model
                st.session_state['models_loaded'] = True
                status.update(label="✅ 模型加载完成!", state="complete")
    
    # 生成故事按钮
    if image is not None and st.session_state.get('models_loaded', False):
        if st.button("✨ 生成故事", type="primary", use_container_width=True):
            with st.spinner("AI正在创作中..."):
                try:
                    # 步骤1: 分析图片
                    st.info("🔍 正在分析图片内容...")
                    image_result = st.session_state['image_model'](image)
                    image_description = image_result[0]['generated_text']
                    
                    # 显示图片描述
                    st.success(f"📝 图片描述: {image_description}")
                    
                    # 步骤2: 生成故事
                    st.info("📖 正在创作故事...")
                    story = generate_story(
                        image_description, 
                        story_style, 
                        story_length
                    )
                    
                    # 显示故事
                    st.markdown("### ✨ 你的专属故事")
                    
                    # 美化故事显示
                    story_container = st.container()
                    with story_container:
                        st.markdown(f"""
                        <div style="
                            background-color: #f0f2f6;
                            padding: 20px;
                            border-radius: 10px;
                            font-family: 'Georgia', serif;
                            line-height: 1.6;
                            font-size: 16px;
                        ">
                        {story}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 保存到历史
                    if 'story_history' not in st.session_state:
                        st.session_state['story_history'] = []
                    
                    st.session_state['story_history'].append({
                        'image': image,
                        'description': image_description,
                        'story': story,
                        'style': story_style
                    })
                    
                except Exception as e:
                    st.error(f"生成过程中出错: {str(e)}")
    
    elif image is None:
        st.info("👆 请先在左侧上传一张图片")
    
    elif not st.session_state.get('models_loaded', False):
        st.warning("⏳ 模型正在加载中，请稍候...")

# 历史故事展示
if st.session_state.get('story_history'):
    st.divider()
    st.subheader("📚 历史故事")
    
    for i, item in enumerate(reversed(st.session_state['story_history'][-3:])):
        with st.expander(f"故事 {i+1} - {item['style']}风格"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(item['image'], caption="原图", width=200)
            with col2:
                st.write(f"**图片描述:** {item['description']}")
                st.write(f"**故事片段:** {item['story'][:100]}...")

# 页脚
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "由 AI 驱动 | 上传图片，让想象力飞翔 ✨"
    "</div>", 
    unsafe_allow_html=True
)
