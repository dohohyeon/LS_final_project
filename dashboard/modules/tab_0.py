# modules/tab_0.py
import base64
import html
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from shared import *

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "banner_image"


def _encode_image_to_data_url(image_path: Path) -> str:
    """Return the image as a data URL so the front-end slider can use it without reloading files."""
    with Image.open(image_path) as img:
        buffer = BytesIO()
        export_format = img.format or image_path.suffix.lstrip(".") or "PNG"
        img.save(buffer, format=export_format)

    mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def create_auto_banner():
    """
    Render a banner carousel that rotates every three seconds using a lightweight HTML component.
    """
    banner_specs = [
        {
            "filename": "banner1.png",
            "title": "LS ELECTRIC SMART FACTORY",
            "subtitle": "스마트 팩토리의 미래 에너지를 만듭니다",
        },
        {
            "filename": "banner2.png",
            "title": "자동화 생산 시스템",
            "subtitle": "AI 기반 에너지 효율 모니터링",
        },
        {
            "filename": "banner3.png",
            "title": "스마트 에너지 관리",
            "subtitle": "효율적인 에너지 솔루션 구축",
        },
    ]

    banners = []

    for spec in banner_specs:
        image_path = ASSETS_DIR / spec["filename"]
        if not image_path.exists():
            st.warning(f"⚠️ 배너 이미지를 찾을 수 없습니다: {image_path}")
            st.info("dashboard/assets/banner_image/ 폴더에 이미지를 배치해 주세요.")
            return

        banners.append(
            {
                "title": spec["title"],
                "subtitle": spec["subtitle"],
                "image": _encode_image_to_data_url(image_path),
            }
        )

    slides_html = "\n".join(
        f"""
        <div class="banner-slide{' active' if index == 0 else ''}">
            <img src="{banner['image']}" alt="{html.escape(banner['title'])}" class="banner-image" />
            <div class="banner-overlay">
                <h1>{html.escape(banner['title'])}</h1>
                <p>{html.escape(banner['subtitle'])}</p>
            </div>
        </div>
        """
        for index, banner in enumerate(banners)
    )

    indicators_html = "\n".join(
        f"<span class='dot{' active' if index == 0 else ''}'></span>"
        for index in range(len(banners))
    )

    html_template = f"""
    <div class="auto-banner-root">
        <style>
            .auto-banner-root {{
                font-family: 'Pretendard', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                margin-top: 24px;
            }}

            .banner-wrapper {{
                position: relative;
                width: 100%;
                height: 360px;
                border-radius: 24px;
                overflow: hidden;
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.25);
                transition: transform 0.6s ease, box-shadow 0.6s ease;
            }}

            .banner-slide {{
                position: absolute;
                inset: 0;
                opacity: 0;
                transition: opacity 0.8s ease;
            }}

            .banner-slide.active {{
                opacity: 1;
            }}

            .banner-image {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
                transition: transform 0.6s ease;
            }}

            .banner-overlay {{
                position: absolute;
                left: 48px;
                bottom: 48px;
                color: #fff;
                max-width: 60%;
                text-shadow: 0 6px 18px rgba(0, 0, 0, 0.65);
            }}

            .banner-overlay h1 {{
                margin: 0 0 16px;
                font-size: 48px;
                font-weight: 700;
                line-height: 1.1;
            }}

            .banner-overlay p {{
                margin: 0;
                font-size: 22px;
                line-height: 1.5;
            }}

            .banner-indicators {{
                position: absolute;
                left: 48px;
                bottom: 24px;
                display: flex;
                gap: 10px;
            }}

            .banner-indicators .dot {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: rgba(255, 255, 255, 0.45);
                transition: background-color 0.3s ease;
            }}

            .banner-indicators .dot.active {{
                background-color: rgba(255, 255, 255, 0.95);
            }}

            .auto-banner-root:hover .banner-wrapper {{
                transform: translateY(-8px);
                box-shadow: 0 24px 50px rgba(0, 0, 0, 0.3);
            }}

            .auto-banner-root:hover .banner-image {{
                transform: scale(1.03);
            }}

            @media (max-width: 768px) {{
                .banner-wrapper {{
                    height: 280px;
                }}

                .banner-overlay {{
                    left: 24px;
                    bottom: 24px;
                    max-width: 80%;
                }}

                .banner-overlay h1 {{
                    font-size: 28px;
                }}

                .banner-overlay p {{
                    font-size: 18px;
                }}

                .banner-indicators {{
                    left: 24px;
                    bottom: 16px;
                }}
            }}
        </style>

        <div class="banner-wrapper">
            {slides_html}
            <div class="banner-indicators">
                {indicators_html}
            </div>
        </div>

        <script>
            const root = document.currentScript.parentElement;
            const slides = root.querySelectorAll('.banner-slide');
            const dots = root.querySelectorAll('.dot');

            if (slides.length) {{
                let activeIndex = 0;

                const activateSlide = (nextIndex) => {{
                    slides[activeIndex].classList.remove('active');
                    dots[activeIndex].classList.remove('active');

                    activeIndex = nextIndex;

                    slides[activeIndex].classList.add('active');
                    dots[activeIndex].classList.add('active');
                }};

                setInterval(() => {{
                    const nextIndex = (activeIndex + 1) % slides.length;
                    activateSlide(nextIndex);
                }}, 3000);
            }}
        </script>
    </div>
    """

    components.html(html_template, height=380)


def show_tab_home(train):
    """
    HOME 탭 - 배너 + 예측 결과
    """
    create_auto_banner()

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; padding: 40px 0;'>
            <h2 style='color: #020202; margin-bottom: 20px;'>
                LS 스마트 공장 전력 모니터링 서비스에 오신 것을 환영합니다
            </h2>
            <p style='font-size: 18px; color: #666; line-height: 1.8;'>
                실시간 설비 전력 사용량을 모니터링하고 AI 기반 수요 예측으로<br>
                효율적인 관리를 도와드립니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🔍 **실시간 모니터링**\n\n설비별 소비 전력을 한눈에 확인하세요.")

    with col2:
        st.success("📊 **추세 분석**\n\n기간별 에너지 사용 패턴을 분석해 드립니다.")

    with col3:
        st.warning("🤖 **AI 챗봇**\n\n추후 공개 예정입니다.")
