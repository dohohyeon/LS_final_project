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
            html, body {{
                background: transparent;
            }}

            .auto-banner-root {{
                font-family: 'Pretendard', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                margin: 0 !important;
                padding: 0 20px !important;
                width: 100vw !important;
                position: relative;
                left: 50%;
                right: 50%;
                margin-left: -50vw !important;
                margin-right: -50vw !important;
                background: transparent !important;
            }}

            .banner-wrapper {{
                position: relative;
                width: calc(100% - 40px);
                max-width: 1400px;
                height: 480px;
                margin: 0 auto;
                border-radius: 24px;
                overflow: hidden;
                /* 입체적인 그림자 효과 */
                box-shadow: 
                    0 10px 30px rgba(0, 0, 0, 0.15),
                    0 20px 60px rgba(0, 0, 0, 0.12),
                    0 30px 80px rgba(4, 20, 46, 0.08);
                /* 부드러운 애니메이션 */
                transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                /* 3D 효과를 위한 transform */
                transform: perspective(1000px) translateZ(0);
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
                transition: transform 0.8s cubic-bezier(0.4, 0, 0.2, 1), 
                            filter 0.8s ease;
                /* 이미지 자체에 미묘한 그림자 */
                filter: brightness(0.95) contrast(1.02);
            }}

            .banner-overlay {{
                position: absolute;
                left: 10%;
                bottom: 80px;
                color: #fff;
                max-width: 60%;
                /* 입체적인 텍스트 그림자 */
                text-shadow: 
                    0 2px 4px rgba(0, 0, 0, 0.3),
                    0 4px 8px rgba(0, 0, 0, 0.25),
                    0 8px 16px rgba(0, 0, 0, 0.2);
                transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                /* 3D 효과 */
                transform: translateZ(20px);
            }}

            .banner-overlay h1 {{
                margin: 0 0 16px;
                font-size: 52px;
                font-weight: 700;
                line-height: 1.1;
                /* 텍스트 외곽선 효과 */
                -webkit-text-stroke: 0.5px rgba(0, 0, 0, 0.2);
            }}

            .banner-overlay p {{
                margin: 0;
                font-size: 24px;
                line-height: 1.5;
                font-weight: 500;
            }}

            .banner-indicators {{
                position: absolute;
                left: 50%;
                transform: translateX(-50%);
                bottom: 32px;
                display: flex;
                gap: 12px;
                z-index: 10;
                /* 인디케이터 컨테이너 그림자 */
                filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.25));
            }}

            .banner-indicators .dot {{
                width: 14px;
                height: 14px;
                border-radius: 50%;
                background-color: rgba(255, 255, 255, 0.45);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
                /* 개별 dot에 그림자 */
                box-shadow: 
                    0 2px 6px rgba(0, 0, 0, 0.2),
                    inset 0 1px 2px rgba(255, 255, 255, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}

            .banner-indicators .dot.active {{
                background-color: rgba(255, 255, 255, 0.95);
                box-shadow: 
                    0 3px 10px rgba(255, 255, 255, 0.4),
                    0 2px 6px rgba(0, 0, 0, 0.3),
                    inset 0 1px 3px rgba(255, 255, 255, 0.5);
                transform: scale(1.15);
            }}

            /* 호버 효과 - 전체 배너 */
            .banner-wrapper:hover {{
                /* 더 입체적으로 떠오르는 효과 */
                transform: perspective(1000px) translateZ(0) translateY(-12px);
                box-shadow: 
                    0 15px 40px rgba(0, 0, 0, 0.2),
                    0 25px 70px rgba(0, 0, 0, 0.15),
                    0 35px 90px rgba(4, 20, 46, 0.12);
            }}

            /* 호버 시 이미지 확대 */
            .banner-wrapper:hover .banner-slide.active .banner-image {{
                transform: scale(1.06);
                filter: brightness(1) contrast(1.05);
            }}

            /* 호버 시 텍스트 움직임 */
            .banner-wrapper:hover .banner-slide.active .banner-overlay {{
                transform: translateZ(30px) translateY(-12px);
                text-shadow: 
                    0 3px 6px rgba(0, 0, 0, 0.35),
                    0 6px 12px rgba(0, 0, 0, 0.3),
                    0 12px 24px rgba(0, 0, 0, 0.25);
            }}

            /* dot 호버 효과 */
            .banner-indicators .dot:hover {{
                transform: scale(1.3);
                background-color: rgba(255, 255, 255, 0.8);
                box-shadow: 
                    0 4px 12px rgba(255, 255, 255, 0.5),
                    0 2px 6px rgba(0, 0, 0, 0.3);
            }}

            /* 반응형 디자인 */
            @media (max-width: 768px) {{
                .banner-wrapper {{
                    height: 320px;
                    border-radius: 16px;
                }}

                .banner-overlay {{
                    left: 5%;
                    bottom: 40px;
                    max-width: 80%;
                }}

                .banner-overlay h1 {{
                    font-size: 32px;
                }}

                .banner-overlay p {{
                    font-size: 18px;
                }}

                .banner-indicators {{
                    bottom: 20px;
                }}

                .banner-wrapper:hover {{
                    transform: perspective(1000px) translateZ(0) translateY(-6px);
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
            (function() {{
                const frame = window.frameElement;
                if (frame) {{
                    frame.style.background = 'transparent';
                    frame.style.boxShadow = 'none';
                    frame.style.border = 'none';
                    const frameParent = frame.parentElement;
                    if (frameParent) {{
                        frameParent.style.background = 'transparent';
                        frameParent.style.boxShadow = 'none';
                        frameParent.style.padding = '0';
                    }}
                    const frameGrandParent = frameParent && frameParent.parentElement;
                    if (frameGrandParent) {{
                        frameGrandParent.style.background = 'transparent';
                        frameGrandParent.style.boxShadow = 'none';
                    }}
                }}

                const root = document.currentScript.parentElement;
                if (!root) {{
                    return;
                }}

                const slides = root.querySelectorAll('.banner-slide');
                const dots = root.querySelectorAll('.dot');

                if (!slides.length) {{
                    return;
                }}

                let activeIndex = 0;

                const activateSlide = (nextIndex) => {{
                    slides[activeIndex].classList.remove('active');
                    dots[activeIndex].classList.remove('active');

                    activeIndex = nextIndex;

                    slides[activeIndex].classList.add('active');
                    dots[activeIndex].classList.add('active');
                }};

                // 자동 슬라이드
                setInterval(() => {{
                    const nextIndex = (activeIndex + 1) % slides.length;
                    activateSlide(nextIndex);
                }}, 3000);

                // 인디케이터 클릭 이벤트
                dots.forEach((dot, index) => {{
                    dot.addEventListener('click', () => {{
                        activateSlide(index);
                    }});
                }});
            }})();
        </script>
    </div>
    """

    components.html(html_template, height=500)


def show_tab_home(train):
    """
    HOME 탭 - 배너 + 예측 결과
    """
    # 배너를 화면 전체 너비로 표시
    create_auto_banner()

    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
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
