"""
search_interface/app.py — giao diện tìm kiếm (Streamlit).
Ô nhập query, hiển thị lưới kết quả kèm thumbnail.
Chạy: streamlit run ui/search_interface/app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    try:
        import streamlit as st  # type: ignore
    except ImportError:
        print("[!] streamlit chưa cài: pip install streamlit")
        return

    st.set_page_config(page_title="VTK Video Search", page_icon="🎬", layout="wide")
    st.title("🎬 VTK Video Search Interface")

    query_type = st.sidebar.selectbox("Loại truy vấn", ["KIS", "QA", "TRAKE"])
    query = st.text_area("Nhập mô tả sự kiện cần tìm:", height=100)
    question = ""
    if query_type == "QA":
        question = st.text_input("Câu hỏi:")
    top_k = st.sidebar.slider("Top-K kết quả", 10, 100, 50)

    if st.button("🔍 Tìm kiếm"):
        if not query.strip():
            st.warning("Vui lòng nhập truy vấn.")
            return

        st.info(f"Đang tìm kiếm... [{query_type}]")
        # Placeholder — kết nối với main pipeline khi ready
        try:
            from main import build_pipeline
            pipeline = build_pipeline()
            if query_type == "KIS":
                results = pipeline.search_kis(query, top_k=top_k)
            elif query_type == "QA":
                results = pipeline.search_qa(query, question, top_k=top_k)
            else:
                results = pipeline.search_trake(query, top_k=top_k)
        except Exception as e:
            st.error(f"Pipeline chưa sẵn sàng: {e}")
            results = []

        if not results:
            st.warning("Không tìm thấy kết quả.")
            return

        st.success(f"Tìm thấy {len(results)} kết quả.")
        cols = st.columns(5)
        for i, r in enumerate(results[:top_k]):
            with cols[i % 5]:
                # Thử load thumbnail
                keyframes_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "keyframes")
                img_path = os.path.join(keyframes_dir, r.get("video_id", ""), f"{int(r.get('frame_id', 0)):06d}.jpg")
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
                st.caption(f"{r.get('video_id')} @ {r.get('frame_id')} | {r.get('score', 0):.3f}")
                if query_type == "QA":
                    st.caption(f"📝 {r.get('answer', '')}")


if __name__ == "__main__":
    main()
