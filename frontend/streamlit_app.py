from os import getenv
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = httpx.Timeout(30.0)


def api_request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    with httpx.Client(base_url=API_BASE_URL, timeout=TIMEOUT) as client:
        return client.request(method, path, **kwargs)


def load_documents() -> list[dict[str, Any]]:
    response = api_request("GET", "/documents")
    response.raise_for_status()
    return response.json()


def upload_document(uploaded_file: Any) -> dict[str, Any]:
    response = api_request(
        "POST",
        "/documents",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        },
    )
    response.raise_for_status()
    return response.json()


def delete_document(filename: str) -> None:
    response = api_request("DELETE", f"/documents/{filename}")
    response.raise_for_status()


def send_chat(question: str, conversation_id: str | None) -> dict[str, Any]:
    response = api_request(
        "POST",
        "/chat",
        headers={"X-User-ID": "streamlit-user"},
        json={"question": question, "conversation_id": conversation_id},
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Atlas Knowledge", page_icon="A", layout="wide")
    st.title("Atlas Knowledge")
    st.caption("A focused workspace for answers grounded in your documents.")

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Workspace")
        st.caption(API_BASE_URL)
        try:
            readiness = api_request("GET", "/health/ready")
            if readiness.is_success and readiness.json().get("status") == "ok":
                st.success("System ready")
            elif readiness.status_code == 503:
                st.warning("System degraded: database unavailable")
            else:
                st.warning("System degraded")
        except httpx.HTTPError:
            st.error("API unavailable")

        st.subheader("Upload documents")
        uploaded_files = st.file_uploader(
            "Choose TXT, Markdown, PDF, DOCX, or XLSX files",
            type=["txt", "md", "pdf", "docx", "xlsx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files and st.button("Index selected files", type="primary"):
            for uploaded_file in uploaded_files:
                try:
                    result = upload_document(uploaded_file)
                    st.success(f"Indexed {result['filename']} ({result['chunks_indexed']} chunks)")
                except httpx.HTTPError as error:
                    st.error(f"Could not index {uploaded_file.name}: {error}")
            st.rerun()

        st.subheader("Document list")
        try:
            documents = load_documents()
            if not documents:
                st.caption("No documents indexed yet.")
            for document in documents:
                left, right = st.columns([4, 1])
                left.write(f"{document['filename']}  ")
                left.caption(f"{document['chunks']} chunks")
                if right.button("×", key=f"delete-{document['filename']}"):
                    try:
                        delete_document(document["filename"])
                        st.rerun()
                    except httpx.HTTPError:
                        st.error("Could not delete document.")
        except httpx.HTTPError:
            st.caption("Document list unavailable.")

        if st.button("New conversation"):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption("Developed by Jitendra Dewangan")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        page = f", page {source['page']}" if source.get("page") else ""
                        st.markdown(f"**{source['source']}{page}** · score {source.get('score', 'n/a')}")
                        st.caption(source["excerpt"])

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"), st.spinner("Searching your documents..."):
                try:
                    result = send_chat(question, st.session_state.conversation_id)
                    st.session_state.conversation_id = result["conversation_id"]
                    st.write(result["answer"])
                    if result["sources"]:
                        with st.expander("Sources"):
                            for source in result["sources"]:
                                page = f", page {source['page']}" if source.get("page") else ""
                                st.markdown(f"**{source['source']}{page}** · score {source.get('score', 'n/a')}")
                                st.caption(source["excerpt"])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                    )
                except httpx.HTTPError:
                    st.error("The assistant could not reach the API. Check that FastAPI is running.")


if __name__ == "__main__":
    main()
