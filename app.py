
import os
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from src.preprocessing import clean_text
from src.entity_extraction import extract_entities
from src.classifier import train_classifier, predict
from src.llm_service import analyze_ticket, chat_with_customer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tickets.csv"

# Load local environment variables from .env
load_dotenv(BASE_DIR / ".env")


# ============================================================
# LOAD DATASET AND TRAIN ML MODEL
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Please check that data/tickets.csv exists."
    )

data = pd.read_csv(DATA_PATH)

if data.empty:
    raise ValueError("The tickets.csv dataset is empty.")

ml_model = train_classifier(data)


# ============================================================
# SUPPORT TICKET ANALYZER
# ============================================================

def analyze_customer_ticket(ticket_text):

    if not ticket_text or not ticket_text.strip():
        return (
            "Please enter a customer support ticket.",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        )

    try:
        cleaned_text = clean_text(ticket_text)

        ml_category, ml_confidence = predict(
            ml_model,
            cleaned_text
        )

        entities = extract_entities(ticket_text)

        # AI-powered ticket analysis
        try:
            ai_result = analyze_ticket(ticket_text)

            category = ai_result.category
            priority = ai_result.priority
            sentiment = ai_result.sentiment
            summary = ai_result.summary
            team = ai_result.suggested_team

            amount = ai_result.amount or (
                ", ".join(entities.get("amount", []))
                if entities.get("amount")
                else "Not detected"
            )

            transaction_id = ai_result.transaction_id or (
                ", ".join(entities.get("transaction_id", []))
                if entities.get("transaction_id")
                else "Not detected"
            )

            payment_method = ai_result.payment_method or (
                ", ".join(entities.get("payment_method", []))
                if entities.get("payment_method")
                else "Not detected"
            )

        except Exception as error:
            print(f"AI ticket analysis failed: {error}")

            # Traditional ML fallback
            category = ml_category
            priority = "MEDIUM"
            sentiment = "UNKNOWN"

            summary = (
                "AI analysis is temporarily unavailable. "
                "The ticket was classified using the "
                "traditional machine-learning model."
            )

            team = "General Support"

            amount = (
                ", ".join(entities.get("amount", []))
                if entities.get("amount")
                else "Not detected"
            )

            transaction_id = (
                ", ".join(entities.get("transaction_id", []))
                if entities.get("transaction_id")
                else "Not detected"
            )

            payment_method = (
                ", ".join(entities.get("payment_method", []))
                if entities.get("payment_method")
                else "Not detected"
            )

        confidence = f"{ml_confidence * 100:.1f}%"

        review_status = (
            "Human Review Required"
            if ml_confidence < 0.70
            else "AI Analysis Complete"
        )

        final_team_status = f"{team} | {review_status}"

        return (
            category,
            priority,
            sentiment,
            confidence,
            amount,
            transaction_id,
            payment_method,
            summary,
            final_team_status
        )

    except Exception as error:
        print(f"Ticket processing error: {error}")

        return (
            "Analysis Error",
            "UNKNOWN",
            "UNKNOWN",
            "N/A",
            "Not detected",
            "Not detected",
            "Not detected",
            f"Unable to analyze ticket: {error}",
            "Please try again or contact support."
        )


# ============================================================
# CUSTOMER AI CHATBOT
# ============================================================

def customer_chat(message, history):

    if not message or not message.strip():
        return history or []

    history = history or []

    try:
        reply = chat_with_customer(message, history)

    except Exception as error:
        print(f"Chatbot error: {error}")

        reply = (
            "I'm temporarily unable to connect to the AI service. "
            "Please try again shortly. For urgent account-security "
            "issues, please use your bank's official support channel."
        )

    updated_history = list(history)

    updated_history.append({
        "role": "user",
        "content": message
    })

    updated_history.append({
        "role": "assistant",
        "content": reply
    })

    return updated_history


# ============================================================
# CLEAR CHAT
# ============================================================

def clear_chat():
    return [], ""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="BankAssist - AI Banking Customer Support",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown("""
# 🏦 BankAssist

### AI-Powered Banking Customer Support

Your intelligent assistant for common banking questions,
payments, cards, transfers, refunds, ATMs, and online banking.
""")

    with gr.Tabs():

        # ----------------------------------------------------
        # CUSTOMER AI ASSISTANT
        # ----------------------------------------------------

        with gr.Tab("💬 Customer AI Assistant"):

            gr.Markdown("""
## How can I help you?

Ask a banking question in normal language.

BankAssist can explain common banking issues and guide you
towards the appropriate support process.

### 🔒 Security Notice

Never enter sensitive information such as:

- Passwords
- PINs
- OTPs
- CVVs
- Full debit or credit card numbers
- Full bank account numbers
""")

            chatbot = gr.Chatbot(
                label="BankAssist Assistant",
                height=500
            )

            with gr.Row():

                chat_input = gr.Textbox(
                    label="Your question",
                    placeholder="Example: Why did my UPI payment fail?",
                    lines=2,
                    scale=8
                )

                send_button = gr.Button(
                    "Send",
                    variant="primary",
                    scale=2
                )

            clear_button = gr.Button(
                "Clear Conversation"
            )

            send_button.click(
                fn=customer_chat,
                inputs=[chat_input, chatbot],
                outputs=chatbot
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=chat_input
            )

            chat_input.submit(
                fn=customer_chat,
                inputs=[chat_input, chatbot],
                outputs=chatbot
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=chat_input
            )

            clear_button.click(
                fn=clear_chat,
                inputs=None,
                outputs=[chatbot, chat_input]
            )

            gr.Markdown("""
**Important:** BankAssist provides general information and
guidance. It does not access bank accounts or perform
banking transactions.
""")

        # ----------------------------------------------------
        # SUPPORT TICKET ANALYZER
        # ----------------------------------------------------

        with gr.Tab("🎫 Support Ticket Analyzer"):

            gr.Markdown("""
## Analyze a Customer Support Ticket

This module combines traditional machine learning
with AI-powered analysis to classify and summarize
banking customer support tickets.
""")

            ticket_input = gr.Textbox(
                label="Customer Support Ticket",
                placeholder=(
                    "Example: I transferred ₹5,000 yesterday, "
                    "but the merchant has not received it. "
                    "Transaction ID TXN12345."
                ),
                lines=7
            )

            analyze_button = gr.Button(
                "🔍 Analyze Ticket",
                variant="primary"
            )

            gr.Markdown("## AI Analysis Results")

            with gr.Row():

                category_output = gr.Textbox(
                    label="Category"
                )

                priority_output = gr.Textbox(
                    label="Priority"
                )

                sentiment_output = gr.Textbox(
                    label="Sentiment"
                )

                confidence_output = gr.Textbox(
                    label="ML Confidence"
                )

            with gr.Row():

                amount_output = gr.Textbox(
                    label="Amount"
                )

                transaction_output = gr.Textbox(
                    label="Transaction ID"
                )

                payment_output = gr.Textbox(
                    label="Payment Method"
                )

            summary_output = gr.Textbox(
                label="AI Summary",
                lines=4
            )

            team_output = gr.Textbox(
                label="Suggested Team / Review Status",
                lines=2
            )

            analyze_button.click(
                fn=analyze_customer_ticket,
                inputs=ticket_input,
                outputs=[
                    category_output,
                    priority_output,
                    sentiment_output,
                    confidence_output,
                    amount_output,
                    transaction_output,
                    payment_output,
                    summary_output,
                    team_output
                ]
            )


# ============================================================
# START APPLICATION - LOCAL AND RENDER
# ============================================================

if __name__ == "__main__":

    # Render automatically provides the PORT variable.
    # When running locally, use port 7862.
    port = int(os.environ.get("PORT", "7862"))

    print("=" * 55)
    print("Starting BankAssist - AI Banking Customer Support")
    print(f"Server port: {port}")
    print("=" * 55)

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
