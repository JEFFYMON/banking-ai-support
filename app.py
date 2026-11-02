import pandas as pd
import gradio as gr

from src.preprocessing import clean_text
from src.entity_extraction import extract_entities
from src.classifier import train_classifier, predict
from src.llm_service import analyze_ticket, chat_with_customer


# ============================================================
# DATA
# ============================================================

DATA_PATH = "data/tickets.csv"

data = pd.read_csv(DATA_PATH)

ml_model = train_classifier(data)


# ============================================================
# TICKET ANALYZER
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

    cleaned_text = clean_text(
        ticket_text
    )

    # Traditional ML classification
    ml_category, ml_confidence = predict(
        ml_model,
        cleaned_text
    )

    # Rule-based entity extraction
    entities = extract_entities(
        ticket_text
    )

    try:

        # Gemini analysis
        gemini_result = analyze_ticket(
            ticket_text
        )

        category = gemini_result.category

        priority = gemini_result.priority

        sentiment = gemini_result.sentiment

        summary = gemini_result.summary

        team = gemini_result.suggested_team

        amount = (
            gemini_result.amount
            or
            (
                ", ".join(entities["amount"])
                if entities["amount"]
                else "Not detected"
            )
        )

        transaction_id = (
            gemini_result.transaction_id
            or
            (
                ", ".join(entities["transaction_id"])
                if entities["transaction_id"]
                else "Not detected"
            )
        )

        payment_method = (
            gemini_result.payment_method
            or
            (
                ", ".join(entities["payment_method"])
                if entities["payment_method"]
                else "Not detected"
            )
        )

    except Exception as error:

        print(
            f"Gemini ticket analysis failed: {error}"
        )

        # ML fallback
        category = ml_category

        priority = "MEDIUM"

        sentiment = "UNKNOWN"

        summary = (
            "Gemini analysis is temporarily unavailable. "
            "The ticket was classified using the traditional "
            "machine-learning model."
        )

        team = "General Support"

        amount = (
            ", ".join(entities["amount"])
            if entities["amount"]
            else "Not detected"
        )

        transaction_id = (
            ", ".join(entities["transaction_id"])
            if entities["transaction_id"]
            else "Not detected"
        )

        payment_method = (
            ", ".join(entities["payment_method"])
            if entities["payment_method"]
            else "Not detected"
        )

    confidence = (
        f"{ml_confidence * 100:.1f}%"
    )

    review_status = (
        "Human Review Required"
        if ml_confidence < 0.70
        else
        "AI Analysis Complete"
    )

    final_team_status = (
        f"{team} | {review_status}"
    )

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


# ============================================================
# CUSTOMER CHAT
# ============================================================

def customer_chat(
    message,
    history
):

    if not message or not message.strip():

        return history or []

    history = history or []

    try:

        reply = chat_with_customer(
            message,
            history
        )

    except Exception as error:

        print(
            f"Chatbot error: {error}"
        )

        reply = (
            "I'm temporarily unable to connect to the "
            "AI service. Please try again shortly. "
            "For urgent account-security issues, "
            "please use the bank's official support channel."
        )

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": reply
        }
    )

    return history


# ============================================================
# CLEAR CHAT
# ============================================================

def clear_chat():

    return [], ""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="BankAssist - AI Banking Support"
) as app:

    gr.Markdown(
        """
# 🏦 BankAssist

### AI-Powered Banking Customer Support

Ask questions about banking services, payments, cards,
transfers, refunds, ATMs, and online banking.
"""
    )

    with gr.Tabs():

        # ====================================================
        # CUSTOMER AI ASSISTANT
        # ====================================================

        with gr.Tab(
            "💬 Customer AI Assistant"
        ):

            gr.Markdown(
                """
## How can I help?

Ask a banking question in normal language.

BankAssist can explain common banking issues and
guide you to the appropriate support process.

### Security Notice

Do not enter:

- Passwords
- PINs
- OTPs
- CVVs
- Full card numbers
- Full account numbers
"""
            )

            chatbot = gr.Chatbot(
                label="BankAssist",
                height=500
            )

            with gr.Row():

                chat_input = gr.Textbox(
                    label="Your question",
                    placeholder=(
                        "Example: Why did my UPI payment fail?"
                    ),
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
                inputs=[
                    chat_input,
                    chatbot
                ],
                outputs=chatbot
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=chat_input
            )

            chat_input.submit(
                fn=customer_chat,
                inputs=[
                    chat_input,
                    chatbot
                ],
                outputs=chatbot
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=chat_input
            )

            clear_button.click(
                fn=clear_chat,
                inputs=None,
                outputs=[
                    chatbot,
                    chat_input
                ]
            )

            gr.Markdown(
                """
**Important:** BankAssist provides information and
guidance. It does not access accounts or perform
banking transactions.
"""
            )

        # ====================================================
        # SUPPORT TICKET ANALYZER
        # ====================================================

        with gr.Tab(
            "🎫 Support Ticket Analyzer"
        ):

            gr.Markdown(
                """
## Analyze a Customer Support Ticket

This module combines traditional machine learning
with Gemini to classify and summarize banking
support tickets.
"""
            )

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

            gr.Markdown(
                "## AI Analysis"
            )

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
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.launch(
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True
    )