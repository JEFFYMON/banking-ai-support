import os
import time
from typing import List, Optional, Literal

from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

# Fast, cost-conscious model for normal BankAssist support tasks.
OPENAI_MODEL = "gpt-6-luna"


# ============================================================
# STRUCTURED OUTPUT MODEL
# ============================================================

class TicketAnalysis(BaseModel):

    category: str = Field(
        description="The banking support category."
    )

    priority: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]

    sentiment: Literal[
        "NEUTRAL",
        "CONCERNED",
        "FRUSTRATED",
        "ANGRY"
    ]

    summary: str = Field(
        description="A short factual summary of the customer's issue."
    )

    suggested_team: str = Field(
        description="The support team that should handle the ticket."
    )

    amount: Optional[str] = None

    transaction_id: Optional[str] = None

    payment_method: Optional[str] = None

    important_entities: List[str] = Field(
        default_factory=list
    )


# ============================================================
# BANKING CHATBOT INSTRUCTIONS
# ============================================================

BANKING_CHATBOT_INSTRUCTIONS = """
You are BankAssist, a friendly AI customer-support assistant
for a banking application.

Your job is to answer common banking support questions clearly
and briefly.

You can explain:

- UPI and digital payments
- card usage and common card problems
- bank transfers
- refunds
- ATM and cash withdrawal issues
- internet/mobile banking
- account/profile questions
- statements
- general banking information

Safety and accuracy rules:

1. Give concise and easy-to-understand answers.

2. Never invent account balances, transaction status,
   fees, limits, policies, or personal information.

3. Never ask for or store passwords, PINs, CVVs, OTPs,
   full card numbers, or full account numbers.

4. Never authorize or execute transfers, refunds,
   account changes, or card blocks.

5. Never make a fraud determination.

6. If a customer reports an unrecognized transaction,
   tell them to use the bank's official fraud/dispute
   process and contact human support.

7. Do not claim to have accessed the customer's account.

8. Do not claim to have checked a transaction.

9. Do not claim to have performed a banking operation.

10. For account-specific issues, explain that an authenticated
    banking system or official support channel is required.

11. If the situation involves unauthorized activity,
    a lost/stolen card, or another urgent security issue,
    recommend contacting the bank's official support/security
    channel promptly.

12. Do not provide financial advice or recommend financial
    products.

13. Be polite and professional.

14. Keep normal answers short.
"""


# ============================================================
# OPENAI CLIENT
# ============================================================

def _get_client():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set."
        )

    return OpenAI(api_key=api_key)


# ============================================================
# OPENAI REQUEST
# ============================================================

def _generate_text(
    client,
    prompt: str,
    schema=None
):
    """
    Send one request to OpenAI.

    If schema is supplied, use structured output so the
    ticket analyzer receives predictable JSON data.
    """

    start_time = time.time()

    print(
        f"Calling OpenAI model: {OPENAI_MODEL}"
    )

    if schema is not None:

        response = client.responses.parse(
            model=OPENAI_MODEL,
            input=prompt,
            text_format=schema
        )

        elapsed = time.time() - start_time

        print(
            f"OpenAI structured response received in "
            f"{elapsed:.2f} seconds"
        )

        if not response:
            raise RuntimeError(
                "OpenAI returned no response."
            )

        if response.output_parsed is None:
            raise RuntimeError(
                "OpenAI did not return structured output."
            )

        return response.output_parsed

    else:

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt
        )

        elapsed = time.time() - start_time

        print(
            f"OpenAI response received in "
            f"{elapsed:.2f} seconds"
        )

        if not response:
            raise RuntimeError(
                "OpenAI returned no response."
            )

        if not response.output_text:
            raise RuntimeError(
                "OpenAI returned an empty response."
            )

        return response.output_text


# ============================================================
# TICKET ANALYSIS
# ============================================================

def analyze_ticket(
    ticket_text: str
) -> TicketAnalysis:

    if not ticket_text or not ticket_text.strip():

        raise ValueError(
            "Ticket text cannot be empty."
        )

    client = _get_client()

    prompt = f"""
Analyze this banking customer-support ticket.

Return only information supported by the ticket.

Rules:

- Identify the most appropriate banking support category.
- Determine urgency from the ticket.
- Identify customer sentiment.
- Give a short factual summary.
- Suggest the appropriate support team.
- Extract the amount only if explicitly present.
- Extract the transaction ID only if explicitly present.
- Extract the payment method only if explicitly present.
- If information is not present, return null.
- Do not invent account information.
- Do not make fraud decisions.
- Do not authorize refunds or transfers.

Customer ticket:

{ticket_text}
"""

    return _generate_text(
        client,
        prompt,
        TicketAnalysis
    )


# ============================================================
# CONVERT GRADIO HISTORY
# ============================================================

def _build_conversation(history):

    conversation = []

    if not history:
        return conversation

    # Keep only the latest few messages.
    # This prevents the prompt from becoming unnecessarily large.
    for item in history[-10:]:

        if isinstance(item, dict):

            role = item.get("role")
            content = item.get("content")

            if content is None:
                continue

            if isinstance(content, str):
                text = content.strip()
            else:
                text = str(content).strip()

            if not text:
                continue

            if role == "user":

                conversation.append(
                    f"Customer: {text}"
                )

            elif role == "assistant":

                conversation.append(
                    f"BankAssist: {text}"
                )

        elif isinstance(item, (list, tuple)):

            if len(item) >= 1 and item[0]:

                conversation.append(
                    f"Customer: {item[0]}"
                )

            if len(item) >= 2 and item[1]:

                conversation.append(
                    f"BankAssist: {item[1]}"
                )

    return conversation


# ============================================================
# CUSTOMER CHATBOT
# ============================================================

def chat_with_customer(
    message: str,
    history=None
) -> str:

    if not message or not message.strip():

        return "Please enter your question."

    message = message.strip()

    conversation = _build_conversation(
        history
    )

    conversation.append(
        f"Customer: {message}"
    )

    conversation_text = "\n".join(
        conversation
    )

    prompt = f"""
{BANKING_CHATBOT_INSTRUCTIONS}

Conversation:

{conversation_text}

Answer the customer's latest question.

Important:

- Answer directly.
- Keep the response concise.
- Use previous conversation when relevant.
- Never invent account information.
- Never claim to have checked an account.
- Never claim to have checked a transaction.
- Never claim to have performed a banking operation.
- Never request passwords, PINs, OTPs, CVVs,
  full card numbers, or full account numbers.
- For account-specific questions, direct the customer
  to the authenticated banking system or official support.

Customer's latest message:

{message}
"""

    client = _get_client()

    return _generate_text(
        client,
        prompt
    ).strip()