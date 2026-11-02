from src.llm_service import analyze_ticket


ticket = """
I transferred ₹5,000 yesterday.
The money was deducted from my account,
but the merchant has not received it.
Transaction ID TXN12345.
"""


result = analyze_ticket(ticket)

print("\n========== GEMINI RESULT ==========")
print("Category:", result.category)
print("Priority:", result.priority)
print("Sentiment:", result.sentiment)
print("Summary:", result.summary)
print("Team:", result.suggested_team)
print("Amount:", result.amount)
print("Transaction ID:", result.transaction_id)
print("Payment Method:", result.payment_method)
print("Entities:", result.important_entities)
print("===================================")