"""
generate_dataset.py
--------------------
Creates a synthetic social media dataset simulating tweets/posts/reviews
for brands like Zomato, Netflix, Amazon, Flipkart, etc.
Run this script once to generate data/social_media_data.csv
"""

import pandas as pd
import random
import os

random.seed(42)

# ── Sample posts by sentiment ─────────────────────────────────────────────────
POSITIVE_POSTS = [
    "Zomato delivery was super fast today! Loved the packaging 🎉",
    "Netflix's new series is absolutely mind-blowing. Binge-watched all night!",
    "Amazon Prime delivery in just 1 day! Great experience as always.",
    "Flipkart sale saved me so much money. Totally worth it!",
    "Swiggy support resolved my issue within minutes. Excellent service!",
    "The new iPhone camera quality is outstanding. Apple never disappoints.",
    "Had the best experience at this restaurant. Will definitely visit again!",
    "Customer support was very helpful and patient. 5 stars!",
    "App update is smooth and the UI looks amazing now.",
    "Product quality exceeded my expectations. Highly recommended!",
    "Ordered food at midnight, arrived hot and fresh. Zomato rocks!",
    "The movie recommendation on Netflix was spot on. Loved it!",
    "Packaging was eco-friendly and the product was intact. Great job!",
    "Super impressed with the quick refund. Will shop again.",
    "Love the new features in the latest app update. So intuitive!",
    "Delivery boy was very polite and on time. Appreciate it.",
    "Best pizza I've ever had. Ordered again the same evening.",
    "Discount coupon worked perfectly. Got 40% off. Amazing deal!",
    "The subscription plan is affordable and the content is top-notch.",
    "Really happy with my purchase. Exactly as described online.",
    "Great sound quality on these earphones. Best buy of the year!",
    "The customer care team was empathetic and resolved my issue fast.",
    "Seamless checkout experience. Will recommend to all my friends.",
    "Food was delicious and portions were generous. Worth every rupee.",
    "The live streaming feature is buttery smooth. 10/10 experience.",
]

NEGATIVE_POSTS = [
    "Zomato order was 2 hours late and food arrived cold. Disgusting!",
    "Netflix keeps buffering even on high-speed internet. Very frustrating.",
    "Amazon delivered a damaged product. Refund process is a nightmare.",
    "Flipkart customer support is completely useless. Never buying again.",
    "Swiggy cancelled my order without notice. Wasted my dinner plans.",
    "The phone battery drains within 3 hours. Total waste of money.",
    "Worst delivery experience ever. Package arrived torn and dirty.",
    "App crashes every time I try to checkout. Fix your bugs!",
    "Paid for premium but still seeing ads. This is unacceptable.",
    "Product is fake. The quality is terrible compared to what was shown.",
    "Received wrong item and now customer care is not picking up.",
    "Food was absolutely stale. Had food poisoning after eating this.",
    "Three-day delivery turned into three weeks. No updates at all.",
    "Prices are misleading. Hidden charges at checkout are ridiculous.",
    "The new update broke all my preferences. Terrible UX design.",
    "Driver couldn't find my location despite sharing live location.",
    "Ordered a birthday cake and it arrived smashed. Ruined the party.",
    "Subscription auto-renewed without any reminder. Scam behavior.",
    "Video quality drops every 5 minutes. Unwatchable experience.",
    "The refund was not credited even after 15 days. Chasing support.",
    "Worst movie recommendation algorithm. Shows irrelevant content.",
    "Chatbot is completely useless and never connects to a human agent.",
    "Packaging was terrible. Product was broken on arrival.",
    "Size chart is completely wrong. Returned the item immediately.",
    "This app is the biggest scam. Taking money without delivering service.",
]

NEUTRAL_POSTS = [
    "Placed my Zomato order. Waiting for delivery.",
    "Just renewed my Netflix subscription for another month.",
    "Received the Amazon package. Will open it later.",
    "Browsing through Flipkart for some deals.",
    "Swiggy app is showing estimated 35 minutes delivery time.",
    "The app asked me to rate my recent experience.",
    "Just downloaded the latest update on my phone.",
    "Checked my order status. It says out for delivery.",
    "Received an email about a new product launch.",
    "The restaurant menu had a lot of options to choose from.",
    "Got a notification about a flash sale starting tomorrow.",
    "Customer support put me on hold for a few minutes.",
    "The product page says it will be delivered by Friday.",
    "Logged into the app to check my order history.",
    "Saw an advertisement for a new show on Netflix.",
    "Reading product reviews before making a decision.",
    "Order tracking shows the package is in transit.",
    "The app is asking for a software update.",
    "Checked the price comparison across multiple platforms.",
    "Received a confirmation SMS for my booking.",
    "The delivery partner called to confirm the address.",
    "Downloaded the receipt for my recent purchase.",
    "Notification says my cashback will be credited in 2 days.",
    "Scrolling through my order history from last month.",
    "The help center page loaded with a list of FAQs.",
]

BRANDS = ["Zomato", "Netflix", "Amazon", "Flipkart", "Swiggy",
          "Apple", "Samsung", "Uber Eats", "Myntra", "BigBasket"]

PLATFORMS = ["Twitter", "Instagram", "Facebook", "Reddit", "Google Reviews"]


def generate_dataset(n: int = 1500, output_path: str = "data/social_media_data.csv"):
    rows = []
    pool = (
        [(p, "Positive") for p in POSITIVE_POSTS] +
        [(p, "Negative") for p in NEGATIVE_POSTS] +
        [(p, "Neutral")  for p in NEUTRAL_POSTS]
    )

    for i in range(n):
        base_text, label = random.choice(pool)
        # Slight variation to avoid exact duplicates
        variations = [
            base_text,
            base_text + " " + random.choice(["#disappointed", "#happy", "#feedback", ""]),
            base_text.replace("!", ".") if "!" in base_text else base_text,
        ]
        text = random.choice(variations).strip()

        rows.append({
            "id":         i + 1,
            "platform":   random.choice(PLATFORMS),
            "brand":      random.choice(BRANDS),
            "text":       text,
            "sentiment":  label,
            "likes":      random.randint(0, 5000),
            "retweets":   random.randint(0, 1000),
            "date":       f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset saved → {output_path}  ({len(df)} rows)")
    print(df["sentiment"].value_counts())
    return df


if __name__ == "__main__":
    generate_dataset()
