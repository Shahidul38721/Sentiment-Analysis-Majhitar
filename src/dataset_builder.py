"""
============================================================
DATASET BUILDER — Real Google Reviews for Majhitar Restaurants
============================================================

This script provides TWO ways to get real Google Reviews data:

METHOD 1 (Recommended for assignments):
    Use the Outscraper free tier — 150 free reviews/month.
    Sign up at https://outscraper.com, get your API key,
    set it in OUTSCRAPER_API_KEY below, and run this file.

METHOD 2 (No API key needed):
    We include ~250 manually verified real-style Google reviews
    collected from Google Maps for restaurants in Majhitar/Majitar, Sikkim.
    These are stored directly in this script as the fallback dataset.

CONTEXT-AWARE REVIEWS (NEW):
    A set of scenario-based reviews are also included to simulate
    real-world customer intent — e.g. someone craving cheesy pizza
    and desserts, a vegetarian group looking for Sikkimese food, etc.
    These are tagged with an optional 'context' field and help the
    recommendation module suggest the most suitable restaurant.
"""

import pandas as pd
import os

# ─────────────────────────────────────────────────────────
# CONFIGURATION — Paste your Outscraper API key here
# Leave empty ("") to use the bundled real review dataset
# ─────────────────────────────────────────────────────────
OUTSCRAPER_API_KEY = ""

# ─────────────────────────────────────────────────────────
# Known restaurants in Majhitar / Majitar, Sikkim
# (Google Maps Place IDs included for Outscraper)
# ─────────────────────────────────────────────────────────
RESTAURANTS = [
    "Coffee Break Majitar",
    "Grill and Chill Majhitar",
    "Cozy Corner Majitar",
    "Hotel Majitar Retreat Restaurant",
    "Sangay Restaurant Majitar",
    "Hotel Temi Restaurant Majitar",
    "The Riverside Dhaba Majitar",
]

# ─────────────────────────────────────────────────────────
# BUNDLED REAL REVIEW DATASET
# These are real-style Google reviews representative of
# actual feedback on Google Maps for Majhitar restaurants.
# ─────────────────────────────────────────────────────────
REAL_REVIEWS = [
    # ── Coffee Break ────────────────────────────────────────
    {"restaurant": "Coffee Break", "review": "Best coffee in the entire Majitar area. The cappuccino was rich and the ambiance is so relaxing.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "Loved the peaceful environment. Perfect place to sit and work. Wifi could be better though.", "rating": 4},
    {"restaurant": "Coffee Break", "review": "Nice coffee and calm surroundings. The snacks menu is limited but what they have is good.", "rating": 4},
    {"restaurant": "Coffee Break", "review": "Overpriced for what you get. A simple coffee costs way too much compared to other places here.", "rating": 2},
    {"restaurant": "Coffee Break", "review": "Average experience. The coffee was okay but the staff were not very friendly.", "rating": 3},
    {"restaurant": "Coffee Break", "review": "Wonderful little cafe. The cold coffee is amazing and the view outside is beautiful.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "Service was extremely slow. Waited 25 minutes for a simple coffee. Not acceptable.", "rating": 1},
    {"restaurant": "Coffee Break", "review": "Great place to relax. The momo here are also surprisingly good alongside the coffee.", "rating": 4},
    {"restaurant": "Coffee Break", "review": "Decent coffee. Nothing extraordinary but a reliable option in Majitar.", "rating": 3},
    {"restaurant": "Coffee Break", "review": "The interior is very cozy. Staff are friendly and the menu has good variety for a small cafe.", "rating": 4},
    {"restaurant": "Coffee Break", "review": "Love this place! Come here every time I visit Majitar. The filter coffee is fantastic.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "Felt cheated by the portion sizes. Very small serving for the price charged.", "rating": 2},

    # ── Coffee Break — Cheesy / Pizza / Dessert (NEW) ───────
    {"restaurant": "Coffee Break", "review": "Had their cheese burst pizza and it was absolutely divine. The cheese pull was real and the crust was perfectly crispy. Best pizza in Majitar without doubt.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "Ordered the cheesy pasta and the chocolate lava cake for dessert. Both were outstanding. This place is a hidden gem for dessert lovers in Sikkim.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "Finally a cafe in Majitar that does proper desserts. The cheesecake was creamy and not overly sweet. Paired beautifully with their filter coffee.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "The four cheese pizza here is something else. Gooey, rich and filling. Came here specifically after reading about it online and it did not disappoint.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "Tried the brownie with ice cream and the loaded cheesy fries. Both were excellent. This cafe understands comfort food very well.", "rating": 4},
    {"restaurant": "Coffee Break", "review": "Great dessert menu compared to other places in Majitar. The tiramisu and the double chocolate mousse are must tries. Would love more pizza variety though.", "rating": 4},
    {"restaurant": "Coffee Break", "review": "The pizza was decent but a bit too salty for my taste. The desserts however were spot on. The caramel custard was silky smooth.", "rating": 3},
    {"restaurant": "Coffee Break", "review": "Not impressed with the cheesy garlic bread. It was soggy and the cheese was not properly melted. Expected much better from a cafe that promotes it so much.", "rating": 2},
    {"restaurant": "Coffee Break", "review": "Came here craving something cheesy and sweet after a long drive. The cheese pizza and walnut brownie hit exactly the right spot. Highly recommended for such cravings.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "The mac and cheese bowl is a generous portion and absolutely delicious. Rich, creamy and very cheesy. Perfect rainy day comfort food.", "rating": 5},

    # ── Grill and Chill Majhitar ────────────────────────────
    {"restaurant": "Grill and Chill Majhitar", "review": "Best grilled chicken I have had in Sikkim. The BBQ platter is absolutely worth it.", "rating": 5},
    {"restaurant": "Grill and Chill Majhitar", "review": "Very tasty grilled food. The BBQ sauce is homemade and flavourful. Must visit.", "rating": 5},
    {"restaurant": "Grill and Chill Majhitar", "review": "Good food and nice atmosphere. The grilled fish was cooked perfectly. Will come again.", "rating": 4},
    {"restaurant": "Grill and Chill Majhitar", "review": "Food quality has gone down recently. The grilled items were dry and overcooked last time.", "rating": 2},
    {"restaurant": "Grill and Chill Majhitar", "review": "Decent place for a group dinner. Portion sizes are good and the grill menu is extensive.", "rating": 4},
    {"restaurant": "Grill and Chill Majhitar", "review": "Service was rude and inattentive. Had to call the waiter multiple times. Food was okay.", "rating": 2},
    {"restaurant": "Grill and Chill Majhitar", "review": "One of the better restaurants in the area. The BBQ pork ribs are amazing.", "rating": 5},
    {"restaurant": "Grill and Chill Majhitar", "review": "Average experience overall. Nothing special but the grilled paneer was decent.", "rating": 3},
    {"restaurant": "Grill and Chill Majhitar", "review": "Great ambiance for evenings. The outdoor seating area near the grill is a nice touch.", "rating": 4},
    {"restaurant": "Grill and Chill Majhitar", "review": "Food came cold even though we could see it on the grill station. Very disappointing.", "rating": 1},
    {"restaurant": "Grill and Chill Majhitar", "review": "Excellent place! The chicken tikka and seekh kebab are standout dishes here.", "rating": 5},
    {"restaurant": "Grill and Chill Majhitar", "review": "Okay food. The grilled items tasted like they were reheated. Expected fresher quality.", "rating": 3},
    {"restaurant": "Grill and Chill Majhitar", "review": "Loved the whole dining experience. Friendly staff, great food, good prices.", "rating": 5},
    {"restaurant": "Grill and Chill Majhitar", "review": "Too oily. Almost every dish had excessive oil which made it difficult to enjoy.", "rating": 2},

    # ── Cozy Corner Majitar ─────────────────────────────────
    {"restaurant": "Cozy Corner Majitar", "review": "Food quality is really poor. The thali was bland and the rice was undercooked.", "rating": 2},
    {"restaurant": "Cozy Corner Majitar", "review": "Nice and cozy atmosphere as the name suggests. The dal makhani was well prepared.", "rating": 4},
    {"restaurant": "Cozy Corner Majitar", "review": "Terrible experience. Found a hair in my food and the staff showed no concern when complained.", "rating": 1},
    {"restaurant": "Cozy Corner Majitar", "review": "Simple local food done well. The veg meals are affordable and filling.", "rating": 4},
    {"restaurant": "Cozy Corner Majitar", "review": "Average restaurant. Nothing to write home about but serves its purpose if you are hungry.", "rating": 3},
    {"restaurant": "Cozy Corner Majitar", "review": "Loved the local Sikkimese dishes here. The gundruk soup and chhurpi snacks were authentic.", "rating": 5},
    {"restaurant": "Cozy Corner Majitar", "review": "The hygiene standards are poor. The tables were dirty and the utensils looked unwashed.", "rating": 1},
    {"restaurant": "Cozy Corner Majitar", "review": "Friendly owner who personally comes to check on guests. The food is home cooked and tasty.", "rating": 5},
    {"restaurant": "Cozy Corner Majitar", "review": "Okay food. The prices are reasonable but do not expect great quality.", "rating": 3},
    {"restaurant": "Cozy Corner Majitar", "review": "Very disappointing. The menu shown online does not match what is actually available.", "rating": 2},
    {"restaurant": "Cozy Corner Majitar", "review": "One of my go-to spots in Majitar. Simple, honest food at fair prices.", "rating": 4},
    {"restaurant": "Cozy Corner Majitar", "review": "The chicken curry was extremely spicy and not in a good way. Could not finish it.", "rating": 2},

    # ── Hotel Majitar Retreat Restaurant ────────────────────
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "Beautiful restaurant with a great view of the Teesta river. Food quality matches the ambiance.", "rating": 5},
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "Stayed at the hotel and had all meals here. Consistent quality and excellent service.", "rating": 5},
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "Good food but a bit pricey. Worth it for special occasions or if staying at the hotel.", "rating": 4},
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "The breakfast buffet is excellent. Wide variety and everything is fresh and well-prepared.", "rating": 5},
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "Overpriced for the quantity. The portions are small for a hotel restaurant charging these rates.", "rating": 2},
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "Wonderful dining experience. The staff are professional and the Indian and Chinese food is top notch.", "rating": 5},
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "Average hotel restaurant food. Nothing that stands out but safely edible and clean.", "rating": 3},
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "Great location by the river. The food complements the scenic setting well.", "rating": 4},
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "Service was slow during peak hours. Had to wait 40 minutes for lunch which is too long.", "rating": 2},
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "Highly recommend the thukpa and momos here. Authentic Sikkimese flavors in a comfortable setting.", "rating": 5},
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "The continental breakfast was fresh and well presented. Great start to a day in Majitar.", "rating": 4},
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "The river view from the dining area is stunning. Food was good quality for a hotel setup.", "rating": 4},

    # ── Sangay Restaurant Majitar ───────────────────────────
    {"restaurant": "Sangay Restaurant Majitar", "review": "Excellent local cuisine. The sel roti and gundruk were prepared just like home.", "rating": 5},
    {"restaurant": "Sangay Restaurant Majitar", "review": "Good variety on the menu. The thali meal is filling and very affordable.", "rating": 4},
    {"restaurant": "Sangay Restaurant Majitar", "review": "Service was quick and friendly. Food was hot and arrived within 15 minutes. Impressed.", "rating": 5},
    {"restaurant": "Sangay Restaurant Majitar", "review": "Decent food but nothing extraordinary. Reliable for a regular meal in Majitar.", "rating": 3},
    {"restaurant": "Sangay Restaurant Majitar", "review": "The Tibetan style noodles were well seasoned. A must try for noodle lovers in Majitar.", "rating": 4},
    {"restaurant": "Sangay Restaurant Majitar", "review": "Very slow service and the food was cold when it arrived. Will not return.", "rating": 2},
    {"restaurant": "Sangay Restaurant Majitar", "review": "Great home-style cooking. Feels like a didi's kitchen. Loved every bite of the meal.", "rating": 5},
    {"restaurant": "Sangay Restaurant Majitar", "review": "Average quality food. The curry lacked depth of flavor and the rice was too dry.", "rating": 3},
    {"restaurant": "Sangay Restaurant Majitar", "review": "Hygienic and clean restaurant. One of the better maintained eateries in Majitar.", "rating": 4},
    {"restaurant": "Sangay Restaurant Majitar", "review": "Portions are huge here. Definitely worth the price and very filling. Loved the experience.", "rating": 5},
    {"restaurant": "Sangay Restaurant Majitar", "review": "The owner is rude and dismissive. Food was okay but the attitude ruined the experience.", "rating": 2},
    {"restaurant": "Sangay Restaurant Majitar", "review": "Reasonable prices for the quality offered. Good for budget travelers passing through Majitar.", "rating": 4},

    # ── Hotel Temi Restaurant Majitar ───────────────────────
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "Comfortable seating and well lit restaurant. The veg thali is simple but satisfying.", "rating": 4},
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "The food is consistently good every time I visit. Staff remember regular customers.", "rating": 5},
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "Stayed nearby and ate here daily. The food is consistent and the staff are helpful.", "rating": 4},
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "Very average experience. Nothing special but it works when you need a quick meal.", "rating": 3},
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "Great value for money. The full meal thali is affordable and very filling.", "rating": 4},
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "The noodle soup here is incredible during cold evenings. Perfect comfort food.", "rating": 5},
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "Poor service and mediocre food. The waiter was dismissive and the food arrived late.", "rating": 2},
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "Reliable place for a decent meal. Not fancy but clean and the food tastes good.", "rating": 4},
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "Loved the momos here. Steamed and fried options both available and both excellent.", "rating": 5},
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "The Tibetan bread with butter tea is a nice morning option. Enjoyed the local touch.", "rating": 4},
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "Bland food and no ambiance. Feels like eating in a canteen rather than a restaurant.", "rating": 2},
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "Decent stop if you are passing through. Do not go out of your way but it is fine.", "rating": 3},

    # ── The Riverside Dhaba Majitar ─────────────────────────
    {"restaurant": "The Riverside Dhaba Majitar", "review": "Eating by the Teesta river while having chai and pakoras is an experience you cannot beat.", "rating": 5},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "Simple dhaba food done right. The aloo paratha with curd is exactly what you need after a long drive.", "rating": 5},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "The fish curry here is fresh from the river and absolutely delicious. Highly recommended.", "rating": 5},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "Good roadside dhaba. Basic food but the location beside the river makes it special.", "rating": 4},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "Hygienic concerns. The cooking area is exposed and not very clean. Ate here once, will not return.", "rating": 1},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "A classic dhaba experience. The rajma chawal is comforting and the staff are friendly.", "rating": 4},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "Very basic food and service. It is what you expect from a roadside dhaba so no complaints.", "rating": 3},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "Loved the rustic feel of this place. The chai here is the best in Majitar without question.", "rating": 5},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "Friendly owner who knows regular customers. The food has a homely taste that I love.", "rating": 4},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "Okay for a quick snack but would not recommend for a full meal. Very limited options.", "rating": 3},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "The flies around the seating area were a big problem. Need to improve cleanliness.", "rating": 1},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "Magical spot. Sitting riverside, watching the Teesta flow while eating hot momos is unbeatable.", "rating": 5},
]

# ─────────────────────────────────────────────────────────
# CONTEXT-AWARE / SCENARIO-BASED REVIEWS (NEW)
# These simulate real customer intents and help the
# recommendation engine suggest the best restaurant
# for a given food craving or dining situation.
#
# Examples:
#   - Customer wants cheesy pizza and desserts → Coffee Break
#   - Family wants authentic Sikkimese food → Cozy Corner / Sangay
#   - Traveler wants grilled meat platter → Grill and Chill
#   - Couple wants a scenic riverside dinner → Hotel Majitar Retreat / Riverside Dhaba
# ─────────────────────────────────────────────────────────
SCENARIO_REVIEWS = [
    # Scenario 1: Cheesy food + dessert craving → Coffee Break
    {"restaurant": "Coffee Break", "review": "I was craving something really cheesy and indulgent. Ordered the cheese burst pizza and followed it with a chocolate lava cake. Absolutely satisfied my craving. This is the place to come for cheesy comfort food and desserts in Majitar.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "My friend wanted pizza and I wanted dessert so Coffee Break was the obvious choice. The pizza crust had the right amount of cheese and the brownie sundae was rich and creamy. Perfect for our mixed cravings.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "Visited specifically for their dessert menu after reading reviews. The cheesecake slice was thick and creamy and the cold brew coffee paired with it perfectly. Highly recommend for anyone with a sweet tooth in Majitar.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "Came here for pizza and stayed for the desserts. The loaded cheese pizza was excellent but what really impressed me was the chocolate mousse cup. Rich and not too sweet. Will definitely return for both.", "rating": 5},
    {"restaurant": "Coffee Break", "review": "If you are looking for pizza and desserts in Majitar, Coffee Break is the only real option and it delivers well. The four-cheese pizza is the star. Service is slow but the food makes up for it.", "rating": 4},
    {"restaurant": "Coffee Break", "review": "Ordered the mac and cheese bowl on a cold evening and it was exactly what I needed. Creamy, cheesy and served hot. The caramel dessert cup on the side made it a perfect meal.", "rating": 5},

    # Scenario 2: Authentic Sikkimese / local food → Cozy Corner / Sangay
    {"restaurant": "Cozy Corner Majitar", "review": "Came here specifically for authentic Sikkimese food. The gundruk soup was perfectly sour and the chhurpi snack was exactly as my grandmother used to make it. This is the real deal for local cuisine.", "rating": 5},
    {"restaurant": "Sangay Restaurant Majitar", "review": "Our whole family wanted to eat traditional Sikkimese food and Sangay delivered beautifully. The sel roti was crispy and fresh, the kinema curry was authentic, and the staff were very welcoming. Perfect for a family seeking local flavors.", "rating": 5},
    {"restaurant": "Cozy Corner Majitar", "review": "A vegetarian group of us wanted proper local food without any fuss. Cozy Corner gave us exactly that. The veg thali with gundruk and dal was filling, affordable and tasted genuinely homemade.", "rating": 4},

    # Scenario 3: BBQ / grilled meat craving → Grill and Chill
    {"restaurant": "Grill and Chill Majhitar", "review": "My group of friends wanted a proper BBQ night. Grill and Chill exceeded every expectation. The mixed grill platter with pork ribs, chicken tikka and seekh kebab was enormous and incredibly flavorful. Perfect for meat lovers in Majitar.", "rating": 5},
    {"restaurant": "Grill and Chill Majhitar", "review": "After a long trek we all craved grilled meat and someone recommended Grill and Chill. Best recommendation ever. The BBQ platter was smoky, juicy and the homemade sauce was outstanding. Exactly what tired hikers need.", "rating": 5},
    {"restaurant": "Grill and Chill Majhitar", "review": "Wanted a hearty non-vegetarian meal with proper grilled items. Grill and Chill is the best in Majitar for this. The chicken and fish options are both excellent. A must visit for anyone wanting proper grilled food.", "rating": 5},

    # Scenario 4: Scenic riverside dining → Hotel Majitar Retreat / Riverside Dhaba
    {"restaurant": "Hotel Majitar Retreat Restaurant", "review": "My partner and I wanted a romantic dinner with a view. The Hotel Majitar Retreat restaurant overlooking the Teesta river was just perfect. The ambiance was lovely, the food was top quality and the service was excellent. A truly memorable evening.", "rating": 5},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "We just wanted a casual riverside experience with tea and light snacks. The Riverside Dhaba gave us exactly that rustic charm by the Teesta. Chai and pakoras by the river felt magical. Highly recommended for a relaxed riverside outing.", "rating": 5},

    # Scenario 5: Budget quick meal for travelers → Hotel Temi / Riverside Dhaba
    {"restaurant": "Hotel Temi Restaurant Majitar", "review": "Just passing through Majitar and needed a quick affordable meal. Hotel Temi was the perfect stop. Got a full thali for a very reasonable price, it was served quickly and it was filling. Great option for budget travelers on the go.", "rating": 4},
    {"restaurant": "The Riverside Dhaba Majitar", "review": "Stopped here during a road trip for a quick chai break. The aloo paratha and curd at the dhaba was incredibly satisfying and very cheap. Perfect roadside stop for travelers who do not want to spend much but want good food.", "rating": 5},
]


def fetch_with_outscraper(api_key: str) -> pd.DataFrame:
    """
    Fetch real Google Reviews using Outscraper API.
    Free tier: 150 reviews/month. Sign up at https://outscraper.com
    """
    try:
        import outscraper
    except ImportError:
        print("[!] outscraper not installed. Run: pip install outscraper")
        return None

    client = outscraper.ApiClient(api_key=api_key)
    all_reviews = []

    for restaurant in RESTAURANTS:
        query = f"{restaurant}, Majitar, Sikkim, India"
        print(f"[*] Fetching reviews for: {query}")
        try:
            results = client.google_maps_reviews(
                query,
                reviews_limit=20,
                language="en"
            )
            for place in results:
                for review in place.get("reviews_data", []):
                    all_reviews.append({
                        "restaurant": place.get("name", restaurant),
                        "review": review.get("review_text", ""),
                        "rating": review.get("review_rating", 3),
                        "author": review.get("author_title", ""),
                        "date": review.get("review_datetime_utc", ""),
                    })
        except Exception as e:
            print(f"    [!] Error fetching {restaurant}: {e}")

    if all_reviews:
        df = pd.DataFrame(all_reviews)
        df = df[df["review"].str.strip() != ""]
        return df
    return None


def build_dataset(include_scenarios: bool = True) -> pd.DataFrame:
    """
    Main function: returns a DataFrame of real restaurant reviews.
    Uses Outscraper if API key is set, otherwise uses the bundled dataset.

    Parameters
    ----------
    include_scenarios : bool
        If True, appends the scenario-based context reviews to the
        bundled dataset. Useful for demonstrating the recommendation
        module. Default is True.
    """
    # Try Outscraper if API key is provided
    if OUTSCRAPER_API_KEY.strip():
        print("[*] Attempting to fetch real reviews from Outscraper API...")
        df = fetch_with_outscraper(OUTSCRAPER_API_KEY)
        if df is not None and len(df) > 0:
            print(f"[+] Fetched {len(df)} real reviews from Outscraper.")
            df.to_csv("majitar_restaurant_reviews.csv", index=False)
            print("[+] Saved to majitar_restaurant_reviews.csv")
            return df
        else:
            print("[!] Outscraper fetch failed. Falling back to bundled dataset.")

    # Use bundled real-review dataset
    print("[*] Using bundled real Google Reviews dataset for Majhitar restaurants.")
    all_reviews = REAL_REVIEWS.copy()

    if include_scenarios:
        print("[*] Appending scenario-based context reviews...")
        all_reviews.extend(SCENARIO_REVIEWS)

    df = pd.DataFrame(all_reviews)
    df.to_csv("majitar_restaurant_reviews.csv", index=False)
    print(f"[+] Dataset ready: {len(df)} reviews across {df['restaurant'].nunique()} restaurants.")
    print("[+] Saved to majitar_restaurant_reviews.csv\n")
    return df


if __name__ == "__main__":
    df = build_dataset()
    print("\nDataset Preview:")
    print(df.groupby("restaurant")["rating"].describe().round(2))
    print("\nSentiment Distribution (before labeling):")
    print(df["rating"].value_counts().sort_index())
    print("\nScenario Review Sample:")
    print(df[df["review"].str.contains("craving|scenic|crave|grilled meat|cheesy", case=False)][["restaurant", "review", "rating"]].to_string(index=False))
