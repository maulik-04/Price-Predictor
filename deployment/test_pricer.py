
import modal

app = modal.App("test-pricer")

@app.local_entrypoint()
def main():
    Pricer = modal.Cls.from_name("pricer-service", "Pricer")
    pricer = Pricer()

    test_items = [
        "Title: Sony WH-1000XM4 Wireless Headphones\nCategory: Electronics\nBrand: Sony\nDescription: Industry leading noise cancelling headphones with 30 hour battery\nDetails: Touch sensor controls, speak to chat, alexa built in",

        "Title: KitchenAid Stand Mixer 5Qt\nCategory: Appliances\nBrand: KitchenAid\nDescription: Professional 5 quart stand mixer for home baking\nDetails: 10 speed settings, includes dough hook, flat beater, wire whip",

        "Title: Lego Technic Bugatti Chiron\nCategory: Toys and Games\nBrand: Lego\nDescription: Advanced building set with 3599 pieces replicating the Bugatti Chiron\nDetails: Working steering, W16 engine with moving pistons, top speed indicator",
    ]

    expected = [280, 350, 350]

    print("Testing pricer service...\n")
    for i, (item, exp) in enumerate(zip(test_items, expected)):
        print(f"Sending item {i+1}...")
        price = pricer.price.remote(item)
        error = abs(price - exp)
        print(f"Item {i+1}: {item[:50]}...")
        print(f"  Expected:  ~${exp}")
        print(f"  Predicted:  ${price:.2f}")
        print(f"  Error:      ${error:.2f}")
        print()

    print("Pricer service working!")
