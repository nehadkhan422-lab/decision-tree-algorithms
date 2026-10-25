from cart import run_cart
from id3 import run_id3
from c45 import run_c45

print("Running all three algorithms...\n")

results = {
    "ID3": run_id3(),
    "C4.5": run_c45(),
    "CART": run_cart(),
}

print("\n\n=== FINAL COMPARISON ===")
for name, acc in results.items():
    print(f"{name:6s} accuracy: {acc:.4f}")