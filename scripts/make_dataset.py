#!/usr/bin/env python3
"""Generate the offline synthetic listings dataset (data/raw/).

The synthetic feed stands in for the bina.az scrape (which requires network
access and Cloudflare handling - see scripts/scrape_bina.py). All planted
ground-truth parameters are recorded next to the data so that downstream
recovery can be audited.
"""

from bakuml.data.synthetic import save_synthetic


def main() -> None:
    listings_path, truth_path = save_synthetic()
    print(f"listings -> {listings_path}")
    print(f"planted truth -> {truth_path}")


if __name__ == "__main__":
    main()
