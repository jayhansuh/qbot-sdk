def check_binance_availability():
    try:
        from binance.client import Client

        client = Client()
        client.ping()
        return False  # Don't skip if successful
    except Exception as e:
        # Skip if any error occurs with Binance client
        return True
