from web_base import WebChat


app = WebChat()


@app.tool
def add(x: float, y: float) -> float:
    """Add two numbers"""
    return x + y


if __name__ == "__main__":
    app.run()
