"""Open the handlab world.

    python main.py

No arguments: everything comes from settings.json, so the world is the same
however it is opened.
"""

from src.app import App

if __name__ == "__main__":
    try:
        App().run()
    except KeyboardInterrupt:
        # App.run has already closed the port on its way out.
        print("\nstopped")
