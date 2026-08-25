"""
main.py

Development entry point: `python main.py`.

The application itself lives in src/app.py so that it ships inside the
package. A wheel installs into site-packages, where a project root -- and
anything sitting beside it -- does not exist; a module inside the package is
the only location that resolves in both a checkout and an install.
"""

from src.app import main

if __name__ == "__main__":
    main()
