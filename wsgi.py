import os
import sys
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

webapp_dir = os.path.join(BASE_DIR, "module5(WebApp)")
if webapp_dir not in sys.path:
    sys.path.insert(0, webapp_dir)

app_path = os.path.join(webapp_dir, "app.py")
spec = importlib.util.spec_from_file_location("app", app_path)
webapp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(webapp_module)

app = webapp_module.app

if __name__ == "__main__":
    app.run()
